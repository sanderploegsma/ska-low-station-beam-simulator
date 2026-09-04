"""
Experimental variant of DirectSynthesisStreamer: pre-generate a bank of
N noise "tiles" once, up front, instead of generating fresh Box-Muller
noise every tick. Each tick then does an O(1) index draw into that bank
plus a memcopy, independent of channel count and (mostly) independent of
CPU budget -- see this module's __main__ for the benchmark this was
built to answer: "can a station pod get away with ~8 CPU cores at 448
channels instead of the ~24-48 DirectSynthesisStreamer needs?"

THIS IS NOT A DROP-IN REPLACEMENT for DirectSynthesisStreamer. It is a
deliberate fidelity/resource-usage tradeoff, kept in its own module
specifically so the two are never confused:

  - MEMORY: the bank is n_tiles * tile_n_samples * num_channels *
    16 bytes (complex128), per polarization. See bank_memory_bytes()
    and the sweep in __main__ for concrete numbers at 448 channels.

  - REPEATS (the real cost of this approach): by the birthday paradox, a
    single station's OWN tile-index sequence hits its first repeat after
    roughly 1.25*sqrt(n_tiles) ticks. For any memory-feasible n_tiles (a
    few hundred to a few thousand -- see the sweep), that's seconds, not
    minutes -- a station will replay its small fixed set of tiles many
    times over a real scan. This is fine for tests that only care about
    per-tick and cross-station statistics (delay-tracking, correlation,
    beamforming functional correctness) but WRONG for any test that
    checks a single station's long-integration noise-floor behaviour
    (total power should keep averaging down with more integration time
    -- it won't, past the bank's repeat cycle). Confirm this doesn't
    matter for CBF's actual test suite before using this in place of
    DirectSynthesisStreamer -- see CLAUDE.md's noise-strategy discussion.

  - CROSS-STATION INDEPENDENCE IS PRESERVED BY CONSTRUCTION: each
    station's tile index is drawn from that station's own (station, pol)
    noise seed, exactly like DirectSynthesisStreamer's noise today --
    NEVER from a value shared across stations. Sharing the index across
    stations (e.g. seeding the draw from obs_time alone) would make every
    station emit byte-identical "noise" for a given tick, which silently
    breaks beamforming-SNR and cross-correlation tests that specifically
    rely on receiver noise being uncorrelated between stations. Do not
    "fix" the seeding to be simpler without re-reading this paragraph.

Reuses direct_synthesis.py's tone kernel and Box-Muller/splitmix64
internals rather than re-deriving them -- this module is an experimental
variant OF the direct-synthesis backend, not a third independent backend
in the sense wideband_streamer.py and direct_synthesis.py are kept
independent of each other (see direct_synthesis.py's module docstring
for why THAT pair doesn't share code).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numba import njit, prange

from ska_low_station_beam_simulator.common import (
    BASE_FREQ_HZ,
    BLOCK_DURATION_S,
    CHANNEL_WIDTH_HZ,
    DelayPolynomial,
    NUM_CHANNELS,
    StationConfig,
    fetch_delay_model_from_cbf,
    log,
)
from ska_low_station_beam_simulator.direct_synthesis import (
    _splitmix64_hash,
    synth_noise_all_channels_into,
    synth_tone_channel,
)

DEFAULT_N_TILES = 256


def bank_memory_bytes(n_tiles: int, tile_n_samples: int, num_channels: int, n_pols: int = 2) -> int:
    """Total resident memory for the noise bank across both pols."""
    return n_tiles * tile_n_samples * num_channels * 16 * n_pols  # complex128 = 16 bytes


@njit(parallel=True, cache=True)
def _copy_tile_into(out, bank, tile_idx):
    """out[:] = bank[tile_idx], parallelized over rows -- lets the
    benchmark test whether spreading the copy across threads matters at
    all once generation itself is no longer in the per-tick path."""
    n_samples = out.shape[0]
    num_channels = out.shape[1]
    for i in prange(n_samples):
        for ch in range(num_channels):
            out[i, ch] = bank[tile_idx, i, ch]


class TiledNoiseStreamer:
    """Same external contract as DirectSynthesisStreamer (tone + per-pol
    noise, dict[pol] -> (n_samples, num_channels) complex128 out), but
    noise comes from a pre-generated tile bank instead of fresh
    Box-Muller generation. See module docstring for the tradeoff."""

    def __init__(
        self,
        station: StationConfig,
        source_cfgs: list[dict],
        obs_time_ref: float,
        noise_cfg: Optional[dict] = None,
        num_channels: int = NUM_CHANNELS,
        base_freq_hz: float = BASE_FREQ_HZ,
        channel_width_hz: float = CHANNEL_WIDTH_HZ,
        n_tiles: int = DEFAULT_N_TILES,
        tile_n_samples: Optional[int] = None,
        parallel_copy: bool = False,
    ):
        for cfg in source_cfgs:
            if cfg["kind"] != "tone":
                raise ValueError(
                    f"TiledNoiseStreamer only supports kind='tone' in "
                    f"source_cfgs, same restriction as DirectSynthesisStreamer; "
                    f"got kind={cfg['kind']!r}."
                )

        self.station = station
        self.num_channels = num_channels
        self.base_freq_hz = base_freq_hz
        self.channel_width_hz = channel_width_hz
        self.channel_output_rate = channel_width_hz

        self._obs_time_ref = obs_time_ref
        self._tone_cfgs = source_cfgs

        self._noise_cfg = noise_cfg
        self._noise_seed_v = noise_cfg["seed"] if noise_cfg else 0
        self._noise_seed_h = (noise_cfg["seed"] + 1_000_003) if noise_cfg else 0
        self._noise_std = noise_cfg["std"] if noise_cfg else 0.0

        self._current_poly: Optional[DelayPolynomial] = None
        self._delay_coeffs: Optional[np.ndarray] = None

        self.n_tiles = n_tiles
        self.tile_n_samples = tile_n_samples or self.tick_n_samples()
        self.parallel_copy = parallel_copy

        self._banks: dict[str, np.ndarray] = {}
        if noise_cfg is not None:
            for pol, seed in (("V", self._noise_seed_v), ("H", self._noise_seed_h)):
                bank = np.empty(
                    (self.n_tiles, self.tile_n_samples, self.num_channels), dtype=np.complex128
                )
                for tile_idx in range(self.n_tiles):
                    synth_noise_all_channels_into(
                        bank[tile_idx],
                        seed,
                        self._noise_std,
                        tile_idx * self.tile_n_samples,  # each tile is independent content
                        self.num_channels,
                        self.tile_n_samples,
                    )
                self._banks[pol] = bank

        self._out_bufs: dict[str, np.ndarray] = {}

    def _get_output_buffer(self, pol: str, n_samples: int) -> np.ndarray:
        buf = self._out_bufs.get(pol)
        if buf is None or buf.shape != (n_samples, self.num_channels):
            buf = np.empty((n_samples, self.num_channels), dtype=np.complex128)
            self._out_bufs[pol] = buf
        return buf

    @property
    def channel_id_map(self) -> np.ndarray:
        return np.arange(self.num_channels)

    def tick_n_samples(self) -> int:
        return int(round(self.channel_output_rate * BLOCK_DURATION_S))

    def _refresh_delay_poly_if_needed(self, t: float):
        if self._current_poly is None or t >= self._current_poly.valid_until:
            self._current_poly = fetch_delay_model_from_cbf(self.station.station_id, t)
            self._delay_coeffs = np.asarray(
                self._current_poly.xypol_coeffs_ns, dtype=np.float64
            )

    def bank_memory_bytes(self) -> int:
        if not self._banks:
            return 0
        return bank_memory_bytes(self.n_tiles, self.tile_n_samples, self.num_channels, len(self._banks))

    def generate_next_tick(self, t: float, n_samples: int) -> dict[str, np.ndarray]:
        self._refresh_delay_poly_if_needed(t)
        poly = self._current_poly

        t_local_rel_start = t - self._obs_time_ref
        poly_t_rel_start = t - poly.start_validity_sec

        # Tick index, NOT tied to n_samples matching tile_n_samples -- the
        # bank is drawn from at whole-tile granularity regardless of how
        # many samples a single generate_next_tick call asks for, so this
        # only behaves sensibly (as intended) when n_samples ==
        # tile_n_samples, which is what ScanRunner always uses in
        # production (see tick_n_samples()).
        tick_index = int(round(t_local_rel_start * self.channel_output_rate)) // max(n_samples, 1)

        results: dict[str, np.ndarray] = {}
        for pol, is_h_pol, noise_seed in (
            ("V", False, self._noise_seed_v),
            ("H", True, self._noise_seed_h),
        ):
            out = self._get_output_buffer(pol, n_samples)

            if self._noise_cfg is not None:
                bank = self._banks[pol]
                # Per-(station, pol) independent draw -- NEVER shared
                # across stations. See module docstring.
                tile_idx = int(_splitmix64_hash(noise_seed, tick_index) % self.n_tiles)
                if self.parallel_copy:
                    _copy_tile_into(out, bank, tile_idx)
                else:
                    out[:] = bank[tile_idx]
            else:
                out.fill(0)

            for cfg in self._tone_cfgs:
                ch_idx, samples = synth_tone_channel(
                    cfg["freq_hz"],
                    cfg.get("amplitude", 1.0),
                    self.base_freq_hz,
                    self.channel_width_hz,
                    self._delay_coeffs,
                    poly_t_rel_start,
                    poly.ypol_offset_ns,
                    is_h_pol,
                    t_local_rel_start,
                    self.channel_output_rate,
                    n_samples,
                )
                if not (0 <= ch_idx < self.num_channels):
                    log.warning(
                        "tone freq_hz=%s maps to channel_idx=%d, outside the "
                        "configured [0, %d) channel range — skipping",
                        cfg["freq_hz"],
                        ch_idx,
                        self.num_channels,
                    )
                    continue
                out[:, ch_idx] += samples

            results[pol] = out

        return results


if __name__ == "__main__":
    # ============================================================
    # CORRECTNESS CHECKS
    # ============================================================
    def _fake_fetch(station_id, at_time):
        return DelayPolynomial(
            station_id=station_id,
            start_validity_sec=at_time,
            validity_period_sec=600.0,
            xypol_coeffs_ns=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ypol_offset_ns=0.0,
        )

    # Patch the name in *this* module's own globals -- when run as
    # `python -m ...`, this file is loaded as __main__, a distinct module
    # object from a fresh `import ska_low_station_beam_simulator.tiled_noise_streamer`,
    # so patching via a self-import would silently miss the class's lookup.
    globals()["fetch_delay_model_from_cbf"] = _fake_fetch

    station = StationConfig(
        station_id=1, substation_id=0, subarray_id=1, beam_id=1, first_channel_id=0, scan_id=1
    )
    streamer = TiledNoiseStreamer(
        station=station,
        source_cfgs=[],
        noise_cfg={"std": 1.0, "seed": 7},
        obs_time_ref=1_800_000_000.0,
        num_channels=96,
        n_tiles=8,
    )
    n_samples = streamer.tick_n_samples()
    tick_dt = n_samples / streamer.channel_output_rate
    obs_time = 1_800_000_000.0

    # determinism/seekability: same t -> same content
    r1 = streamer.generate_next_tick(obs_time, n_samples)["V"].copy()
    r2 = streamer.generate_next_tick(obs_time, n_samples)["V"].copy()
    assert np.array_equal(r1, r2), "same t must give identical content"
    print("determinism/seekability: OK")

    # repeat behaviour: with n_tiles=8, expect a repeat well within a
    # couple dozen ticks (demonstrates the documented tradeoff concretely)
    seen = {}
    first_repeat_at = None
    for i in range(200):
        t = obs_time + i * tick_dt
        out = streamer.generate_next_tick(t, n_samples)["V"]
        key = out[0, 0]
        if key in seen and first_repeat_at is None:
            first_repeat_at = i
        seen[key] = i
    print(f"first exact repeat (n_tiles=8): tick {first_repeat_at} (expect early, ~dozens)")

    # cross-station independence: two stations with different seeds must
    # NOT pick the same tile at the same tick (spot check, not a proof)
    streamer_b = TiledNoiseStreamer(
        station=StationConfig(station_id=2, substation_id=0, subarray_id=1, beam_id=1, first_channel_id=0, scan_id=1),
        source_cfgs=[],
        noise_cfg={"std": 1.0, "seed": 99},
        obs_time_ref=1_800_000_000.0,
        num_channels=96,
        n_tiles=8,
    )
    out_a = streamer.generate_next_tick(obs_time, n_samples)["V"]
    out_b = streamer_b.generate_next_tick(obs_time, n_samples)["V"]
    assert not np.array_equal(out_a, out_b), "different stations must not emit identical noise"
    print("cross-station independence (different seeds -> different content): OK")

    print("\nAll correctness checks passed.")
