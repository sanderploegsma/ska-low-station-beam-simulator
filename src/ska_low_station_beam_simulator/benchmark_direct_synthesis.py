# %% [markdown]
# # Benchmark: `DirectSynthesisStreamer` — combined tone + noise + pulsar
#
# DirectSynthesisStreamer is now the sole backend, converging what used
# to be three separate prototypes (tone from this module's original
# scope, a pre-generated noise tile bank, and a coherent per-channel
# pulsar) plus deleting the legacy wideband+FFT StationStreamer entirely.
# Each piece was benchmarked in isolation in earlier sessions (see
# CLAUDE.md's Benchmarking section) and individually cleared an ~8-core
# target at 448 channels -- this benchmark checks whether that still
# holds when all three run TOGETHER in one streamer, which is the
# combination a real scan is likely to actually use.
#
# Per-tick TIME BUDGET is fixed regardless of channel count
# (HEAP_LEN / CHANNEL_WIDTH_HZ ~= 2.621ms). n_samples per tick is always
# HEAP_LEN, by construction (ScanRunner relies on this).
#
# Uses a fake delay polynomial (the real fetch_delay_model_from_cbf is a
# stub that raises NotImplementedError) — benchmarks generation cost
# only, not a real Tango delay-poly client's latency.

# %%
import statistics
import time

import numba

import ska_low_station_beam_simulator.direct_synthesis as sim
from ska_low_station_beam_simulator.common import HEAP_LEN

# %%
def _fake_fetch_delay_model(station_id: int, at_time: float) -> sim.DelayPolynomial:
    return sim.DelayPolynomial(
        station_id=station_id,
        start_validity_sec=at_time,
        validity_period_sec=600.0,
        xypol_coeffs_ns=[750.0, 0.0046, 0.0, 0.0, 0.0, 0.0],
        ypol_offset_ns=2.0,
    )


sim.fetch_delay_model_from_cbf = _fake_fetch_delay_model

# Modest pulsar/tile-bank settings so streamer CONSTRUCTION (rebuilt once
# per sweep point below) stays fast -- period_s and n_tiles both trade
# one-time build time for fidelity (see CLAUDE.md); this benchmark cares
# about per-tick STEADY-STATE cost, so construction is kept cheap on
# purpose, not tuned for realism here.
PULSAR_PERIOD_S = 0.01  # 10ms
PULSAR_WIDTH_S = 0.0005
PULSAR_DM = 2.0
N_TILES = 256


def build_streamer(num_channels: int) -> sim.DirectSynthesisStreamer:
    station = sim.StationConfig(
        station_id=1, substation_id=0, subarray_id=1, beam_id=1, first_channel_id=0, scan_id=99
    )
    # Offset from base_freq_hz, NOT absolute -- the streamer's channel
    # mapping is round((freq_hz - base_freq_hz) / channel_width_hz), and
    # this benchmark uses DEFAULT_PULSAR_BASE_FREQ_HZ (50MHz) as the band
    # start, not 0, since pulsed sources require a nonzero base_freq_hz.
    tone_freq = sim.DEFAULT_PULSAR_BASE_FREQ_HZ + 20 * sim.CHANNEL_WIDTH_HZ + 150_000.0
    return sim.DirectSynthesisStreamer(
        station=station,
        source_cfgs=[
            {"kind": "tone", "freq_hz": tone_freq, "amplitude": 1.0},
            {"kind": "pulsed", "period_s": PULSAR_PERIOD_S, "width_s": PULSAR_WIDTH_S,
             "amplitude": 1.0, "dm_pc_cm3": PULSAR_DM},
        ],
        noise_cfg={"std": 0.05, "seed": 7},
        obs_time_ref=1_800_000_000.0,
        num_channels=num_channels,
        base_freq_hz=sim.DEFAULT_PULSAR_BASE_FREQ_HZ,  # required for pulsed sources
        n_tiles=N_TILES,
    )


def benchmark_tick(
    streamer: sim.DirectSynthesisStreamer,
    n_samples: int,
    n_warmup: int = 5,
    n_measured: int = 30,
) -> dict:
    """n_warmup also absorbs numba JIT compilation cost for the first call
    to each kernel."""
    obs_time = 1_800_000_000.0
    tick_dt = n_samples / streamer.channel_output_rate

    for i in range(n_warmup):
        streamer.generate_next_tick(obs_time + i * tick_dt, n_samples)

    samples_ms = []
    for i in range(n_warmup, n_warmup + n_measured):
        t0 = time.perf_counter()
        streamer.generate_next_tick(obs_time + i * tick_dt, n_samples)
        samples_ms.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": statistics.mean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "stdev_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


# %%
# --- Sweep numba thread count, at each of the two target channel counts,
# with tone + tiled noise + pulsar all active together ---

n_samples = HEAP_LEN
budget_ms = (HEAP_LEN / sim.CHANNEL_WIDTH_HZ) * 1000
max_numba_threads = numba.config.NUMBA_NUM_THREADS

print(
    f"n_samples/tick: {n_samples}   budget: {budget_ms:.3f} ms/tick   "
    f"numba thread ceiling: {max_numba_threads}\n"
    f"pulsar: period={PULSAR_PERIOD_S*1000:.0f}ms width={PULSAR_WIDTH_S*1000:.2f}ms DM={PULSAR_DM}   "
    f"noise: n_tiles={N_TILES}\n"
)

candidate_counts = sorted(set([1, 2, 4, 8, 16, 24, 32, max_numba_threads // 2, max_numba_threads]))
candidate_counts = [n for n in candidate_counts if 1 <= n <= max_numba_threads]

all_results: dict[int, dict[int, dict]] = {}
for num_channels, label in [(96, "current (75 MHz)"), (448, "full band (350 MHz)")]:
    print("=" * 70)
    print(f"{label}, NUM_CHANNELS={num_channels} -- tone + tiled noise + pulsar combined")
    print("=" * 70)

    results = {}
    for n_threads in candidate_counts:
        numba.set_num_threads(n_threads)
        streamer = build_streamer(num_channels)
        stats = benchmark_tick(streamer, n_samples)
        results[n_threads] = stats
        within_budget = "OK" if stats["mean_ms"] <= budget_ms else "OVER BUDGET"
        pct = stats["mean_ms"] / budget_ms * 100
        print(
            f"numba_threads={n_threads:>3}  mean={stats['mean_ms']:7.3f} ms "
            f"({pct:5.1f}% of budget)  median={stats['median_ms']:7.3f} ms  "
            f"stdev={stats['stdev_ms']:6.3f}  max={stats['max_ms']:7.3f} ms  [{within_budget}]"
        )

    all_results[num_channels] = results

    best_threads = min(results, key=lambda k: results[k]["mean_ms"])
    baseline = results[1]["mean_ms"]
    speedup = baseline / results[best_threads]["mean_ms"]
    print(
        f"\nsequential baseline (numba_threads=1): {baseline:.3f} ms/tick\n"
        f"best config: numba_threads={best_threads} -> "
        f"{results[best_threads]['mean_ms']:.3f} ms/tick ({speedup:.2f}x vs sequential)\n"
    )

# %%
# --- ~8-core target check, 448 channels, combined workload ---
print("=" * 70)
print("TARGET CHECK: ~8 CPU cores/pod, 448 channels, tone+noise+pulsar combined")
print("=" * 70)
numba.set_num_threads(8)
streamer = build_streamer(448)
stats = benchmark_tick(streamer, n_samples, n_warmup=10, n_measured=50)
pct = stats["mean_ms"] / budget_ms * 100
flag = "OK" if stats["mean_ms"] <= budget_ms else "OVER BUDGET"
print(
    f"8 threads: mean={stats['mean_ms']:.4f}ms ({pct:.1f}% of budget), "
    f"max={stats['max_ms']:.4f}ms  [{flag}]"
)
print(f"noise tile-bank memory (both pols): {streamer.bank_memory_bytes()/1e9:.3f} GB")

# %%
# --- Repeatability check at the best config for EACH channel count ---
# Single-pass sweep results have not been trustworthy on this class of
# hardware without repeat checks (see CLAUDE.md) — confirm before
# treating any of the above as a real number.

print("\n" + "=" * 70)
print("REPEATABILITY CHECK")
print("=" * 70)

n_repeats = 5
for num_channels in (96, 448):
    best_threads = min(all_results[num_channels], key=lambda k: all_results[num_channels][k]["mean_ms"])
    numba.set_num_threads(best_threads)
    print(f"\nNUM_CHANNELS={num_channels}, numba_threads={best_threads}:")
    repeat_means = []
    for rep in range(n_repeats):
        streamer = build_streamer(num_channels)
        stats = benchmark_tick(streamer, n_samples, n_warmup=5, n_measured=20)
        repeat_means.append(stats["mean_ms"])
        print(f"  repeat {rep + 1}: mean={stats['mean_ms']:7.3f} ms")
    mean_of_means = statistics.mean(repeat_means)
    print(
        f"  spread: min={min(repeat_means):.3f}  max={max(repeat_means):.3f}  "
        f"stdev={statistics.stdev(repeat_means):.3f}  mean={mean_of_means:.3f}  "
        f"({mean_of_means / budget_ms * 100:.1f}% of budget)"
    )
