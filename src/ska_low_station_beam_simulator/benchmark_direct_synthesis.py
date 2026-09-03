# %% [markdown]
# # Benchmark: `DirectSynthesisStreamer.generate_next_tick`
#
# Companion to benchmark.py, which benchmarks the LEGACY wideband+FFT
# StationStreamer (wideband_streamer.py). This one targets
# direct_synthesis.DirectSynthesisStreamer — no ring buffer, no FFT
# channelization, tone delay applied as a continuous phase term. Per the
# handoff doc, the whole point of this path is that its per-tick cost
# should scale much more weakly with NUM_CHANNELS than the legacy path's
# does — this script benchmarks BOTH 96 (current, 75 MHz) and 448 (full
# band, 350 MHz) channel counts specifically to check that claim on real
# multi-core hardware, not just the single-threaded sandbox comparison
# baked into direct_synthesis.py's __main__ block.
#
# Per-tick TIME BUDGET is fixed regardless of channel count
# (HEAP_LEN / CHANNEL_WIDTH_HZ ~= 2.621ms) — see CLAUDE.md's "Scaling math"
# note. n_samples per tick for this streamer is always HEAP_LEN, by the same
# construction ScanRunner relies on.
#
# Uses a fake delay polynomial (the real fetch_delay_model_from_cbf is a
# stub that raises NotImplementedError) — benchmarks generation cost only,
# not a real Tango delay-poly client's latency.

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


# %%
def build_streamer(num_channels: int) -> sim.DirectSynthesisStreamer:
    station = sim.StationConfig(
        station_id=1, substation_id=0, subarray_id=1, beam_id=1, first_channel_id=0, scan_id=99
    )
    # One tone per channel-count case, deliberately off-center within a
    # channel (matches direct_synthesis.py's own correctness-check
    # convention) — not load-bearing for the timing measurement, just avoids
    # a suspiciously "nice" freq_hz that happens to land exactly on a bin edge.
    tone_freq = 20 * sim.CHANNEL_WIDTH_HZ + 150_000.0
    return sim.DirectSynthesisStreamer(
        station=station,
        source_cfgs=[{"kind": "tone", "freq_hz": tone_freq, "amplitude": 1.0}],
        noise_cfg={"std": 0.05, "seed": 7},
        obs_time_ref=1_800_000_000.0,
        num_channels=num_channels,
    )


def benchmark_tick(
    streamer: sim.DirectSynthesisStreamer,
    n_samples: int,
    n_warmup: int = 5,
    n_measured: int = 30,
) -> dict:
    """n_warmup also absorbs numba JIT compilation cost for the first call
    to each kernel — same discipline as benchmark.py's benchmark_tick."""
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
# --- Sweep numba thread count, at each of the two target channel counts ---

n_samples = HEAP_LEN  # fixed by construction, independent of channel count
budget_ms = (HEAP_LEN / sim.CHANNEL_WIDTH_HZ) * 1000
max_numba_threads = numba.config.NUMBA_NUM_THREADS

print(
    f"n_samples/tick: {n_samples}   budget: {budget_ms:.3f} ms/tick   "
    f"numba thread ceiling: {max_numba_threads}\n"
)

candidate_counts = sorted(set([1, 2, 4, 8, max_numba_threads // 2, max_numba_threads]))
candidate_counts = [n for n in candidate_counts if 1 <= n <= max_numba_threads]

all_results: dict[int, dict[int, dict]] = {}
for num_channels, label in [(96, "current (75 MHz)"), (448, "full band (350 MHz)")]:
    print("=" * 60)
    print(f"{label}, NUM_CHANNELS={num_channels}")
    print("=" * 60)

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
# --- Cross-channel-count comparison at the same thread count, to check the
# "flat cost vs. channel count" scaling claim directly on this hardware ---

print("=" * 60)
print("SCALING CHECK: 96 -> 448 channels, at each numba_threads value")
print("=" * 60)
for n_threads in candidate_counts:
    m96 = all_results[96][n_threads]["mean_ms"]
    m448 = all_results[448][n_threads]["mean_ms"]
    print(
        f"numba_threads={n_threads:>3}  96ch={m96:7.3f} ms  448ch={m448:7.3f} ms  "
        f"ratio={m448 / m96:.2f}x  (legacy wideband+FFT path would be ~4.67x+ here)"
    )

# %%
# --- Repeatability check at the best config for EACH channel count ---
# Single-pass sweep results have not been trustworthy on this class of
# hardware in earlier rounds of benchmarking (see CLAUDE.md) — confirm
# before treating any of the above as a real number.

print("\n" + "=" * 60)
print("REPEATABILITY CHECK")
print("=" * 60)

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
