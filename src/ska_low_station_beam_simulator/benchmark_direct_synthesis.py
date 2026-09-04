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
# Every source needs its own DelayFeed (see common.py -- there is no
# default/fallback delay). Uses fixed fake polynomials, applied once at
# construction via .update(); benchmarks generation cost only, not a real
# Tango delay-poly subscription's latency.

# %%
import statistics
import time

import numba

import ska_low_station_beam_simulator.direct_synthesis as sim
from ska_low_station_beam_simulator.common import DelayFeed, HEAP_LEN

# %%
def _fixed_delay_feed(name: str) -> DelayFeed:
    feed = DelayFeed(name=name)
    feed.update(
        sim.DelayPolynomial(
            station_id=1,
            start_validity_sec=1_800_000_000.0,
            validity_period_sec=600.0,
            xypol_coeffs_ns=[750.0, 0.0046, 0.0, 0.0, 0.0, 0.0],
            ypol_offset_ns=2.0,
        )
    )
    return feed


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
    # this benchmark uses the default BASE_FREQ_HZ (the confirmed
    # 50.78125MHz lowest valid SKA-Low frequency) as the band start.
    tone_freq = sim.BASE_FREQ_HZ + 20 * sim.CHANNEL_WIDTH_HZ + 150_000.0
    return sim.DirectSynthesisStreamer(
        station=station,
        source_cfgs=[
            {"kind": "tone", "freq_hz": tone_freq, "amplitude": 1.0,
             "delay_feed": _fixed_delay_feed("bench-tone")},
            {"kind": "pulsed", "period_s": PULSAR_PERIOD_S, "width_s": PULSAR_WIDTH_S,
             "amplitude": 1.0, "dm_pc_cm3": PULSAR_DM,
             "delay_feed": _fixed_delay_feed("bench-pulsar")},
        ],
        noise_cfg={"std": 0.05, "seed": 7},
        obs_time_ref=1_800_000_000.0,
        num_channels=num_channels,
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
# --- ONE-TIME CONSTRUCTION budget check ---
# Target 10s, hard limit 30s, per-station, before a scan can start.
# Noise-bank fill and the pulsar's wideband sky-carrier are now plain
# (parallelized) numpy rather than numba -- see direct_synthesis.py's
# NUMBA section -- so this checks that replacement actually meets budget.
#
# Kept deliberately ISOLATED (noise-only sweep, then pulsar-only sweep)
# rather than combined at the largest sizes of both at once: a combined
# n_tiles=1024 + a long pulsar period run was tried and used enough
# transient memory (large retained noise banks + a large pulsar wideband/
# FFT working set, concurrently) to threaten node stability on this
# shared host -- caught by a `ulimit -v` safety net, not by reasoning
# about it beforehand. Revisit together only under a memory budget, not
# just a time budget, if a real deployment actually needs both large at
# once.
#
# The pulsar sweep below is deliberately capped at 300ms, NOT because
# longer periods are uninteresting but because a period around 1s was
# separately confirmed to risk node memory (see CLAUDE.md's Pulsed
# sources section) -- re-test longer periods only under an explicit
# `ulimit -v` safety net, never bare on a shared host. 50ms is included
# ON PURPOSE, not swept past: `build_pulsar_template`'s cost depends far
# more on whether the wideband array length happens to factor into small
# primes than on period length itself -- 50ms's array has a large prime
# factor (19531) and measures ~3-6x slower than the better-factored
# 100/200/300ms points despite being the SMALLEST array here. This is a
# permanent regression check for that finding, not an oversight if the
# ordering below looks non-monotonic.
print("=" * 70)
print("ONE-TIME CONSTRUCTION BUDGET (target 10s, hard limit 30s), 448 channels")
print("=" * 70)

print("-- noise tile-bank fill only (fill_noise_bank, both pols) --")
for n_tiles_check in (256, 512, 1024):
    t0 = time.perf_counter()
    for _seed in (7, 1_000_010):  # V, H -- same as DirectSynthesisStreamer's two calls
        sim.fill_noise_bank(_seed, 0.05, n_tiles_check, HEAP_LEN, 448)
    build_s = time.perf_counter() - t0
    mem_gb = sim.bank_memory_bytes(n_tiles_check, HEAP_LEN, 448) / 1e9
    flag = "OK" if build_s <= 10.0 else ("OVER 10s TARGET" if build_s <= 30.0 else "OVER 30s HARD LIMIT")
    print(f"n_tiles={n_tiles_check:>4}  bank_mem={mem_gb:6.2f}GB  build={build_s:7.3f}s  [{flag}]")

print("\n-- pulsar wideband sky-carrier + dispersion + channelize only (build_pulsar_template) --")
for period_s in (0.01, 0.05, 0.1, 0.2, 0.3):
    n_wide_check = 448 * int(round(period_s * sim.CHANNEL_WIDTH_HZ))
    fast = sim.scipy.fft.next_fast_len(n_wide_check) == n_wide_check
    t0 = time.perf_counter()
    sim.build_pulsar_template(
        448, sim.CHANNEL_WIDTH_HZ, sim.BASE_FREQ_HZ, sim.CHANNEL_WIDTH_HZ,
        period_s, period_s * 0.05, 1.0, PULSAR_DM,
    )
    build_s = time.perf_counter() - t0
    flag = "OK" if build_s <= 10.0 else ("OVER 10s TARGET" if build_s <= 30.0 else "OVER 30s HARD LIMIT")
    print(f"  (n_wide={n_wide_check}, fast-factored={fast})", end="  ")
    print(f"period={period_s*1000:>6.0f}ms  build={build_s:7.3f}s  [{flag}]")

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
