# %% [markdown]
# # Benchmark: `TiledNoiseStreamer` (pre-generated noise bank)
#
# Companion to benchmark_direct_synthesis.py. That streamer generates
# fresh Box-Muller noise every tick; this one pre-generates a bank of
# N tiles once, then does an O(1) index draw + memcopy per tick. The
# question this benchmark answers: at 448 channels (full 350MHz band),
# what's the actual resource cost of each piece --
#   (a) MEMORY: how big does the noise bank get as N grows?
#   (b) STARTUP: how long does generating that bank take, and how does
#       thread count affect it?
#   (c) STEADY-STATE PER-TICK COST: once the bank exists, how many CPU
#       threads does *serving* ticks from it actually need? The target
#       (per the user) is ~8 CPU cores/pod, if that's enough to stay
#       under the 2.621ms/tick budget at 448 channels.
#
# Per CLAUDE.md's noise-strategy discussion: this tradeoff is a real
# fidelity compromise (finite bank -> a station's own noise repeats
# every ~1.25*sqrt(n_tiles) ticks), not a free lunch -- this benchmark
# is about quantifying resource usage, not about validating correctness
# (tiled_noise_streamer.py's own __main__ covers that).

# %%
import statistics
import time

import numba

import ska_low_station_beam_simulator.tiled_noise_streamer as tiled
from ska_low_station_beam_simulator.common import HEAP_LEN, CHANNEL_WIDTH_HZ
from ska_low_station_beam_simulator.direct_synthesis import DelayPolynomial, StationConfig

BUDGET_MS = (HEAP_LEN / CHANNEL_WIDTH_HZ) * 1000
NUM_CHANNELS = 448  # full band -- the case this benchmark exists to answer


def _fake_fetch(station_id, at_time):
    return DelayPolynomial(
        station_id=station_id,
        start_validity_sec=at_time,
        validity_period_sec=600.0,
        xypol_coeffs_ns=[750.0, 0.0046, 0.0, 0.0, 0.0, 0.0],
        ypol_offset_ns=2.0,
    )


tiled.fetch_delay_model_from_cbf = _fake_fetch


def build_station(station_id=1):
    return StationConfig(
        station_id=station_id, substation_id=0, subarray_id=1, beam_id=1, first_channel_id=0, scan_id=99
    )


def time_bank_build(n_tiles, n_threads, num_channels=NUM_CHANNELS):
    numba.set_num_threads(n_threads)
    t0 = time.perf_counter()
    streamer = tiled.TiledNoiseStreamer(
        station=build_station(),
        source_cfgs=[{"kind": "tone", "freq_hz": 20 * CHANNEL_WIDTH_HZ + 150_000.0, "amplitude": 1.0}],
        noise_cfg={"std": 0.05, "seed": 7},
        obs_time_ref=1_800_000_000.0,
        num_channels=num_channels,
        n_tiles=n_tiles,
    )
    build_s = time.perf_counter() - t0
    return streamer, build_s


def bench_ticks(streamer, n_warmup=5, n_measured=30):
    obs_time = 1_800_000_000.0
    n_samples = streamer.tick_n_samples()
    tick_dt = n_samples / streamer.channel_output_rate
    for i in range(n_warmup):
        streamer.generate_next_tick(obs_time + i * tick_dt, n_samples)
    times = []
    for i in range(n_warmup, n_warmup + n_measured):
        t0 = time.perf_counter()
        streamer.generate_next_tick(obs_time + i * tick_dt, n_samples)
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": statistics.mean(times),
        "max_ms": max(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def gb(n_bytes):
    return n_bytes / (1024**3)


# %%
print("=" * 70)
print(f"MEMORY + STARTUP-TIME SWEEP -- n_tiles, {NUM_CHANNELS} channels, threads pinned at 8")
print("(8 threads = the target per-pod CPU budget, so startup time here")
print(" is what a pod would actually see at scan setup)")
print("=" * 70)
for n_tiles in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    # 4096 tiles would be ~120GB (both pols) on this machine's 188GB --
    # skipped to leave headroom on a shared box; the pattern below
    # extrapolates linearly (tile size is fixed), so it's not needed to
    # see where this stops being viable memory-wise.
    streamer, build_s = time_bank_build(n_tiles, n_threads=8)
    mem_gb = gb(streamer.bank_memory_bytes())
    first_repeat_ticks = 1.25 * (n_tiles ** 0.5)
    print(
        f"n_tiles={n_tiles:>5}  bank={mem_gb:7.3f} GB (both pols)  "
        f"build={build_s:7.3f}s  expected first repeat ~{first_repeat_ticks:6.1f} ticks "
        f"(~{first_repeat_ticks * BUDGET_MS:8.1f}ms of scan time)"
    )

# %%
print()
print("=" * 70)
print(f"STARTUP-TIME vs THREAD COUNT, {NUM_CHANNELS} channels, n_tiles=256 fixed")
print("=" * 70)
for n_threads in [1, 2, 4, 8, 16, 24, 32, 48, 96]:
    try:
        streamer, build_s = time_bank_build(256, n_threads)
    except ValueError:
        continue
    print(f"threads={n_threads:>3}  build={build_s:7.3f}s")

# %%
print()
print("=" * 70)
print(f"STEADY-STATE PER-TICK COST vs THREAD COUNT, {NUM_CHANNELS} channels")
print(f"budget={BUDGET_MS:.3f}ms/tick -- n_tiles=256, testing both plain numpy")
print("copy and a numba-parallelized copy, to see if threading the copy")
print("itself is even worth it once generation is no longer per-tick")
print("=" * 70)

for parallel_copy, label in [(False, "plain numpy out[:] = bank[idx]"), (True, "numba prange copy")]:
    print(f"\n--- {label} ---")
    for n_threads in [1, 2, 4, 8, 16, 24, 32, 48]:
        numba.set_num_threads(n_threads)
        streamer = tiled.TiledNoiseStreamer(
            station=build_station(),
            source_cfgs=[{"kind": "tone", "freq_hz": 20 * CHANNEL_WIDTH_HZ + 150_000.0, "amplitude": 1.0}],
            noise_cfg={"std": 0.05, "seed": 7},
            obs_time_ref=1_800_000_000.0,
            num_channels=NUM_CHANNELS,
            n_tiles=256,
            parallel_copy=parallel_copy,
        )
        stats = bench_ticks(streamer)
        pct = stats["mean_ms"] / BUDGET_MS * 100
        flag = "OK" if stats["mean_ms"] <= BUDGET_MS else "OVER"
        print(
            f"  threads={n_threads:>3}  mean={stats['mean_ms']:7.4f}ms ({pct:6.1f}% budget)  "
            f"max={stats['max_ms']:7.4f}ms  stdev={stats['stdev_ms']:6.4f}  [{flag}]"
        )

# %%
print()
print("=" * 70)
print(f"TARGET CHECK: ~8 CPU cores/pod at {NUM_CHANNELS} channels")
print("=" * 70)
numba.set_num_threads(8)
streamer = tiled.TiledNoiseStreamer(
    station=build_station(),
    source_cfgs=[{"kind": "tone", "freq_hz": 20 * CHANNEL_WIDTH_HZ + 150_000.0, "amplitude": 1.0}],
    noise_cfg={"std": 0.05, "seed": 7},
    obs_time_ref=1_800_000_000.0,
    num_channels=NUM_CHANNELS,
    n_tiles=256,
    parallel_copy=False,
)
stats = bench_ticks(streamer, n_warmup=10, n_measured=50)
pct = stats["mean_ms"] / BUDGET_MS * 100
print(
    f"8 threads, n_tiles=256, plain-numpy-copy steady state: "
    f"mean={stats['mean_ms']:.4f}ms ({pct:.1f}% of budget), max={stats['max_ms']:.4f}ms"
)
print(f"bank memory (both pols): {gb(streamer.bank_memory_bytes()):.3f} GB")
