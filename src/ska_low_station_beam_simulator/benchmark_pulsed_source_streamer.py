# %% [markdown]
# # Benchmark: `PulsedSourceStreamer` (pre-generated, dispersion-aware pulsar)
#
# Two costs to measure, same split as benchmark_tiled_noise_streamer.py:
#   (a) ONE-TIME build cost: build_pulsar_template's cost scales with
#       n_subfreq (the intra-channel-smear sub-sampling resolution) and
#       channel count -- how expensive is getting this right?
#   (b) STEADY-STATE per-tick cost: add_pulsar_tick does a per-sample
#       delay-polynomial eval (like tone) plus a template lookup +
#       multiply-add per channel -- more work than the noise tile bank's
#       plain memcopy. How many threads does it need to clear budget at
#       448 channels?

# %%
import statistics
import time

import numba

import ska_low_station_beam_simulator.pulsed_source_streamer as pulsar
from ska_low_station_beam_simulator.common import HEAP_LEN, CHANNEL_WIDTH_HZ
from ska_low_station_beam_simulator.direct_synthesis import DelayPolynomial, StationConfig

BUDGET_MS = (HEAP_LEN / CHANNEL_WIDTH_HZ) * 1000
NUM_CHANNELS = 448  # full band

DM_TEST = 2.0
PERIOD_S = 0.1
WIDTH_S = 0.005


def _fake_fetch(station_id, at_time):
    return DelayPolynomial(
        station_id=station_id, start_validity_sec=at_time, validity_period_sec=600.0,
        xypol_coeffs_ns=[750.0, 0.0046, 0.0, 0.0, 0.0, 0.0], ypol_offset_ns=2.0,
    )


pulsar.fetch_delay_model_from_cbf = _fake_fetch


def build_station(station_id=1):
    return StationConfig(
        station_id=station_id, substation_id=0, subarray_id=1, beam_id=1, first_channel_id=0, scan_id=99
    )


def build_streamer(n_subfreq, num_channels=NUM_CHANNELS):
    return pulsar.PulsedSourceStreamer(
        station=build_station(),
        source_cfgs=[{"kind": "pulsed", "period_s": PERIOD_S, "width_s": WIDTH_S, "amplitude": 1.0, "dm_pc_cm3": DM_TEST}],
        obs_time_ref=1_800_000_000.0,
        num_channels=num_channels,
        n_subfreq=n_subfreq,
    )


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


# %%
print("=" * 70)
print(f"ONE-TIME BUILD COST vs n_subfreq, {NUM_CHANNELS} channels, 8 threads")
print("(8 threads = plausible per-pod steady-state budget)")
print("=" * 70)
numba.set_num_threads(8)
for n_subfreq in [1, 8, 16, 32, 64, 128, 256]:
    t0 = time.perf_counter()
    streamer = build_streamer(n_subfreq)
    build_s = time.perf_counter() - t0
    print(f"n_subfreq={n_subfreq:>4}  build={build_s:7.3f}s")

# %%
print()
print("=" * 70)
print(f"ONE-TIME BUILD COST vs THREAD COUNT, {NUM_CHANNELS} channels, n_subfreq=64 fixed")
print("=" * 70)
for n_threads in [1, 2, 4, 8, 16, 24, 32, 48]:
    numba.set_num_threads(n_threads)
    t0 = time.perf_counter()
    streamer = build_streamer(64)
    build_s = time.perf_counter() - t0
    print(f"threads={n_threads:>3}  build={build_s:7.3f}s")

# %%
print()
print("=" * 70)
print(f"STEADY-STATE PER-TICK COST vs THREAD COUNT, {NUM_CHANNELS} channels")
print(f"budget={BUDGET_MS:.3f}ms/tick -- n_subfreq=64 (build cost excluded from this timing)")
print("=" * 70)
for n_threads in [1, 2, 4, 8, 16, 24, 32, 48]:
    numba.set_num_threads(n_threads)
    streamer = build_streamer(64)
    stats = bench_ticks(streamer)
    pct = stats["mean_ms"] / BUDGET_MS * 100
    flag = "OK" if stats["mean_ms"] <= BUDGET_MS else "OVER"
    print(
        f"threads={n_threads:>3}  mean={stats['mean_ms']:7.4f}ms ({pct:6.1f}% budget)  "
        f"max={stats['max_ms']:7.4f}ms  stdev={stats['stdev_ms']:6.4f}  [{flag}]"
    )

# %%
print()
print("=" * 70)
print(f"TARGET CHECK: ~8 CPU cores/pod at {NUM_CHANNELS} channels")
print("=" * 70)
numba.set_num_threads(8)
streamer = build_streamer(64)
stats = bench_ticks(streamer, n_warmup=10, n_measured=50)
pct = stats["mean_ms"] / BUDGET_MS * 100
print(
    f"8 threads, n_subfreq=64, steady state: mean={stats['mean_ms']:.4f}ms "
    f"({pct:.1f}% of budget), max={stats['max_ms']:.4f}ms"
)
