# %% [markdown]
# # Benchmark: `StationStreamer.generate_next_tick` (numba-integrated version)
#
# Run this on your REAL target hardware. Generation (tone/pulse/noise) now
# runs via numba/prange, NOT ThreadPoolExecutor — the old `workers`/
# `time_chunks` sweep no longer measures generation parallelism at all,
# since that concept doesn't exist in this architecture anymore. The
# relevant knob is now `numba.set_num_threads()` / the `NUMBA_NUM_THREADS`
# env var. The channelization stage still uses a small fixed executor
# (2 tasks/tick, V and H) — not worth sweeping, 2 is enough by construction.
#
# Uses a fake delay polynomial (the real `fetch_delay_model_from_cbf` is
# a stub that raises `NotImplementedError`) — benchmarks generation/
# channelization cost only, not your real Tango delay-poly client latency.

# %%
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

import numba

import ska_low_station_beam_simulator.wideband_streamer as sim

# %%
# --- Fake delay model so we don't need a real CBF connection to benchmark ---


def _fake_fetch_delay_model(station_id: int, at_time: float) -> sim.DelayPolynomial:
    return sim.DelayPolynomial(
        station_id=station_id,
        start_validity_sec=at_time,
        validity_period_sec=600.0,
        xypol_coeffs_ns=[100.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        ypol_offset_ns=2.0,
    )


sim.fetch_delay_model_from_cbf = _fake_fetch_delay_model


# %%
def build_streamer(executor: ThreadPoolExecutor) -> sim.StationStreamer:
    station = sim.StationConfig(
        station_id=1,
        substation_id=0,
        subarray_id=1,
        beam_id=1,
        first_channel_id=0,
        scan_id=99,
    )
    return sim.StationStreamer(
        station=station,
        source_cfgs=[{"kind": "tone", "freq_hz": 150_000.0, "amplitude": 1.0}],
        noise_cfg={"std": 0.05, "seed": 7},
        executor=executor,
        obs_time_ref=1_800_000_000.0,  # must match benchmark_tick's obs_time below
    )


def benchmark_tick(
    streamer: sim.StationStreamer,
    n_input_samples: int,
    n_warmup: int = 5,
    n_measured: int = 30,
) -> dict:
    """Times generate_next_tick, returns per-tick timing stats in ms.
    n_warmup here also absorbs numba's JIT compilation cost (first call
    to each kernel compiles) — don't reduce n_warmup below what's needed
    to get past that, or the first measured samples will be polluted by
    a one-time cost that isn't representative of steady state."""
    obs_time = 1_800_000_000.0

    for i in range(n_warmup):
        streamer.generate_next_tick(
            obs_time + i * sim.BLOCK_DURATION_S, n_input_samples
        )

    samples_ms = []
    for i in range(n_warmup, n_warmup + n_measured):
        t0 = time.perf_counter()
        streamer.generate_next_tick(
            obs_time + i * sim.BLOCK_DURATION_S, n_input_samples
        )
        samples_ms.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": statistics.mean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "stdev_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
    }


# %%
# --- Sweep numba thread count (this is the real parallelism knob now) ---
# Ceiling is numba.config.NUMBA_NUM_THREADS, detected once at import —
# set_num_threads() raises if you exceed it. If you want to test MORE
# threads than that ceiling, set NUMBA_NUM_THREADS as an environment
# variable BEFORE running this script (not via set_num_threads()).

n_input_samples = int(sim.SAMPLE_RATE_HZ * sim.BLOCK_DURATION_S)
budget_ms = sim.BLOCK_DURATION_S * 1000
max_numba_threads = numba.config.NUMBA_NUM_THREADS

print(
    f"n_input_samples/tick: {n_input_samples}   budget: {budget_ms:.3f} ms/tick   "
    f"NUM_CHANNELS={sim.NUM_CHANNELS}   numba thread ceiling: {max_numba_threads}\n"
)

candidate_counts = sorted(set([1, 2, 4, 8, max_numba_threads // 2, max_numba_threads]))
candidate_counts = [n for n in candidate_counts if 1 <= n <= max_numba_threads]

# Channelization executor stays fixed at NUM_WORKER_THREADS (2) throughout —
# it's not the thing being swept here.
results = {}
for n_threads in candidate_counts:
    numba.set_num_threads(n_threads)
    executor = ThreadPoolExecutor(max_workers=sim.NUM_WORKER_THREADS)
    streamer = build_streamer(executor)
    stats = benchmark_tick(streamer, n_input_samples)
    executor.shutdown(wait=True)
    results[n_threads] = stats

    within_budget = "OK" if stats["mean_ms"] <= budget_ms else "OVER BUDGET"
    print(
        f"numba_threads={n_threads:>3}  mean={stats['mean_ms']:7.3f} ms  "
        f"median={stats['median_ms']:7.3f} ms  stdev={stats['stdev_ms']:6.3f}  "
        f"max={stats['max_ms']:7.3f} ms  [{within_budget}]"
    )

# %%
# --- Sequential baseline (numba_threads=1) for comparison ---

baseline_stats = results[1]
print(
    f"\nsequential baseline (numba_threads=1): mean={baseline_stats['mean_ms']:.3f} ms/tick"
)

best_threads = min(results, key=lambda k: results[k]["mean_ms"])
speedup = baseline_stats["mean_ms"] / results[best_threads]["mean_ms"]
print(
    f"best config tried: numba_threads={best_threads} -> "
    f"{results[best_threads]['mean_ms']:.3f} ms/tick ({speedup:.2f}x vs sequential baseline)"
)

# %%
# --- FFT_WORKERS sweep, at the best numba_threads found above ---
# scipy.fft's `workers` parameter is a DIFFERENT parallelism knob from
# numba's thread count — it controls concurrency WITHIN one FFT call,
# not generation. Untested default (4); sweep it directly since the
# switch from numpy.fft (no threading at all) to scipy.fft is the new
# hypothesis for where channelization time is actually going.

print("\n" + "=" * 60)
print("FFT_WORKERS SWEEP (scipy.fft internal parallelism)")
print("=" * 60)

numba.set_num_threads(best_threads)  # hold generation parallelism at its best config
fft_worker_counts = sorted(set([1, 2, 4, 8, sim.NUM_WORKER_THREADS]))
fft_results = {}
for n_fft_workers in fft_worker_counts:
    sim.FFT_WORKERS = n_fft_workers
    executor = ThreadPoolExecutor(max_workers=sim.NUM_WORKER_THREADS)
    streamer = build_streamer(executor)
    stats = benchmark_tick(streamer, n_input_samples)
    executor.shutdown(wait=True)
    fft_results[n_fft_workers] = stats
    within_budget = "OK" if stats["mean_ms"] <= budget_ms else "OVER BUDGET"
    print(
        f"fft_workers={n_fft_workers:>2}  mean={stats['mean_ms']:7.3f} ms  "
        f"median={stats['median_ms']:7.3f} ms  stdev={stats['stdev_ms']:6.3f}  "
        f"max={stats['max_ms']:7.3f} ms  [{within_budget}]"
    )

best_fft_workers = min(fft_results, key=lambda k: fft_results[k]["mean_ms"])
print(
    f"\nbest fft_workers tried: {best_fft_workers} -> "
    f"{fft_results[best_fft_workers]['mean_ms']:.3f} ms/tick"
)
print("As with numba_threads=48 earlier, don't trust this single-pass result on")
print("its own — repeat the best-looking config a few times before committing to it.")

# Pin FFT_WORKERS to the value the sweep above actually identified as
# best, for every section below. Without this, they'd silently inherit
# whatever value the loop above happened to leave behind (the LAST
# candidate tried, not the best one) — a real bug caught in exactly this
# spot in an earlier run of this script, worth guarding against here.
sim.FFT_WORKERS = best_fft_workers
print(f"(FFT_WORKERS pinned to {best_fft_workers} for all sections below)")

# %%
# --- Phase breakdown: generation vs channelization, at a few thread counts ---
# Tests the Amdahl's-law hypothesis directly: channelization runs on a
# FIXED 2-worker executor regardless of numba thread count, so as
# generation gets faster, channelization's fixed cost should become a
# growing share of the total. Uses the streamer's own private methods
# (same module, not reaching into internals from outside) rather than
# adding permanent instrumentation to the production class.

print("\n" + "=" * 60)
print("PHASE BREAKDOWN (generation vs channelization)")
print("=" * 60)

breakdown_thread_counts = [
    t for t in [8, 48, max_numba_threads] if t <= max_numba_threads
]
n_phase_measured = 20

for n_threads in breakdown_thread_counts:
    numba.set_num_threads(n_threads)
    executor = ThreadPoolExecutor(max_workers=sim.NUM_WORKER_THREADS)
    streamer = build_streamer(executor)
    obs_time = 1_800_000_000.0

    # warm up (JIT + let any thread-pool reconfiguration settle)
    for i in range(5):
        streamer.generate_next_tick(
            obs_time + i * sim.BLOCK_DURATION_S, n_input_samples
        )

    gen_times_ms, chan_times_ms, total_times_ms = [], [], []
    for i in range(5, 5 + n_phase_measured):
        t = obs_time + i * sim.BLOCK_DURATION_S
        t_rel = t - streamer._obs_time_ref
        streamer._refresh_delay_poly_if_needed(t)

        t0 = time.perf_counter()
        shared = streamer._generate_shared_and_fallback(t_rel, n_input_samples)
        raw = {}
        for pol, seed in (("V", streamer._noise_seed_v), ("H", streamer._noise_seed_h)):
            noise = sim.generate_pol_noise(
                seed, streamer._noise_std, t_rel, streamer.sample_rate, n_input_samples
            )
            raw[pol] = shared + noise
        gen_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for pol in ("V", "H"):
            tau_s = streamer._current_poly.eval_delay_seconds(t, pol)
            coarse_shift, tau_frac_s = sim.split_coarse_fine(
                tau_s, streamer.sample_rate
            )
            ring = streamer.history_buffer[pol]
            ring.write(raw[pol])
            shifted_block = ring.read_window(n_input_samples, end_offset=coarse_shift)
            if shifted_block is not None:
                streamer._channelize_and_correct(pol, shifted_block, tau_frac_s)
        chan_time = time.perf_counter() - t0

        gen_times_ms.append(gen_time * 1000)
        chan_times_ms.append(chan_time * 1000)
        total_times_ms.append((gen_time + chan_time) * 1000)

    executor.shutdown(wait=True)

    gen_mean = statistics.mean(gen_times_ms)
    chan_mean = statistics.mean(chan_times_ms)
    total_mean = statistics.mean(total_times_ms)
    print(
        f"threads={n_threads:>3}  generation={gen_mean:7.3f} ms ({gen_mean / total_mean * 100:4.1f}%)  "
        f"channelization={chan_mean:7.3f} ms ({chan_mean / total_mean * 100:4.1f}%)  "
        f"total={total_mean:7.3f} ms"
    )

print("\nIf channelization's ms stays roughly FLAT across thread counts while its")
print("percentage share grows, that confirms it's now the Amdahl's-law-limiting")
print("fixed cost — the next lever would be parallelizing channelization itself")
print("(e.g. more than 2 executor workers, or numba-izing that stage too),")
print(
    "not further numba thread tuning on generation, which is already near its ceiling."
)

# %%
# --- Repeatability check for the COMBINED best config found above
# (best_threads + best_fft_workers) — single-sample sweep results
# haven't been trustworthy on this hardware so far; confirm the actual
# candidate config with repeats before treating it as real, same
# discipline as every other number in this script.

print("\n" + "=" * 60)
print(
    f"REPEATABILITY CHECK: numba_threads={best_threads}, FFT_WORKERS={best_fft_workers}"
)
print("=" * 60)

n_repeats = 5
numba.set_num_threads(best_threads)
sim.FFT_WORKERS = best_fft_workers

repeat_means = []
for rep in range(n_repeats):
    executor = ThreadPoolExecutor(max_workers=sim.NUM_WORKER_THREADS)
    streamer = build_streamer(executor)
    stats = benchmark_tick(streamer, n_input_samples, n_warmup=5, n_measured=20)
    executor.shutdown(wait=True)
    repeat_means.append(stats["mean_ms"])
    print(f"  repeat {rep + 1}: mean={stats['mean_ms']:7.3f} ms")

print(
    f"\nspread across {n_repeats} repeats: min={min(repeat_means):.3f}  "
    f"max={max(repeat_means):.3f}  stdev={statistics.stdev(repeat_means):.3f}"
)
print(
    f"budget: {budget_ms:.3f} ms/tick — "
    f"{'WITHIN BUDGET on average' if statistics.mean(repeat_means) <= budget_ms else 'still over budget on average'}"
)
print("\nIf this spread is large, don't trust the mean alone — a real-time system")
print("needs to survive its worst case, not just its average. Compare against")
print("your co-tenancy situation on this node (see earlier discussion) before")
print("deciding whether this number represents your real deployment target.")
