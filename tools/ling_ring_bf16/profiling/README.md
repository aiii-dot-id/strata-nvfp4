# Observe a verified Ring or Ling run

AI Identity Incorported <james@aiii.id>

This Linux helper observes a process that the experiment driver already owns. It works with both verified BF16 executables. It does not load a model, launch or stop a process, claim the GPU, or change a serving runtime. The driver retains the lock, admission checks, deadline, cancellation, waiting and cleanup.

## Driver integration

Import `WrapperObserver` from `wrapper_observer.py`. After starting the verified wrapper and before waiting for it, call:

```python
observer = WrapperObserver(output / "profile.json", poll_seconds=1.0)
observer.start(wrapper.pid, admitted_native_executable)
try:
    stdout, stderr = wrapper.communicate(timeout=outer_timeout)
finally:
    # The driver must terminate and reap its owned process tree first on failure.
    # finish() observes and joins its own sampler; it never terminates the model.
    profile = observer.finish()
```

The driver must apply its existing process cleanup in the failure path. This fragment is an observation hook, not a replacement timeout or GPU controller. It requires a live wrapper PID and the exact native executable path. Sampling tracks descendants by PPID and start time, including native children started in a separate process group. It rejects a reused wrapper PID. `finish()` raises if the sampler failed or could not stop within its bounded wait.

Require `observation_complete == true` and an `observed` native envelope when a native measurement is mandatory. A short run that was never sampled is explicitly `unavailable_exact_executable_never_observed`; it is not a zero-cost run.

## What is measured

At intervals of at least one second the helper records owned descendant identities, RSS, host available memory and pressure, and read-only GPU memory telemetry. GPU process details are retained only for the wrapper lineage. Shared device utilization is labelled as shared and cannot be attributed to this model.

The pre-native, native and post-native envelopes are bounded by polling observations. They are not exact phase boundaries. Native lifetime includes initialization, mapping and trace writes. The verified wrapper's duration also includes checkpoint verification and orchestration. Subtracting native time from wrapper time does not isolate hashing time. Summed process RSS can count shared pages more than once.

This tool does not collect CUDA events, copies, kernel durations, per-token decoding costs, or profiler-grade prefill/decode separation. Use a separately admitted targeted GPU profile to resolve those costs. The current shared host results are observational and do not establish a causal speedup.

## Validation and evidence

Run the CPU controls without a GPU:

```sh
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools/ling_ring_bf16/profiling -p 'test_*.py' -v
```

The shipped observer bytes are identical to the helper used in the retained long replays, generation checks, cache-capacity comparison and Ling state-accounting qualification on dev8 on 20 October 2026. That helper has SHA-256 `b58a80201b00ce09fa59b100c0068a5bb0e85e63d6762628ca8604933d633042`. Live samples required exact native-executable observation and successful sampler shutdown. See [../NEXT_GATE_20261020.json](../NEXT_GATE_20261020.json) for the pinned evidence index and [../UPSTREAM_QUALIFICATION_20261020.md](../UPSTREAM_QUALIFICATION_20261020.md) for scope.
