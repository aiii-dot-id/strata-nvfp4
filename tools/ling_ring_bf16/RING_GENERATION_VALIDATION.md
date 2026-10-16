# Ring generation and context qualification

AI Identity Incorported <james@aiii.id>

This is the 2026-10-16 dev8 RTX 4070 Ti qualification of the opt-in Strata
`ling-ring-bf16` reference path. It uses the original Ring-mini-2.0 BF16
checkpoint and FP32 working state, without quantization. It does not qualify
BF16 activation arithmetic, training, a production server, or general answer
quality. The earlier four-token report remains historical evidence.

## Measured outcome

| Case | Prompt / generated tokens | Evaluated positions | Original next-token choices | Strict internal comparison |
|---|---:|---:|---|---|
| Short generation | 29 / 128 | 156 | All match | Pass |
| Context 128 | 128 / 8 | 135 | All match | Pass |
| Context 512 | 512 / 8 | 519 | All match | Fail |
| Context 1024 | 1024 / 8 | 1031 | All match | Fail |
| Natural completion | 29 / 347 | 375 | All match | Pass |

The completion case asks for 17 + 25. It closes the reasoning section, returns
42, and stops at the checkpoint's configured end token. Generation uses the
checkpoint's original tokenizer/chat template and deterministic greedy sampling.
The reference evaluates the native causal token trace; this is teacher-forced
reference verification of every greedy choice, not an independently cached
reference generation run. These cases are development checks, not a capability
benchmark or a validated 1,024-token context quality score.

The 512- and 1,024-token **strict gates remain failed**. Matching token choices
does not establish matching hidden states. No threshold was relaxed after the
results were seen. The gate requires every hidden/logit value to satisfy
`abs(candidate-reference) <= 0.001 + 0.001*abs(reference)`, exact ordered expert
IDs, and router weights within `1e-5 + 1e-4*abs(reference)`.

At 512 tokens, the original CPU and CUDA implementations also disagree on
21 ordered routing rows, first at position 172, layer 7, while all 519 top-1
choices match. Diagnostics show that upstream rounding can switch an almost
tied expert at that point. This establishes a portability issue in the
reference itself; it does not excuse every native discrepancy.

Compared directly with the original CPU reference, the native 512-token trace
has zero expert-set differences and one ordered-ID difference. Final hidden
states and logits pass the unchanged numeric tolerance, but 29 intermediate
elements do not. Thus neither reference makes strict native parity pass.

## Cache, cancellation, and repairs

- A 6 MiB expert cache reproduced every layer, final state, logit, route, and
  generated token from the 1 GiB-cache short case bit for bit. A 6 MiB cache
  holds one original expert; it is a correctness stress, not a speed setting.
- Cancellation during active native execution terminated the owned experiment
  process tree, left no successful provenance receipt, and left the existing
  AII OS and voice processes unchanged.
- A balanced FP32 RMS reduction fixed an initial short-case tolerance failure.
- RoPE now follows the original reciprocal-positive-power operation order.
  This correction did not remove the longer-context route divergence.
- The memory receipts distinguish resident host and device expert copies,
  projection workspace, host KV storage, and excluded allocation overhead.

The main 1,024-prompt run evaluated 1,031 positions in about 310 seconds inside
the native executor. This excludes checkpoint hashing, reference execution,
and other qualification overhead. It is a single reference-path timing, not a
production throughput claim or a comparison with another runtime.

The completed CPU reference for that same 1,031-token native trace took 112.6
seconds with four threads and allocated zero GPU bytes. Compared directly with
that CPU reference, all 1,031 top-1 rows and all eight generated choices match,
but strict internal parity fails: 66 ordered route rows and 17 route-set rows
differ; slotwise router-weight max error is 0.80815 (keyed common-expert max
error 0.34564); intermediate hidden, final hidden, and logits max errors are
56.4079, 4.08121, and 2.06936 respectively. The separate original CPU-versus-
CUDA comparison has 31 ordered route mismatches and 4 route-set mismatches,
while all top-1 rows still match. These are portability and internal-parity
measurements, not relaxed generation gates.

## Evidence and use boundary

Raw results, source/executable/checkpoint bindings, plans, diagnostic controls,
and earlier failed attempts are retained on dev8 under:

`/work/AI/experiments/ring-generation-20261016-r2`

Earlier attempts remain in the sibling `ring-generation-20261016` and
`ring-generation-20261016-r1` directories. The original 32-bit floating-point
reference widens BF16 values exactly and uses the checkpoint's original
decoder classes; CUDA TF32 is disabled. The checkpoint is never rewritten.

Use the opt-in path for bounded generation and cache experiments with this
known numerical limitation. Do not treat the longer traces as an exact
intervention or gradient oracle, or promote this path to production on this
evidence. The next discriminator is cached-versus-full reference execution on
the same long traces and task-level quality validation, not a wider tolerance.

Sev's Mac oMLX runtime was not stopped, restarted, or used for these tests.
