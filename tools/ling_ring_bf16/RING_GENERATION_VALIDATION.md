# Ring generation and context qualification

AI Identity Incorported <james@aiii.id>

Current readmission: the opt-in original-frequency repair passes 173, 375,
and 519 positions. At 1,031 positions the additional CUDA router matches all
19,589 ordered routes but numerical gates still fail. Adding CUDA RMS is not
a qualified repair. See [NUMERIC_PARITY_DIAGNOSTICS.md](NUMERIC_PARITY_DIAGNOSTICS.md)
and [CURRENT_NUMERIC_QUALIFICATION.json](CURRENT_NUMERIC_QUALIFICATION.json).
The measurements below are preserved historical default-path evidence.

This is the 2026-10-16 dev8 RTX 4070 Ti qualification of the opt-in Strata
`ling-ring-bf16` reference path. It uses the original Ring-mini-2.0 BF16
checkpoint and FP32 working state, without quantization. It does not qualify
BF16 activation arithmetic, training, a production server, or general answer
quality. The earlier four-token report remains historical evidence.

## Historical default-path generation evidence (before frequency repair)

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

For that historical default profile, the 512- and 1,024-token **strict gates failed**. Matching token choices
does not establish matching hidden states. No threshold was relaxed after the
results were seen. The gate requires every hidden/logit value to satisfy
`abs(candidate-reference) <= 0.001 + 0.001*abs(reference)`, exact ordered expert
IDs, and router weights within `1e-5 + 1e-4*abs(reference)`.

In the historical 512-token case, the original CPU and CUDA implementations also disagree on
21 ordered routing rows, first at position 172, layer 7, while all 519 top-1
choices match. Diagnostics show that upstream rounding can switch an almost
tied expert at that point. This establishes a portability issue in the
reference itself; it does not excuse every native discrepancy.

### Historical FP32-cuBLAS diagnostic (before source-ordered candidate)

An optional candidate executable was exercised with `--matmul fp32-cublas`.
It widens each resident BF16 matrix into a temporary FP32 weight buffer of
1,287,651,328 bytes (about 1.2 GiB), uses pedantic cuBLAS math with atomics
disabled, and keeps the same 1,024 MiB expert-cache budget. This is a
diagnostic backend, not an optimization claim or promotion path.

The executable SHA-256 is
`30df55ecd84b668b2e33d4d9510085d23e4757694e89941c9fd4b4e4848d5001`.
The external protocol is pinned by
`/work/AI/experiments/ring-cublas-qualification-20261016/run.py`, with fresh
output directories, CUDA cached original references, `OMP_NUM_THREADS=4`,
`NVIDIA_TF32_OVERRIDE=0`, 600-second native and 1,200-second reference
timeouts, and the unchanged strict comparator. The Ring CPU CTest suite
remained 6/6 before this optional run.

The 4-position cuBLAS candidate passed strict comparison (hidden max
`0.001953125`, final max `0.0000152587890625`, logits max
`0.0000057220458984375`, zero route mismatches). The 173-position candidate
failed at the same position 172/layer 7 expert flip, 227 versus 241, as the
default backend; all top-1 choices remained exact. The 519- and 1,031-position
cuBLAS candidates were not run after that failure. A separate default-backend
4-position regression subsequently passed bitwise against its frozen expected
outputs in 38.59 seconds; its result SHA-256 is
`9cb0eb2e8fdc521ca7cc744e7a9b5a31e108443c6339497befbda57c06a9adf2`.

The cuBLAS diagnostic therefore does not resolve the known long-context route
tie and does not change the default backend, strict gates, or promotion status.

Compared directly with the original CPU reference, the native 512-token trace
has zero expert-set differences and one ordered-ID difference. Final hidden
states and logits pass the unchanged numeric tolerance, but 29 intermediate
elements do not. Thus neither reference makes strict native parity pass.

The retained pre-frequency-repair CUDA-cached comparisons make the historical boundary explicit. The
135-position case passed all unchanged strict gates (2,565 route rows); its
comparison result is
`/work/AI/experiments/ring-cached-20261016-r1/case-128/compare/result.json`
(SHA-256
`04c898cd71d3f88a26982b394023281a206a1418f28301d69cdfcd5f4008134b`). The historical
519-position case failed first at position 172/layer 7, expected expert 227
versus native 241; its result is
`/work/AI/experiments/ring-cached-20261016-r1/case-512/compare/result.json`
(SHA-256
`42937b64c71bb85c9310511e319400d030f8cf08dc1478e371de3193e9048f27`). The
independent 173-position probe observed the same first set flip and failed
numeric parity; its result is
`/work/AI/experiments/ring-cached-20261016-r2/probe173/compare/result.json`
(SHA-256
`f21f4d6187a54e4e84382eceb3c6ffca652ea794ff5226b8afe70e52990ab9d9`).
The separate full CPU-versus-CUDA and FP64 input-rounding diagnostics remain
historical evidence under
`/work/AI/experiments/ring-generation-20261016-r2`; the retained
`cpu-native512-diagnostic.json` is SHA-256
`e245132ca6699ff0b0ae40064d6bdf96d49c931775e56d4d9c6b3a1d5b58f099`. Those
diagnostics point to upstream pre-router input rounding as the dominant cause
of the near-tie, rather than a sigmoid formula defect.

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
evidence. The original pre-router isolation recommendation has now produced the bounded
frequency repair described in NUMERIC_PARITY_DIAGNOSTICS.md. Remaining failures
require further same-input CUDA operator attribution; another cached-versus-full
rerun or a wider tolerance would not establish parity.

Sev's Mac oMLX runtime was not stopped, restarted, or used for these tests.
