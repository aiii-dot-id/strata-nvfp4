# Numeric parity diagnostics

Status: bounded numeric qualification only.

The Release parity admission is passed with exact route and top-1 checks as recorded below.

## Provenance

- Pilot implementation identity: `93eee22f118fe0085869f257dbbff2a56fe70a5a`.
- Release build source head: `b399029efac8070e4be0166d0a8969485ae256f5`; build record reports dirty source files.
- These identities are retained separately; the build record is not rewritten to the pilot commit.

## Release binaries

- ring: `8ed46dfcb4c6e4eb913b892a868f8903ae2b9347e816e12a850a47b62b607137` (/work/AI/experiments/strata-release-qualification-20261019-r1/ring-build/strata_ring_bf16)
- ling: `048b9b59e4eb64270015f77708ad64e3b8653260a40e09dfac5587269da2c837` (/work/AI/experiments/strata-release-qualification-20261019-r1/ling-build/strata_ling_bf16)

## Parity gates

- ring_1031: passed; receipt SHA-256 `50c94c316e35784b1d116dd0815dfba22a6afdd01a325b5c99f8c4bfceb3586b`; route mismatches `0`; exact top-1 `True`.
- ling_195: passed; receipt SHA-256 `1331a3823cf5bebb928f2145b3e73c268e5006c7fa328af5d4aa197d34e24787`; route mismatches `0`; exact top-1 `True`.
- ling_358: passed; receipt SHA-256 `78ad387e7b449d3fd77d176a92ac412c4e7b8c8c35657322c8109f9c54f5bd08`; route mismatches `0`; exact top-1 `True`.
- fresh_ring_c02: passed; receipt SHA-256 `1787afbdd411b98cdf08cf3ec7e39045b1717132af9dba95376679d5984b323d`; route mismatches `0`; exact top-1 `True`.
- fresh_ling_a01: passed; receipt SHA-256 `8192920fdf527569d814d353771ef64e1227ec6665a33e9d19471c12c71cf596`; route mismatches `0`; exact top-1 `True`.

## Fresh held-out interpretation

- Ring: 5 correct/10; 4 timeouts; 2.816021886473314 correct verified/hour.
- Ling: 1 correct/10; 0 timeouts; 4.304160587483729 correct verified/hour.
- Ring allows native reasoning up to 2048 total generated tokens; Ling uses thinking-off with 256 total generated tokens. These are different policy scopes and are not a speedup comparison.
- Independent audit records C02 as a substantive semantic failure; C01/C02 also have an explicit marker-spacing ambiguity that must not be posthoc scored as a model error.
Fresh quality remains bounded to the completed campaign receipts. Any timeout, length termination, invalid receipt, or missing completion remains a negative outcome and does not earn correctness.
The campaign does not establish broad model ranking, training benefit, production readiness, or isolated prefill/decode throughput.

No quantization, weight modification, or Mac inference is included.

## Historical diagnostics preserved

# Ring and Ling numerical parity, 2026-10-16

AI Identity Incorported <james@aiii.id>

Current qualification is in the 2026-10-19 section below. Earlier failures are retained historical measurements, not the current status of the new opt-in profiles.

This record covers the opt-in `ling-ring-bf16` path on dev8's RTX 4070 Ti.
Original BF16 checkpoint values and existing F32 metadata are retained; working
activations are FP32. No weights are quantized, trained, exported, or promoted.
The Mac's serving runtime was not used or interrupted.

## Ring: a measured frequency-construction defect

At position 172, layer 7, the previous native trace selected expert 241 where
the original CUDA-cached decoder selected 227. Same-input operator replay
isolated a rotary-frequency difference. Of the 32 inverse-frequency values,
index 2 was `0x3edee9bd` in the native constructor and `0x3edee9bf` in the
original implementation. At position 172 that produces angles
74.88484954833984 and 74.88485717773438 respectively.

`--rope-frequency original` uses the exact original FP32 frequency buffer for
the admitted Ring geometry (`theta=600000`, rotary dimension 64). Unsupported
geometry fails closed. The source-defined buffer is metadata, not a modified
checkpoint or an adjusted router score. `native` remains the default.

Same-input CUDA replay reduced query rotary error from 3.23057e-5 to 4.76837e-7
and key rotary error from 2.86102e-5 to 4.76837e-7 at the captured position.
This is local operator attribution, not a global bitwise-equivalence claim.

| Original-frequency case | Evaluated positions | Route rows | Strict result |
|---|---:|---:|---|
| Original first-failure trace | 173 | 3,287 | Pass |
| Natural generation, 17 + 25 | 375 | 7,125 | Pass; EOS, final answer 42 |
| Context-512 generation trace | 519 | 9,861 | Pass |
| Context-1024 generation trace | 1,031 | 19,589 | Fail |

The 1,031-position trace has two ordered-route differences, at positions 680
and 714 in layer 16, and no expert-set differences. All top-token choices,
final hidden values, and logits pass their existing gates. There are 46
intermediate-hidden violations and three expert-keyed route-weight violations.
The first hidden violation occurs at position 461, before either order swap;
it must not be attributed to those swaps.

The position-680 probe separates another issue from upstream rounding:
native and original corrected scores for experts 171 and 194 are exactly
tied. Both original-CUDA same-input gate replays retain that tie. The native
CPU selector orders them 171/194; the original CUDA selector orders them
194/171. The installed `torch.topk` documentation does not guarantee stable
tied indices. Matching this reference therefore requires matching its actual
selection implementation, not inventing an epsilon or reordering comparator
outputs. The strict failure remains retained.

The optional normalized-attention/divide-scale combination was rejected at
173 positions: one hidden element exceeded its unchanged tolerance. Matching
routes and final outputs did not make that profile pass. These options remain
diagnostic; they are not defaults or qualified replacements.

## Ling: isolate the cause before changing routing

The retained failing trace has 358 evaluated positions and 8,234 route rows.
Its one order mismatch is position 337, layer 19. The original order is
`[25,79,8,22,116,124,26,66]`; the native order is
`[25,79,8,22,124,116,26,66]`. The expert set and all top-token choices match.
Original bias-corrected scores for experts 116 and 124 are exactly tied at
0.04303882271051407. Native rounding separates them.

An optional `--rms-mode torch-cuda-order` models the original CUDA mean
reduction: four accumulators per lane, geometry-derived block widths, a
descending reduction including offset 32, and FP32 reciprocal multiplication
for the mean. KDA's separate fused output normalization is unchanged.

This normalization-only candidate does **not** resolve the route mismatch.
Its hidden, final-hidden, and logit numerical gates pass, but the ordered route
still fails. Same-input original CUDA post-normalization reproduces both the
native and original captured gate inputs bitwise at the probe. Running the
actual original gate on the native input retains the native order. This
places the remaining cause upstream of that local normalization, rather than
in a tie-breaking policy. The original RoPE frequency buffer already matches
the native Ling constructor on this host; its optional `original` mode is not
claimed as a repair.

`--matmul fp32-cublas` is an additional isolated diagnostic. It widens each
original BF16 matrix into a temporary FP32 buffer, uses pedantic cuBLAS math,
and disables cuBLAS atomics. It retains the original BF16 expert cache. The
extra matrix scratch is exactly 965,738,496 bytes (921 MiB); the total explicit
matrix workspace is 1,449,254,912 bytes. This excludes internal library and
allocator overhead and is not a device-wide peak measurement. The wrapper
requires exact backend-specific workspace accounting and rejects drift or
mismatched profile receipts. `bf16-gemv` remains the default.

The combined `--matmul fp32-cublas --rms-mode torch-cuda-order` profile passed
the unchanged 358-position strict gate: all 8,234 route rows and all top-token
choices match. Hidden, final-hidden, and logit maximum absolute errors were
0.00012826919555664062, 0.00002372264862060547, and
0.00003814697265625 respectively. This is a bounded development-trace result.
The newly built default BF16 GEMV path also reproduced all 27 retained
four-position layer/final/logit/route files bitwise.

## Ring: matching CUDA selection is necessary but insufficient

The additional opt-in `--router torch-cuda` backend executes the original
CUDA selection operators, including their tied-index behavior, through the
installed ATen library. It does not add a score epsilon. With the corrected
frequency buffer and BF16 GEMV, the 173-position case passes. At 1,031
positions, all 19,589 ordered routes and all top-token choices now match,
but intermediate values and router weights still fail the numerical gate.
Matching routing is a real improvement, not complete numerical qualification.

`--rms-mode torch-cuda` is a separate operator diagnostic. Adding it to that
profile makes the 173-position case fail again at position 172, layer 7:
one expert-set difference. It is rejected as a repair. A final source-ordered
profile combines the original frequency buffer, CUDA router and RMS,
`fp32-cublas`, normalized-first Q/K attention, and division by the attention
scale. That profile matches every route and top-token choice at 173 positions,
but still fails numerical comparison. Neither failed short candidate proceeds
to its conditional 1,031-position check. Exactness of an isolated operator
does not imply exactness of its composed pipeline.

| Ring profile | Positions | Ordered-route mismatches | Strict result |
|---|---:|---:|---|
| Original frequency + native BF16 GEMV/RMS/router | 519 | 0 | Pass |
| Original frequency + native BF16 GEMV/RMS/router | 1,031 | 2 | Fail |
| Original frequency + CUDA router; native BF16 GEMV/RMS | 173 | 0 | Pass |
| Original frequency + CUDA router; native BF16 GEMV/RMS | 1,031 | 0 | Fail: numerical values |
| Previous profile + CUDA RMS | 173 | 1 | Fail |
| Source-ordered CUDA RMS/router + FP32 cuBLAS | 173 | 0 | Fail: numerical values |

The optional Torch backend is built with `RING_ENABLE_TORCH_ROUTER=ON`.
Both CUDA router and CUDA RMS require it. A build without it rejects those
options; the RMS check occurs before checkpoint loading or CUDA initialization.
A final build check exposed two unguarded RMS references in no-Torch compilation;
conditional guards now keep that build working. The already-present early RMS
rejection was also directly checked with dummy paths, without loading a model.
Torch C++ flags come from the installed package. The verified launcher hashes
the actual executable's resolved dynamic-library closure, including the CUDA
and CPU Torch libraries and c10. It rejects unresolved or missing required
dependencies. Native-only execution does not acquire a Torch dependency.

## Ling: repaired thinking trace, then a fresh counterexample

The combined cuBLAS/RMS-order profile was exercised on seven generation
controls. Six pass strict comparison: two thinking-off arithmetic cases,
JSON output, the original 358-position thinking-on case, the 521-position
context case, and a one-token length-control case. The length control checks
stopping behavior, not answer completion.

A fresh thinking-on equation case generates the correct final answer `13`,
closes its reasoning section, and reaches EOS. Its 195-position trace still
fails strict comparison: 29 of 4,485 route rows differ, although all top-token
choices match. These differences include changed expert sets, not merely
permutations. Its first mismatch is position 46, layer 13: the original
selects expert 75 where native selects 73. The original controller stops
before its answer gate because strict comparison fails. A separate supplemental
answer-format check confirms the decoded result; it does not waive that failure.

Same-input replays locate this fresh cause before the local router:

- Actual original CUDA gate projection reproduces each captured native and
  reference gate-logit array bitwise. On native input it still selects 73.
  The corrected score margin for 75 minus 73 is -1.1175870895385742e-8 on
  native input and +4.842877388000488e-8 on reference input.
- Actual original CUDA post-attention RMS reproduces each captured gate input
  bitwise. Their variance and inverse scale are identical. A local RMS or
  tie-breaking substitution therefore does not repair this failure.
- Replaying the actual original layer-13 KDA attention on native upstream
  history changes the post-attention vector by at most 1.7881393432617188e-7
  but still selects 73. The reference-history self-control is bitwise exact
  and selects 75. Replacing this one attention block is insufficient: its
  input already carries the consequential rounding difference.

The verified Ling launcher now binds the actual cuBLAS executable-library
closure as well as checkpoint, implementation and output bytes. A fresh
runtime-bound rerun passes the repaired 358-position case again and reproduces
the 195-position failure bitwise across all 27 native trace files. The negative
readmission returns success only for *reproducing the declared negative
control*; its strict qualification result remains failed.

## Defaults and test controls

All new arithmetic modes remain opt-in. The final Ring native-default
173-position control reproduces all 23 retained trace files bitwise. Ling's
native-default four-position control reproduces all 27 retained trace files
bitwise. Trace-file equality excludes receipts, whose timestamps and source
bindings necessarily differ. The final conditional-compilation repair was
rebuilt in both Torch-enabled and native-only CUDA configurations. The final
Torch executable then passed a new 173-position default control and the
173-position original-frequency/CUDA-router positive control; each reproduced
all 23 files of its respective retained native trace bitwise. These are final
source readmissions, not new long-context or global parity claims. An initially selected historical Ring four-token
baseline predated the already-committed RMS repair and was invalid for this
task; its failed attempt and baseline correction are preserved separately.

CPU validation: 183 Python tests passed; Ring CTest passed 7/7, including
actual CPU Torch operator controls; Ling CTest passed 2/2. A dedicated Ring
attention test checks raw normalized Q/K versus pre-normalized Q/K with the
skip flag. Tests run with assertions enabled. CPU results supplement, rather
than replace, the actual RTX comparisons above.

## Remaining work and admission decision

Both model paths remain partially qualified development tools. Neither is a
globally validated intervention, gradient, training, serving, or capability
oracle. Original checkpoint bytes, thresholds, native defaults, and serving
processes are preserved. No candidate is promoted.

The next repair should follow the earliest same-input operator divergence
before the first fresh route-set change: compare input normalization,
attention projections/state updates, nonlinearities, expert outputs, and their
weighted reductions against actual original CUDA operators. Attribute a stage
before replacing it, freeze that candidate, then replay both the known positive
and negative traces before fresh generation controls. Another tie heuristic,
a wider tolerance, or final-answer agreement would not settle this failure.


## Evidence, gates, and limits

The numerical gate remains:

- every hidden and logit element: `abs(delta) <= 0.001 + 0.001*abs(reference)`;
- exact ordered expert IDs and exact top-token choices;
- every slot-wise router weight: `abs(delta) <= 1e-5 + 1e-4*abs(reference)`.

No tie epsilon, route canonicalization, or tolerance change is used. The
original reference runs its actual tokenwise CUDA cache path without a KDA
mode override. Generation checks use the original tokenizer and template;
the original reference verifies each causal native generation trace. They
are bounded engineering checks, not a held-out capability benchmark or an
independent original-model generation run.

Raw evidence is retained on dev8:

- `/work/AI/experiments/ring-rounding-20261016-r1`
- `/work/AI/experiments/ring-repaired-generation-20261016-r2`
- `/work/AI/experiments/ring-router680-20261016-r1`
- `/work/AI/experiments/ring-router680-20261016-r2`
- `/work/AI/experiments/ling-router-rounding-20261016-r1`
- `/work/AI/experiments/ling-rms-order-20261016-r1`
- `/work/AI/experiments/ling-cublas-qualification-20261016-r1`
- `/work/AI/experiments/ring-torch-qualification-20261016-r2`
- `/work/AI/experiments/ring-torch-rms-qualification-20261016-r1`
- `/work/AI/experiments/ring-matched-source-qualification-20261016-r1`
- `/work/AI/experiments/ring-final-source-readmission-20261016-r1`
- `/work/AI/experiments/ring-native-rms-guard-build/NO_TORCH_BUILD_RECEIPT.json`
- `/work/AI/experiments/ling-repaired-generation-20261016-r1`
- `/work/AI/experiments/ling-runtime-readmission-20261016-r1`
- `/work/AI/experiments/ling-fresh-routing-20261016-r1`

`CURRENT_NUMERIC_QUALIFICATION.json` binds the consolidated evidence inventories
and selected exact receipts by SHA-256. Older machine-readable qualification
files and the historical sections in the per-model validation documents are
preserved; they do not describe these new opt-in profiles.

Native/reference observation controls preserve the compared model trace bytes
bitwise. The enhanced Ling reference was checked against its retained
pre-enhancement trace. An early Ring probability observer incorrectly used a
running denominator; its probability/context metrics are excluded. The
corrected observer records final normalized probabilities and has CPU and
actual-trace controls.

Failed harness attempts remain visible: an original observer retained a
previous GPU decoder layer and tripped the memory gate; a legacy Ling probe
lacked the new diagnostic fields; a generation controller pre-created its
output directory; and a concurrent build-file edit invalidated a source
freeze. Corrections and fresh/readmitted runs are recorded separately. A
source-freeze rejection is not a model parity result.

The main dev8 AII process identity is checked before and after each admitted
run. No serving process or voice worker was terminated for these tests.


## Current composed operator profiles, 2026-10-19

The new opt-in profiles pass the retained failing traces with the same comparison
thresholds. All original BF16 checkpoint files were hashed before and after each
run. BF16 expert caching and FP32 working activations remain in use. No weights
were quantized, trained, exported, or promoted, and the Mac runtime was not used.

| Profile and trace | Positions | Ordered routes | Route mismatches | Strict result |
|---|---:|---:|---:|---|
| Ring composed CUDA operators, long trace | 1,031 | 19,589 | 0 | Pass |
| Ling composed CUDA operators, retained counterexample | 195 | 4,485 | 0 | Pass |
| Ling composed CUDA operators, previous positive trace | 358 | 8,234 | 0 | Pass |

Every greedy next-token choice matches the cached original on these traces.
Intermediate hidden values, final hidden values, logits and route weights pass
their unchanged absolute-plus-relative tolerances. This is bounded numerical
parity, not global bitwise equality or evidence of greater model intelligence.

Ring uses `--matmul fp32-cublas --router torch-cuda --rms-mode torch-cuda
--rope-frequency original --attention-backend torch-cuda --pointwise torch-cuda`.
The attention backend executes batched CUDA QK, division by the head-dimension
scale, FP32 softmax and AV. Pointwise operations use CUDA SwiGLU and the original
three-dimensional expert-weighted reduction. Legacy CPU attention-order/scale
switches remain at native defaults and are rejected when combined with this
backend. Enable `RING_ENABLE_CUDA` and `RING_ENABLE_TORCH_ROUTER` to build it.

Ling uses `--matmul fp32-cublas --rms-mode torch-cuda-order --pointwise torch-cuda`.
Pointwise mode overrides the legacy RMS implementation with actual CUDA RMS,
and also executes CUDA routing, beta sigmoid, composed KDA output norm/gating,
SwiGLU and the original expert-weighted reduction. Its KDA recurrence remains
native C++; the composed output norm is not claimed to be the original fused
kernel. MLA cache and attention remain native. Enable `LING_ENABLE_CUDA` and
`LING_ENABLE_TORCH_OPS`. Runtime receipts name the selected operator scope.

Neither profile changes the defaults. The new executables reproduced every
retained layer, final-hidden, logit and route byte on four-position default
controls for both models. The full Python suite passed 189 tests. CPU CTests
passed 7/7 for the Torch Ring build and 3/3 for Torch Ling. The native-only
6/6 and 2/2 counts originally attributed here came from an older source receipt
and did not qualify that commit. This attribution is corrected by fresh
no-Torch builds in the 2026-10-20 update below.

Evidence is under `/work/AI/experiments/strata-parity-repair-20261019-r1/`.
`CURRENT_NUMERIC_QUALIFICATION.json` records exact comparison-result hashes.
These first artifacts have an unspecified CMake build type. Optimized Release
artifacts must pass the same gates before they enter the separate fresh quality
and efficiency run. Cross-date run times are not a controlled speedup estimate.


## Opt-in residency and verification update, 2026-10-20 UTC

Implementation commits add original-BF16 output-head residency, repair a real
Ring no-Torch compilation guard, and add bounded parallel full-file SHA-256.
CUDA arithmetic, comparison thresholds, default profile selections and model
checkpoint bytes are unchanged. Head mode stays off and hash workers stay 1
unless explicitly selected.

Head-enabled Ring 1,031 and Ling 195/358 retained traces matched every native
payload and route byte; their original-source comparisons passed unchanged
thresholds. These repeat 32,308 routed positions, not new independent quality
examples. Both real insufficient-memory controls refused before output.
The final Ring executable was replayed after the no-Torch guard repair.
Fresh builds passed Ring Torch 7/7, native-only 6/6, Ling Torch 3/3 and native-only
2/2 CPU CTests. The previous no-Torch receipt referred to older source, as
noted above; it is retained as historical evidence rather than reused as proof.

The hashing helper performs the same complete serial/parallel per-file digest
in deterministic caller order. Every checkpoint shard is still hashed before
and after each run. Source-closure review caught a missing helper pin before
integration; it was repaired. All 201 CPU tests pass, including missing/read/
hash failures, helper mutation, strict worker bounds and actual CLI forwarding.
Real four-worker Ring/Ling checkpoint digests match the retained full-checkpoint
receipts; four-position native payloads/routes are bitwise exact. Thread caps
are recorded per job, with no machine-global configuration change.

Efficiency and quality remain separate. The two-case direct Ling smoke has
identical tokens with residency off/on and 1.151x native / 1.097x whole-case
observed throughput; both answers are wrong in both modes, so verified work
per hour is zero. Shared-host Ring timing became uninterpretable when load
exceeded 100 and I/O pressure exceeded 60 percent. The partial experiment is
interrupted, not passed. A 512-token Ring completion smoke was also stopped
for feasibility; length termination cannot earn correctness.

Ling's own documentation defaults to thinking on. A consumed arithmetic
calibration changed direct-mode 81 to correct 79 when thinking was enabled;
both arms reached EOS. This is one diagnostic case, not fresh improvement or
production admission. The anonymous six-case paired pilot is independently
oracled and prepared separately. No research weights or serving changes were
promoted. All execution was on dev8; Sev's Mac runtime was untouched.

Current artifact paths and SHA-256 pins are in CURRENT_NUMERIC_QUALIFICATION.json.
