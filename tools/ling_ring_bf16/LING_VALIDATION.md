# Ling-3.0-tiny BF16 reference validation

AI Identity Incorported <james@aiii.id>

Measured on 2026-10-16 on an NVIDIA RTX 4070 Ti (12 GiB, sm_89), with the
original Ling-3.0-tiny checkpoint. This record covers one bounded four-token
forward pass and a cache-budget replay. It is a partial forward qualification;
it is not a generation, long-context, answer-quality, throughput, serving, or
production qualification.

## Initial gate

The input token IDs were `156890,128,256,1024`. The native executor ran all 24
layers: 18 KDA layers and 6 MLA layers. Layer 0 is dense; layers 1 through 23
emit one ordered eight-expert route each per token, for 92 route rows. The
original reference used the same token trace and all logits.

| Check | Result |
| --- | --- |
| Native/reference comparison | passed |
| Hidden trace | 24 x 4 x 1536 |
| Logits trace | 4 x 157184 |
| Ordered route rows | 92/92 exact |
| Expert IDs | exact, 0..127, eight per route |
| Top-1 logits | 4/4 exact |
| 5 MiB versus 1024 MiB native trace | bitwise identical across 27 trace files |

The elementwise gates were declared before execution: `0.001 + 0.001 *
abs(reference)` for hidden and logits, and `1e-5 + 1e-4 * abs(reference)` for
route weights. Measured maximum absolute differences were:

- hidden layers: `0.00030517578125`
- final hidden: `0.000003337860107421875`
- logits: `0.0000095367431640625`
- route weights: `0.0000005973287964256002`

The 5 MiB replay was bitwise identical for all 24 layer files, final hidden,
logits, and routes. That proves cache-budget trace stability between the two
native runs; it does not compare cache runs against each other as a quality
claim beyond the stated reference gates.

The native run retained original BF16 matrix weights and original F32 metadata
where present, including router expert bias and KDA `A_log`/`dt_bias`. Working
activations and emitted arrays were FP32. No weight quantization was used.
The architecture split is 18 KDA layers and 6 MLA layers, with 128 experts and
top-8 routing.

The original reference used the dedicated environment with Transformers
4.55.4, `fla-core` 0.5.0, PyTorch 2.14.1+cu130, CUDA 13.0, Python 3.14.4,
and TF32 disabled. This dependency boundary differs from the Ring reference
environment (Transformers 4.52.3). The native CPU Ling components and the
actual GPU forward are separate evidence classes: CPU KDA/MLA tests do not
replace the original GPU reference comparison.

The launcher receipt hashes the manifest, executable, all 32 checkpoint
shards, config/index, original model/configuration source, Ling/Ring sources,
shared kernels, and every native output. The initial controller recorded
unchanged source and executable pins and unchanged monitored service
identities. No service was stopped or changed.

Evidence is retained under `/work/AI/experiments/ling-strata-20261016/initial/`:
`native4/`, `original4/`, `compare4/`, `native4-cache5/`, `PLAN.json`, and
`RESULT.json`.

## Bounded 64/65 boundary evidence

A later bounded GPU controller evaluated the same original BF16/F32-metadata,
FP32-working-state path at 64 and 65 tokens. The 64-token case passed the
declared strict gates: 1,472 route rows, zero route mismatches, exact top-1
logits, and maximum errors of 0.000202178955078125 for hidden layers,
0.00003814697265625 for final hidden, 0.000030517578125 for logits, and
0.0000010708093260980434 for route weights.

The standard 65-token original-reference comparison failed strict gates. It
recorded 1,495 route rows, 319 ordered-route mismatches, exact top-1 logits,
and maximum errors of 17.52444076538086 hidden, 3.1274614334106445 final
hidden, 1.4571490287780762 logits, and 0.18729710610913086 route weights.
The prefix diagnostic is diagnostic-only and does not convert this failure
into a pass. A separate forced-recurrent diagnostic then ran all 18 original
KDA layers with KDA.mode=fused_recurrent. It matched all 1,495 ordered route
rows and all top-1 logits, with maximum errors of
0.000274658203125 hidden, 0.00005245208740234375 final hidden,
0.00002956390380859375 logits, and 0.0000011323858948308896 route
weights; all declared numeric gates were satisfied. This result is explicitly
diagnostic-only with default_path_qualification: false: it does not qualify
the default path or HF default behavior beyond 64 tokens, and does not isolate
the cause to a particular Triton operation. The standard 65-token default-path
strict failure remains.

These results are bounded evidence, not a completed long-context
qualification. All 65 top-1 choices matching does not waive the strict hidden,
logit, route-ID, and route-weight gates. Evidence receipts are retained under
/work/AI/experiments/ling-strata-20261016/boundary-preparation/, including
the controller result, case-64/compare/result.json,
case-65/compare/result.json, and prefix-diagnostic.json.

The native and original paths use original BF16 weights, original F32 metadata,
and FP32 working state; no quantization or weight conversion is involved.
CUDA and cuDNN TF32 flags were disabled for the host/runtime configuration.
That flag state does not establish precision behavior inside the installed FLA
Triton chunk kernel: chunk_intra.py contains explicit TF32 solve constants,
so this boundary evidence must not be described as a Triton precision
qualification.

## Boundaries

The measured four-token and 64-token cases are bounded forward comparisons.
The 65-token default-path strict failure remains open despite the successful
forced-recurrent diagnostic; default chunk-path parity is not established.
Generation, tokenizer/chat-template behavior, sampling, answer quality,
throughput, memory-pressure qualification, and production serving remain
unmeasured. The native path is an opt-in executor and does not change the
default runtime or establish a release claim.

The CPU reference attempt is not a substitute for the original GPU result:
the installed FLA KDA path requires Triton GPU execution for this model. CPU
KDA/MLA component tests remain useful structural and state-transition checks,
but do not establish original-model equivalence.
