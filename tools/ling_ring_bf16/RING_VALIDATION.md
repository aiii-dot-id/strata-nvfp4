# Ring BF16 reference qualification

AI Identity Incorported <james@aiii.id>

Qualified on 2026-10-16 on an RTX 4070 Ti with 12 GiB VRAM, CUDA architecture
89. The complete 20-layer Ring-mini-2.0 forward path runs from its original
checkpoint through a bounded expert cache. No weight quantization was used.
The four original checkpoint shards were verified by SHA-256 before and after
every final native run, and match the original-reference checkpoint hashes.

## What passed

| Check | Four-token input | 29-token chat input |
|---|---:|---:|
| Layers compared | 20 | 20 |
| Exact top-1 token choices | 4/4 | 29/29 |
| Exact routed expert-ID lists | 76/76 | 551/551 |
| Maximum absolute logit difference | 0.00004101 | 0.00010491 |
| 6 MiB versus 1 GiB expert cache | Bitwise identical | Bitwise identical |

Each routed decision selects eight experts. Every layer's hidden state, final
hidden state, and logit passed the predeclared elementwise tolerance of
`0.001 + 0.001*abs(reference)`. Routing weights passed
`1e-5 + 1e-4*abs(reference)`; all expert IDs and top-1 token IDs matched exactly.
The cache comparison covers all 20 layer outputs, final hidden states, logits,
and route traces. Its bitwise result is between the two native cache settings,
not between native execution and the original CPU reference.

The short input used IDs `156890,128,256,1024`. The 29-token input used the
original tokenizer and chat template for “What is 17 plus 25? Answer with only
the number.” These are teacher-forced architecture probes, not scored generated
answers or an arithmetic capability result.

The final CPU suite passed 59 Python tests and five C++ test executables.
Controls cover malformed manifests and tensors, wrong shapes, numerical
overflow, invalid attention state, cache eviction/pinning and upload failures,
missing or altered provenance, wrong tokens, invalid expert IDs, truncated or
nonfinite outputs, and numerical/route disagreement. Release-mode C++ test
assertions are explicitly enabled.

The final rerun followed fixes for overflow rejection, callback exceptions,
separate host/device memory reporting, and provenance binding. Acceptance
tolerances were not loosened. Source, binary, original checkpoint and output
hashes are recorded in `RING_QUALIFICATION.json`. Native output bytes are bound
to the verified launcher receipts; source and checkpoint drift fail the run.

## What this establishes

Strata's existing GPU BF16 GEMV kernel can execute Ring's original matrix
weights while the new Ring path supplies GQA, Q/K RMS normalization, 64-dimension
partial RoPE, causal KV state, grouped sigmoid routing, and shared/routed
SwiGLU experts. Attention and routing currently run on the CPU. All activation
arithmetic is FP32; original expert-bias tensors remain FP32.

The original implementation is evaluated in the same FP32 activation domain,
with BF16 weights widened exactly. This isolates architecture and weight-layout
correctness from BF16 activation rounding. It does not establish bitwise or
answer-quality equivalence to a native BF16-activation runtime.

The tested expert-cache budgets were 6 MiB and 1 GiB. One Ring expert occupies
6 MiB. The cache keeps equal-size host and GPU payload copies, reported
separately. The GPU matrix workspace reserves 644,474,880 bytes, in addition to
the expert cache and CUDA overhead. The whole checkpoint is mapped on the
host; its roughly 30.28 GiB payload is never loaded onto the GPU at once.
Host mmap pages, working buffers and one incoming expert are additional host
memory. The cache budget is not a total-process RAM or VRAM measurement.

## Boundaries and next work

This is an opt-in reference executor, not a production serving integration.
It accepts token IDs and writes internal states, logits and routes. It has no
tokenizer front end, generation scheduler, server endpoint or training path.
Although the executable permits more positions, only the stated short windows
were qualified here. No throughput improvement or answer-quality gain is claimed.

The production runtime and its defaults were unchanged. No serving process was
stopped for this qualification. No research model was promoted. Ling has not
been implemented or executed in this phase.

Keep this original-model comparator as the acceptance gate while adding cached
generation and longer-position tests. Profile dense-weight transfers, expert
cache misses and CPU attention before choosing further acceleration. Ling's
KDA/MLA state and routing need their own implementation and original-model
comparison; passing Ring does not qualify Ling.
