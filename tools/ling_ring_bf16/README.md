# Strata Ling/Ring BF16 streaming reference

This directory is the opt-in reference path for running the Ring-mini-2.0
architecture from original BF16 checkpoint weights on a host plus a bounded GPU
working set. The implementation uses Strata's BF16 GEMV kernel for GPU matrix
products, keeps activations in FP32, and uses a bounded BF16 expert cache. The
cache budget caps the GPU payload. Receipts separately report the equal-size
host payload copy; those two numbers must not be counted as one allocation.
The original FP32 routing bias remains FP32. No model quantization or checkpoint weight
conversion is performed.

The current Ring reference executor includes:

- Ring GQA attention with per-head Q/K RMS normalization, partial RoPE, causal
  KV state, and FP32 softmax/reference arithmetic.
- Ring grouped top-k routing with FP32 sigmoid scores, original FP32 expert
  bias, deterministic selection, and expert-weight normalization.
- Original BF16 matrix projections through Strata's existing BF16 GEMV kernel,
  including shared and routed SwiGLU experts.
- Bounded expert residency with host backed payloads, GPU upload leases, LRU
  eviction, and explicit byte accounting.
- A token-ID full-forward reference executor that writes hidden states, logits,
  route traces, and an execution receipt.

The executors are opt-in reference tools and do not provide production
server/model-loader integration. Both have greedy frontends using their
original tokenizer and template. The current numerical investigation is
recorded in [NUMERIC_PARITY_DIAGNOSTICS.md](NUMERIC_PARITY_DIAGNOSTICS.md),
with exact receipt pins in
[CURRENT_NUMERIC_QUALIFICATION.json](CURRENT_NUMERIC_QUALIFICATION.json).

The composed Release profiles now pass the retained Ring 1,031-position and
Ling 195/358-position comparisons, plus the sampled fresh traces recorded in
CURRENT_NUMERIC_QUALIFICATION.json. Earlier failures below are historical.
The cached original-source reference widens stored BF16 weights to FP32; this
is not parity to the vendor's native BF16 runtime. The completed ten-case
quality pilot remains negative: Ring 5/10 and Ling direct mode 1/10. Neither
model is admitted for production serving on this evidence.

Opt-in output-head residency (`--lm-head-residency cuda`) preserves original
BF16 bytes and the existing arithmetic, using an additional 614 MiB on Ring or
460.5 MiB on Ling. Defaults remain off. Retained head-enabled traces and actual
low-memory refusals passed. `--hash-workers 1..4` provides bounded parallel
whole-file checkpoint hashing; default 1 preserves serial verification. Both
full before/after passes remain mandatory, including the hashing helper's own
source binding. Four-worker full-checkpoint verification and four-position
native replay passed for both models; no model conversion is involved.

Successful wrapper readmission used per-job `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `OMP_WAIT_POLICY=PASSIVE`.
These are explicit experiment settings, not a global runtime change. The
preceding unconstrained Ring run timed out; that alone does not establish a
causal threading diagnosis. Source, environment, failed attempts and exact
receipts are recorded in the current qualification document.

The two-case Ling direct-mode ABBA smoke preserved identical output IDs:
1.15x native and 1.10x whole-case observed throughput, but zero correct answers
in either mode. It therefore demonstrates no verified-work gain. Ring timing
was interrupted after a shared-host load/I/O spike and remains inconclusive.
On one consumed arithmetic case, enabling documented default thinking changed
an incorrect 81 to the correct 79 with valid EOS receipts. Fresh paired
validation is a separate gate, not supplied by this example.

Author attribution: AI Identity Incorported <james@aiii.id>

## Evidence boundaries

inspect_checkpoint.py is CPU-only and stdlib-only. It reads config.json, the
safetensors index, tokenizer JSON metadata, and safetensors headers. It does
not read tensor payload bytes, import MLX or torch, start a service, connect
over SSH, or establish Strata/CUDA/RTX readiness. Its report is a structural
inspector/header record, not a payload-integrity result.

~~~sh
python3 inspect_checkpoint.py /path/to/checkpoint -o report.json
python3 inspect_checkpoint.py /path/to/checkpoint --compatibility request.json
~~~

export_expert_fixture.py is the small-fixture payload reader. After a fresh
inspector report, it reads exactly layer 1, expert 0 gate_proj, up_proj, and
down_proj tensors, requiring BF16 and the Ling/Ring shapes, rechecking config
and shard/header pins, applying Darwin F_NOCACHE, and writing exact raw bytes
plus offsets and SHA-256 values to manifest.json. It refuses an existing
output directory and never converts or quantizes data. The fixture is a sample
of three tensors, not full-checkpoint integrity or a GGUF/Strata conversion.

~~~sh
python3 export_expert_fixture.py \
  --checkpoint /path/to/checkpoint \
  --report report.json \
  --output-dir /path/to/new/fixture \
  --family ling                         # or ring
~~~

Run the source-level Python tests with:

~~~sh
PYTHONPATH=tools/ling_ring_bf16 python3 -m pytest tools/ling_ring_bf16/tests -q
~~~

## Standalone CMake build

The standalone CMake project builds the Ring CPU reference components and their
CPU tests without starting a service or loading model weights. Use `/usr/bin/cmake`
for the build and test commands:

~~~sh
/usr/bin/cmake -S tools/ling_ring_bf16 -B /tmp/strata-ring-cpu
/usr/bin/cmake --build /tmp/strata-ring-cpu
/usr/bin/ctest --test-dir /tmp/strata-ring-cpu --output-on-failure
~~~

The opt-in CUDA build adds the token-ID reference executor and targets CUDA
architecture 89. It requires a local CUDA toolchain and a local Strata source
tree; it does not change production defaults:

~~~sh
/usr/bin/cmake -S tools/ling_ring_bf16 -B /tmp/strata-ring-cuda \
  -DRING_ENABLE_CUDA=ON
/usr/bin/cmake --build /tmp/strata-ring-cuda
~~~

The CUDA executor's bounded cache and workspace checks fail before execution
when the requested GPU budget cannot fit. It never stops another process.
CUDA execution is hardware qualification work and is separate from CPU tests.

The optional Ring `fp32-cublas` backend is a diagnostic comparison only. It
widens each resident BF16 matrix into a temporary FP32 buffer of
1,287,651,328 bytes (about 1.2 GiB), uses pedantic cuBLAS math with atomics
disabled, and keeps the normal bounded expert cache. It is not an optimization
or promotion claim. The frozen external protocol, source bindings, and result
records are under
`/work/AI/experiments/ring-cublas-qualification-20261016`; the candidate
executable SHA-256 is
`30df55ecd84b668b2e33d4d9510085d23e4757694e89941c9fd4b4e4848d5001`.
The 4-position diagnostic passed the unchanged strict gate; the 173-position
diagnostic reproduced the default backend's position-172/layer-7 route flip.
Longer cuBLAS cases were not run after that failure. A separate 4-position
default-backend regression subsequently passed bitwise against its frozen
outputs; its result is retained outside the source tree.

## Run and compare Ring

Run these commands from the repository root. Use fresh output directories and
the original, trusted Ring checkpoint, including its Python model and
configuration source files. No checkpoint copy or conversion is required.

~~~sh
python3 tools/ling_ring_bf16/prepare_ring_manifest.py \
  /path/to/Ring-mini-2.0 /tmp/ring-admission
python3 tools/ling_ring_bf16/run_ring_verified.py \
  --checkpoint /path/to/Ring-mini-2.0 \
  --manifest /tmp/ring-admission/manifest.tsv \
  --executable /tmp/strata-ring-cuda/strata_ring_bf16 \
  --tokens 156890,128,256,1024 --cache-mib 1024 \
  --output /tmp/ring-native
~~~

The verified launcher rebuilds the manifest, requires exact agreement, and
hashes all checkpoint shards, the configuration, the original model sources,
the executable, and the relevant implementation sources before and after the
run. It binds those hashes to the actual output bytes. Hashing the checkpoint
reads all payload bytes; allow for that CPU and storage work. The raw executable
is useful for development, but an unwrapped run cannot pass the comparator.

The original Ring reference runs the checkpoint's own decoder classes one
layer at a time, with original BF16 values widened exactly to FP32. It supports
CPU and CUDA reference execution. Ling's reference runner uses the same
one-layer residency and widened-FP32 policy; its model forward requires CUDA
with the installed FLA path, while CPU coverage is limited to import/meta,
report-contract, and negative CLI tests. The measured four-token Ling GPU
forward and 5 MiB cache replay are recorded in `LING_VALIDATION.md`.
Ling CUDA execution requires at least 6 GiB free before execution and rejects a
reference peak above 5 GiB. CUDA and cuDNN TF32 flags are disabled for the host/runtime configuration.
This does not establish Triton/FLA chunk-kernel precision: the installed FLA
chunk_intra.py contains explicit TF32 solve constants. These runners execute
trusted checkpoint Python code. The comparator needs NumPy; the
launchers are stdlib-only.

~~~sh
python3 tools/ling_ring_bf16/ring_original_reference.py \
  --checkpoint /path/to/Ring-mini-2.0 \
  --source /path/to/Ring-mini-2.0/modeling_bailing_moe_v2.py \
  --tokens 156890,128,256,1024 --threads 4 --output /tmp/ring-original
python3 tools/ling_ring_bf16/compare_ring_reference.py \
  --reference /tmp/ring-original --candidate /tmp/ring-native \
  --output /tmp/ring-comparison
~~~

## Ling original reference

The Ling runner streams the original 24 decoder layers one at a time, widens
BF16 checkpoint tensors to FP32, records all hidden states, logits, and routed
expert traces, and emits comparator-compatible provenance. It covers the
source KDA/MLA implementation and the 24-layer geometry. The separate verified
Ling generation frontend applies the original tokenizer and chat template,
performs greedy decoding, binds EOS configuration, and compares generated
choices against the original reference logits. Its completed cases and limits
are recorded in `LING_VALIDATION.md`; this remains bounded evidence rather than
production serving or broad answer-quality qualification.

Use the isolated reference environment prepared for Ling:

~~~sh
/work/AI/experiments/ling-strata-20261016/reference-env/bin/python \
  tools/ling_ring_bf16/ling_original_reference.py \
  --checkpoint /path/to/Ling-3.0-tiny \
  --source /path/to/Ling-3.0-tiny/modeling_bailing_moe_v3.py \
  --tokens 1,2,3 --device cuda --logits-mode all \
  --output /tmp/ling-original
~~~

The focused CPU tests use the same isolated environment:

~~~sh
OMP_NUM_THREADS=4 /work/AI/experiments/ling-strata-20261016/reference-env/bin/python \
  -m pytest -q tools/ling_ring_bf16/tests/test_ling_original_reference.py
~~~

## Greedy generation frontend

`generate_ring.py` is the local text frontend for the verified Ring executor.
It loads the original checkpoint tokenizer and applies its chat template, then
binds `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`,
and any available `special_tokens_map.json` through the verified-run receipt.
The frontend uses greedy next-token selection and prints the decoded generated
text. It does not download tokenizer files or use an external model.

The generation request requires the checkpoint's configured EOS IDs and a
positive `--max-new-tokens` value. Generation stops at the first configured
EOS token or at the requested length, and the prompt plus generated sequence
must fit the 4096-token evaluation window. The native trace records the prompt,
generated IDs, EOS configuration, budget, and finish reason; the comparator
derives the expected greedy IDs from the original reference logits.

~~~sh
python3 tools/ling_ring_bf16/generate_ring.py \
  --checkpoint /path/to/Ring-mini-2.0 \
  --manifest /tmp/ring-admission/manifest.tsv \
  --executable /tmp/strata-ring-cuda/strata_ring_bf16 \
  --prompt "Write a short greeting." \
  --output /tmp/ring-generation \
  --cache-mib 1024 --max-new-tokens 128
~~~

Generation output and tokenizer decoding are frontend evidence. They do not
establish native BF16 activation equivalence, generated-answer quality, or
production serving readiness.

This is a numerical architecture gate against original weights and original
model code in a common FP32 internal activation and trace domain. It is not a
claim of native BF16 activation equivalence, unrestricted context support,
generated-answer quality, or production serving readiness. Reference
receipts bind both original modeling and configuration sources; the original
reference report records both source hashes and the verified wrapper binds them
to the native run.
The receipt and source files are operator-controlled evidence, not signatures
or protection against a malicious evaluator modifying every record together.

Every layer, final hidden state, and logit must satisfy
`abs(candidate-reference) <= 0.001 + 0.001*abs(reference)`. All selected expert
IDs and every position's top-1 token must match exactly. Route weights use
`1e-5 + 1e-4*abs(reference)`. Missing, malformed, nonfinite, or mismatched
artifacts fail. See `RING_VALIDATION.md` for measured coverage and limitations.

## BF16 composition probe

bf16_stream_probe.cu uses Strata's existing bf16_gemv_fp32_mmvf for gate, up,
and down, with a local SwiGLU kernel. It uses synthetic BF16 expert arrays by
default, or exact exporter fixtures with `--fixtures`. The synthetic dimensions
are Ling (n=1536, ff=512) and Ring (n=2048, ff=512); fixture files must be named
`ling-gate.bf16`, `ling-up.bf16`, `ling-down.bf16` and the corresponding
`ring-*` files.

The probe performs serial host-to-device copies through two pinned host buffers,
reads each weight back for byte identity, then makes three compute calls and
compares against an independent CPU FP32 oracle. The acceptance check is
`abs <= 1e-5 + 1e-4 * abs(reference)`, with finite-output and bitwise-reuse
checks. `--mutate-copy` and `--mutate-output` are negative validator paths.
This probe is a component check; it is not a full Ring executor validation
receipt.

From this directory, compile against a local Strata checkout. The command is
portable because `STRATA_ROOT` is supplied by the operator:

~~~sh
export STRATA_ROOT=/path/to/strata-nvfp4
nvcc --gpu-architecture=sm_89 -std=c++17 -O2 \
  -Xcompiler=-ffunction-sections -Xlinker=--gc-sections \
  -I"$STRATA_ROOT/include" \
  bf16_stream_probe.cu "$STRATA_ROOT/src/kernels/cuda/native_bf16.cu" \
  -o bf16_stream_probe
./bf16_stream_probe
./bf16_stream_probe --fixtures /path/containing/both/fixture/sets
./bf16_stream_probe --mutate-copy
./bf16_stream_probe --mutate-output
~~~

The probe is bounded serial staging only. It does not qualify overlap, cache
replacement, H2D cache policy, model routing, attention, full-model loading,
full-model inference, training, or production offload of changing model state.
The reported GPU buffer footprint is the probe allocation, excluding CUDA
context overhead; it is not a model-memory budget. Operators must verify
fixture SHA-256 values against their export manifests before execution.

## Architecture boundary

Ring and the Ling original-model reference are implemented as opt-in tools. Ling
has bounded four-token, current cached 65-token, and 511-token long-context
GPU evidence. The historical batched standard 65-token comparison failed with
319 ordered-route differences; the forced-recurrent diagnostic is separate and
diagnostic-only. Five frozen generation cases completed: four strict
recompares passed, while thinking-on retains one ordered expert-route swap
although its independent decoded answer was 437. The one-token length-control
case passed. These are bounded forward, parity, and narrow functional controls;
they do not qualify broad answer quality, production serving, throughput, or
production defaults. Ring's separate generation frontend uses the original
checkpoint tokenizer and chat template, as described above. The existing
production Qwen runtime files and defaults remain unchanged.
