# Strata Ling/Ring BF16 streaming reference

This directory is the opt-in reference path for running the Ring-mini-2.0
architecture from original BF16 checkpoint weights on a host plus a bounded GPU
working set. The implementation uses Strata's BF16 GEMV kernel for GPU matrix
products, keeps activations in FP32, and uses a bounded BF16 expert cache. The
cache budget caps the GPU payload. Receipts separately report the equal-size
host payload copy; those two numbers must not be counted as one allocation.
The original FP32 routing bias remains FP32. No model quantization or weight
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

The executor is a reference tool. It does not provide the production server,
tokenizer, text generation loop, or production model-loader integration. Ling-
3.0-tiny is not implemented yet; its KDA/MLA path remains a later architecture
boundary. There are no timing comparison or throughput claims here.

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

The original-model reference runs the checkpoint's own decoder classes on the
CPU, one layer at a time, with original BF16 values widened exactly to FP32.
It executes the checkpoint's Python code, so the checkpoint must be trusted.
The measured reference used CPU torch 2.14.1 and transformers 4.52.3 in an
isolated environment. The comparator needs NumPy; the launcher is stdlib-only.

~~~sh
python3 tools/ling_ring_bf16/ring_original_reference.py \
  --checkpoint /path/to/Ring-mini-2.0 \
  --source /path/to/Ring-mini-2.0/modeling_bailing_moe_v2.py \
  --tokens 156890,128,256,1024 --threads 4 --output /tmp/ring-original
python3 tools/ling_ring_bf16/compare_ring_reference.py \
  --reference /tmp/ring-original --candidate /tmp/ring-native \
  --output /tmp/ring-comparison
~~~

This is a numerical architecture gate against original weights and original
model code in a common FP32 activation domain. It is not a claim of bitwise
agreement with the model's native BF16 activation path, unrestricted context
support, generated-answer quality, or production serving readiness. Reference
receipts bind the original modeling source; the wrapper also records the
configuration source, whose hash was not part of the original reference report.
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

Ring is the first implemented reference path. Ling requires a separate KDA/MLA
implementation, tensor-layout validation, attention state, routing semantics,
and BF16 expert dispatch. The existing production Qwen runtime files and
defaults remain unchanged; this directory is an opt-in reference tool.
