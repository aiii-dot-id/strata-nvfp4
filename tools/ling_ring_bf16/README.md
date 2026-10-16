# Strata Ling/Ring BF16 streaming preflight

This directory documents a bounded, lossless BF16 preparation path for Strata on
an RTX 4070 Ti (sm_89). It does not add a model loader, GGUF converter,
quantizer, router, attention implementation, or inference service. The target is
original BF16 weights held on the host and streamed to a limited GPU working set.

## Evidence boundaries

inspect_checkpoint.py is CPU-only and stdlib-only. It reads config.json, the
safetensors index, tokenizer JSON metadata, and safetensors headers. It does not
read tensor payload bytes, import MLX or torch, start a service, connect over
SSH, or establish Strata/CUDA/RTX readiness. Its report is a structural
inspector/header record, not a payload-integrity result.

~~~sh
python3 inspect_checkpoint.py /path/to/checkpoint -o report.json
python3 inspect_checkpoint.py /path/to/checkpoint --compatibility request.json
~~~

export_expert_fixture.py is the only bounded payload reader here. After a fresh
inspector report, it reads exactly layer 1, expert 0 gate_proj, up_proj, and
down_proj tensors, requiring BF16 and the Ling/Ring shapes, rechecking config and
shard/header pins, applying Darwin F_NOCACHE, and writing exact raw bytes plus
offsets and SHA-256 values to manifest.json. It refuses an existing output
directory and never converts or quantizes data. The fixture is a sample of three
tensors, not full-checkpoint integrity or a GGUF/Strata conversion.

~~~sh
python3 export_expert_fixture.py \
  --checkpoint /path/to/checkpoint \
  --report report.json \
  --output-dir /path/to/new/fixture \
  --family ling                         # or ring
~~~

Run the source-level tests with:

~~~sh
python3 -m unittest discover -s tests -v
~~~

## Strata BF16 composition probe

bf16_stream_probe.cu uses Strata's existing bf16_gemv_fp32_mmvf for gate, up,
and down, with a local SwiGLU kernel. It uses synthetic BF16 expert arrays by
default, or exact exporter fixtures with --fixtures. The synthetic dimensions
are Ling (n=1536, ff=512) and Ring (n=2048, ff=512); fixture files must be named
ling-gate.bf16, ling-up.bf16, ling-down.bf16 and the corresponding ring-* files.

The probe performs serial host-to-device copies through two pinned host buffers,
reads each weight back for byte identity, then makes three compute calls and
compares against an independent CPU FP32 oracle. The actual acceptance check is
abs <= 1e-5 + 1e-4 * abs(reference), with finite-output and bitwise-reuse
checks. --mutate-copy and --mutate-output are negative validator paths.
Measured component results and their limits are recorded in
[QUALIFICATION.md](QUALIFICATION.md).

From this directory, compile against a local Strata checkout. The command is
portable because STRATA_ROOT is supplied by the operator:

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

The probe is bounded serial staging only. It does not qualify overlap,
eviction, H2D cache policy, model routing, full-model loading, full-model
inference, training, or production offload of changing model state. The reported
GPU buffer footprint is the probe allocation, excluding CUDA context overhead;
it is not a model-memory budget. Both families' fixture files must be placed in
the directory passed to --fixtures. Operators must verify their SHA-256 hashes
against both export manifests before execution; the CUDA probe checks file
sizes but does not itself parse or authenticate those manifests.

## Next implementation boundary

The smallest architecture sequence is Ring's GQA attention and routing path
first, followed by Ling's KDA/MLA path. Each requires its own geometry,
tensor-layout validation, attention state, routing/top-k semantics, and BF16
expert dispatch. The probe only establishes the reusable BF16 matrix composition
and bounded host-to-device ownership pattern; it is not evidence that either
architecture is already supported by Strata.
