Historical component gate; current integration scope: [UPSTREAM_QUALIFICATION_20261020.md](UPSTREAM_QUALIFICATION_20261020.md).

# BF16 expert component qualification

Measured on 16 October 2026 with NVIDIA RTX 4070 Ti (sm_89), driver 595.91.07,
CUDA compiler 13.1.115. Strata base source:
`3fa66b4379e7e2c0af349718d9004bd28005f975`.

The target is original BF16 checkpoint weights in host RAM, streamed into a
bounded GPU working set. Original FP32 tensors must also retain their precision
when a model loader is added. No model quantization is part of this work.

## Executed checks

- CPU checkpoint inspection and fixture-export tests passed on macOS and Linux.
- CUDA probe compiled against Strata's existing `native_bf16.cu` kernel.
- Synthetic expert weights passed both model geometries.
- Exact layer 1, expert 0 gate/up/down weights from both original checkpoints
  passed with synthetic activations. Sample hashes were checked before and after.
- Host-to-device-to-host weight bytes were identical.
- Three identical compute repetitions produced bitwise-identical outputs.
- Copy-result and output-result corruption controls failed as intended for both
  synthetic and checkpoint weights. These are validator controls, not injected
  hardware failures.

| Checkpoint sample | Weight bytes | Maximum absolute output error against CPU FP32 |
| --- | ---: | ---: |
| Ling-3.0-tiny | 4,718,592 | 8.195638657e-8 |
| Ring-mini-2.0 | 6,291,456 | 2.235174179e-8 |

Acceptance was declared before execution: every output must be finite and have
`abs(error) <= 1e-5 + 1e-4 * abs(reference)`. This is a combined tolerance, not a
standalone maximum-relative-error threshold. Near-zero reference values make
maximum relative error misleading; the raw values remain in the receipt.

The probe allocated 11,046,912 GPU buffer bytes and 4,194,304 pinned-host bytes,
excluding CUDA context/driver overhead. Source BF16 values were copied without
conversion; multiplication and accumulation used FP32 activations.

Receipt: [qualification/bf16-component-receipt-r4.json](qualification/bf16-component-receipt-r4.json).
It binds the probe source and executable hashes to results and fixture hashes.
The checkpoint payload samples themselves are not included in this repository.

## Boundaries

This is one expert per model, one synthetic activation vector each, serial
staging, and one hardware configuration. It does not exercise native model
activations, routing, attention, expert combination, cache replacement,
asynchronous overlap, full-model inference, gradients, training, or capability.
Recorded timings include first-call initialization and validation transfers;
they are not a throughput benchmark and must not be extrapolated to tokens/s.

The existing Qwen runtime files and defaults were not edited. The probe is an
opt-in CUDA tool and is not linked into the production engine. HIP and SYCL
paths were not modified or qualified. Full Ring and Ling architecture support
remains to be implemented and tested before either model can run in Strata.
