# Upstream integration qualification — 2026-10-20

AI Identity Incorported <james@aiii.id>

## Source state

`origin/main` `57293b8ec57203a741dc7b2abd69f24f077037ab` is merged into the integration branch. The earlier integration source is `22942f42038083cf063c48d8ae13093d94c9e2c7`; its merge ancestry includes `b1063847c383745acc50b286725e00adc24377ba` (the zero-conflict upstream merge), `4b269bf737726803a695bd0e9616017d6b407df1`, and the frozen pilot base `763f5161f45271bb31e914474e2a0f1af30d1d28`.

Between `b1063847` and `22942f`, the only source additions were the offline `tools/ling_ring_bf16/cache_trace/` analyzer and its tests, plus static UI/no-acquisition test changes. Ring and Ling engines, their headers, standalone Ling CMake, verified wrappers, Torch backends, and `src/kernels/cuda/native_bf16.cu` were unchanged from the frozen qualification closure at that stage. The later accounting implementation `40842a43` changes only logical-state reporting and its controls; the publication also adds the passive observer and documentation. Its separate qualification is recorded below.

The upstream boundary is not wholly byte-identical: `include/strata/kernels/bf16_gemv.hpp` adds an eight-line `bf16_gemv_pair` declaration. The current Ring/Ling engine sources do not reference it, and it was not enabled. Other shared upstream kernel and root-CMake changes are recorded in `FINAL-SOURCE-CLOSURE.txt`; this is source-closure evidence, not a new GPU replay.

## CPU evidence

The earlier clean integration source (`22942f`) passed the rerun recorded at `/work/AI/experiments/strata-upstream-final-cpu-20261020-r3/RESULT.json`: 227 pytest items passed, with 9 subtests passed, return code 0, source status empty, `CUDA_VISIBLE_DEVICES` disabled, and 39.775761941 seconds wall time. The exact log SHA-256 is `63dacdebce7dd8d45cbe98f82321df6fee21b791b440336677a681d1deaa857d`.

The preceding CPU-only build receipt from the merged source `b1063847` is in `/work/AI/experiments/strata-upstream-integration-20261020-r1/`. It built with `CUDA_VISIBLE_DEVICES=''`, CUDA 13.1 compiler, and at most two build jobs; CTests passed for Ring no-Torch (6), Ring Torch (7), Ling no-Torch (2), and Ling Torch (3). This compiles CUDA targets but does not execute model inference or a GPU qualification run.

## Binary identities

The build receipt records these exact executable SHA-256 values:

| Variant | SHA-256 |
| --- | --- |
| Ring no-Torch | `e27d766c849ec71528de5fd06cf03c2081354fce1b9106ff979f3547ac7db860` |
| Ring Torch | `63c206ea61b07229c3a847df393fe1fba3578d626587a79073958cd83b24985b` |
| Ling no-Torch | `9fde48f9bf80c8ee6bb28a58cc35b5550887ae54ac98b156972c9ce26b9efe03` |
| Ling Torch | `dd1a638dedae9bfb761792b7265280166305c4695624377c4a9160965d27a8e9` |

The Torch binary hashes differ from the former GPU-qualified Ring (`f43a1ac15e4e2aa23b1927d775caf179073fe1562e5b82635f1d232072ff3704`) and Ling (`d5c6f61093dd3df45cb4efd6a70be664f18c1ce6017acaa835d244dd31a57463`) binaries. Therefore no GPU-parity result is carried forward by executable identity.

## New GPU replay

Both rebuilt Torch executables passed a four-position replay against retained native outputs. Every layer state, final hidden state, full-vocabulary logits and ordered route bytes matched bitwise (76 Ring and 92 Ling routed-layer positions). Full checkpoint and manifest digests matched the older admitted records after whole-file before/after hashing with four workers. The source stayed clean at `22942f` throughout. No weights, services or foreign GPU processes were changed.

Replay result: `/work/AI/experiments/strata-upstream-native-replay-20261020-r1/functional-r1/RESULT.json`, SHA-256 `a4354c1a716a806485ac0b3c8fd4e77ea9cfb743cc84cf17682aabc5494e301d`.

All six selected upstream CPU framework smoke groups passed at clean `22942f`: fork-update mocks, release config, GPU fit, no-acquisition controls, launch plans/config and local inventory/import plans. Receipt: `/work/AI/experiments/strata-upstream-framework-smoke-20261020-r2/RESULTS.json`, SHA-256 `dbc93a2cc6b19922f93a4b7a8b1152ffb219e22fc64e629de36e17e480f95b1c`.

## Longer replay and generation qualification

After the four-position gate, clean source `9ff03254940470e4b6b860ae31e745104244795d` passed full retained replay at Ring-1031 and Ling-195/358 on the rebuilt executables. All 23 Ring or 27 Ling stable native payloads matched the prior native path bitwise. Each direct cached-original comparison passed with zero ordered-route mismatches and exact top-1 under the unchanged declared float tolerances. Both checkpoint payloads are original BF16 widened to FP32 for this reference domain.

A separate frontend generation gate reproduced all retained prompt/generated IDs and EOS behavior. Ring generated 347 tokens including EOS and evaluated 375 positions; Ling generated 146 and 320 including EOS and evaluated 195 and 358 positions. Current checkpoint tokenizers reproduced the retained prompts and stop IDs on CPU before GPU execution.

## Ling logical state accounting repair

Implementation source `40842a43e6e6e98932786ca3de8bb9c14ed8bf1d` repairs a reporting gap without changing decoder arithmetic. It derives live and capacity logical-payload bytes from each actual KDA/MLA state object. The legacy live-use field remains compatible. Both no-Torch and Torch builds passed their CPU CTests (2/2 and 3/3).

The new Ling executable SHA-256 is `9ddbf37232ff4add1c7a07a46366315031fb9b25547140023eaacb0de862c086`. Its four-position replay and 195-position EOS generation passed, with all 27 numerical payloads identical to their retained native baselines. The generation also passed the cached-original comparator: zero route mismatches and exact top-1. Live/capacity bytes were 21,135,360/21,135,360 for four positions and 44,605,440/89,702,400 for early EOS. Both independent audits passed. The 358-position case was not rerun on this new accounting executable; its evidence remains bound to `dd1a638d`.

The Ring executable remains `63c206ea61b07229c3a847df393fe1fba3578d626587a79073958cd83b24985b`. The earlier Ling replay/generation executable remains identifiable as `dd1a638dedae9bfb761792b7265280166305c4695624377c4a9160965d27a8e9`.

## Profiling and cache-capacity outcome

The [passive wrapper observer](profiling/README.md) is now reusable tooling in this fork. Its exact shipped bytes were used in live Ring and Ling checks, with native observation and sampler shutdown required. It records polling envelopes and memory/host telemetry, not CUDA copy/kernel or exact prefill/decode timings. Shared-host observations cannot establish causal speedups.

On one retained Ling-195 trace, two GiB reduced misses 21.27% (23,638 to 18,610), exactly as the offline analyzer predicted, and preserved numerical payloads. Native wall time was slower, 57.214 versus 54.636 seconds at one GiB. Keep the one-GiB default. Fewer cache misses alone did not improve this measured workload.

## Qualification boundary

These results establish retained numerical and generation parity, bounded tool correctness, and memory-accounting repair in the declared widened-FP32 domain. They do not establish fresh answer-quality improvement, native vendor-BF16 equivalence, training readiness, complete continuation snapshots, throughput improvement, serving integration, production admission, or qualification of unrelated upstream quantized-model features. No weights, model precision, service or production defaults changed. Sev was untouched, Dawn remained active at the same PID, and the foreign GPU process identities were preserved.

Evidence paths and full hashes are recorded in [NEXT_GATE_20261020.json](NEXT_GATE_20261020.json), including independent reviews and the combined postflight.

## Capability gate and delivered tools

The frozen six-case paired Ling pilot used the unchanged research source and earlier qualified binary. Thinking on scored 4/6 versus 0/6 strict passes for thinking off, but the predeclared gate failed: one thinking-on case was wrong and one hit the native deadline. Two off-arm failures had correct content in the wrong format. This does not admit a production workflow or establish a general capability gain. The raw timeout counter omitted the native deadline; a separately pinned audit corrects the classification without changing the raw result.

The offline cache trace analyzer is now checked in under [cache_trace/](cache_trace/README.md), with exact one-GiB native counter calibration before any larger-cache or offline-oracle estimate. It does not change the cache policy or claim measured speed.

The detailed source-linked plan is [CAPABILITY_PLAN_20261020.md](CAPABILITY_PLAN_20261020.md). Source/receipt hashes are indexed in [UPSTREAM_INTEGRATION_20261020.json](UPSTREAM_INTEGRATION_20261020.json). The current publication adds separately qualified logical-state accounting, the exact live-used observer and documentation. Earlier replay evidence remains bound to its own implementation/executable identities; the accounting implementation source is `40842a43`. The frozen pilot source `763f5161` is preserved separately.

Final CPU integration: 229 tests and nine subtests passed with CUDA disabled. The receipt binds the `40842a43` engine source and exact observer/test additions before publication; it does not claim that tests ran on a later documentation commit.
