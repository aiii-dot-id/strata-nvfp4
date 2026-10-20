# Complete Ring and Ling continuation

AI Identity Incorported <james@aiii.id>

This opt-in research interface saves actual live model state and resumes without replaying its committed prefix. It supports independent forks and rollback by restoring a preserved ancestor snapshot. Existing `generate_ring.py`, `generate_ling.py` and ordinary verified forward commands remain the default interfaces. No quantization or model-weight update is performed.

## State contract

A snapshot is a directory containing exactly `state.bin` and `manifest.json`. The payload contains the committed token history, absolute position, last-position logits and every persistent attention state. Ring includes all 20 K/V cache pairs. Ling includes six MLA K/V pairs and all four state vectors (recurrent, Q/K/V convolution) for each of its 18 KDA layers: 84 named arrays in total. Unused capacity is excluded; restore allocates the current request's bounded capacity.

The manifest binds the checkpoint/configuration/manifest, local tokenizer and original-source files, executable, relevant source and runtime libraries, arithmetic profile, EOS contract and absence of adapters. Prefix history must agree exactly with the committed part of the requested token sequence. Future tokens may differ deliberately for a fork. Incompatible model, source, executable, tokenizer, profile, corrupted payload, missing component, invalid shape, position and nonfinite state are rejected. Native CRC32/IEEE and wrapper SHA-256 checks protect integrity; they are not an authorization or authenticity mechanism.

The sampler is greedy, so no random-number state is needed. A snapshot ends before sampling the token following its last committed position. On resume the sampler reconstructs the previous generated prefix and samples that pending token exactly once from retained logits. The terminal sampled EOS/length token remains uncommitted, matching the existing executable's generation contract. Empty and completed snapshots are supported; completing an already finished snapshot does not evaluate another model position.

Expert LRU contents, output-head residency and temporary workspaces are performance caches and are rebuilt in each process. Snapshots therefore save prefix computation, not all process setup. Both full before/after checkpoint verification passes remain mandatory per invocation. The qualified arithmetic uses original BF16 weights widened to FP32; this is not a native vendor-BF16 equivalence claim.

## Use

Run on dev8 with its admitted Python environment and the exact qualified executable. The experiment driver owns the shared GPU lock, sustained resource checks, foreign-process identity checks, deadlines and cleanup. This tool never stops a service or another process. Do not use the Mac serving runtime for this work.

Example prefill snapshot on Ling (development prompt only):

```sh
/work/AI/experiments/ling-strata-20261016/reference-env/bin/python tools/ling_ring_bf16/run_continuation.py --model ling --checkpoint /work/AI/checkpoints/Ling-3.0-tiny --manifest /work/AI/experiments/ling-strata-20261016/admission/manifest.tsv --executable /path/to/qualified/strata_ling_bf16 --prompt 'Compute 14*6 - 17.' --thinking on --max-new-tokens 512 --stop-after 8 --output /new/evidence/prefix
```

`--stop-after` is an absolute committed-position limit, including zero for an empty snapshot. It must not precede the loaded position. Omitting it runs to the normal EOS or token limit. Every invocation of this continuation tool writes a new snapshot; ordinary generation uses the existing generation tools. `--tokens` accepts an exact token sequence instead of rendering a prompt. A rendered prompt plus output budget must fit the local executor's bounded window: 1,024 total positions on Ling and 4,096 on Ring, not the checkpoint's advertised training window.

Fork on CPU without model or GPU execution:

```sh
python3 tools/ling_ring_bf16/run_continuation.py --fork-source /new/evidence/prefix/snapshot --fork-destination /new/evidence/fork-a
```

Resume with the same model/executable/profile and request, adding `--state-in /new/evidence/fork-a` and choosing a new output directory. Change only future tokens if deliberately testing a branch. Input snapshots are read-only to the tool and checked again after execution. Forks copy payload bytes into independent files; changing a child does not change its parent. Partial bundles and existing destinations fail closed.

Rollback means choosing the preserved ancestor snapshot as the next `--state-in`. It restores model history and attention memory; it does not reverse external file, network or tool effects. This is a one-shot continuation interface, not a persistent serving daemon or an AII OS identity snapshot.

## Outputs and validation

`continuation.json` records the start, final committed position, number of newly evaluated positions and pause status. Layer states, final hidden states, logits and ordered routes cover only the newly evaluated suffix, using absolute positions. A native zero-work operation emits explicit empty traces; the wrapper never invents missing native data. `execution.json` records the full committed token prefix and the suffix logit count. A partial generation has no completed `generation.json`. `provenance.json` is written only after input/output and state validation succeeds.

Whole-model acceptance compares uninterrupted execution with concatenated prefix/suffix outputs, including every layer, logits and ordered routes. It separately checks empty state, distinct forks, parent isolation, rollback, rejected bindings/corruption, mid-generation resume and terminal EOS/length behavior. Build/CPU checks alone do not establish this acceptance.

## Generation-logit readout

`generation_readout.py` is a CPU-only, post-execution audit helper. It memory-maps a retained `logits.f32` trace, checks its sealed `provenance.json` and `execution.json`, then verifies that each recorded generated token is the deterministic argmax of its corresponding logits row. It does not run a model, regrade an answer, establish semantic quality, or measure serving/decode performance. It rejects resumed suffix traces in this initial form and writes its report outside the input artifact.

Example, after the producing receipt has been retained and its provenance digest recorded:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 tools/ling_ring_bf16/generation_readout.py /new/evidence/full-generation \
  --expected-provenance-sha256 <sealed-provenance-sha256> \
  --output /new/evidence/generation-readout.json
```

The 2026-10-20 qualified baseline artifacts produced sealed alignment evidence for Ling (146 generated EOS-terminated tokens) and Ring (347 generated EOS-terminated tokens). This is trace alignment evidence only; it makes no capability, semantic-quality, native-original equivalence, throughput, or production-serving claim.

## Qualified scope on 2026-10-20

The full-model structural gate passed 28/28 process cases. The fresh pilot completed 32 logical arms: all 16 uninterrupted/resumed generated-token pairs matched, and an independent audit compared all 400 retained layer/logit/route payload splices bitwise. Ring retains 40 state arrays; Ling retains 84.

Strict answer quality was **0/8 per model in each arm** under the frozen response contract and 512-token allowance. Ring exhausted that allowance on all eight tasks. Ling reached EOS on three tasks with correct values but omitted the required `FINAL:` prefix; five tasks exhausted the allowance. These are strict usability results, not evidence that every computed value was wrong. Continuation did not improve answers and no capability or speedup promotion was made.

One original arm stopped because the observer read an exited process. The narrow observer repair was CPU-tested and qualified during the remaining GPU runs. Its failed attempt remains invalid and preserved; its time and the entire repair pause are charged in the primary efficiency metrics. The retry received only the remaining active allowance, an explicitly recorded interruption exception.

The GPU-tested model implementation is `b0338fc59ef77271fae2e8893259ae31c4713219`. The later CPU readout, observer repair, tests and documentation do not change model arithmetic. See [the qualification index](CONTINUATION_QUALIFICATION.json) for exact evidence hashes. Full reports remain under `/work/AI/experiments/strata-continuation-report-20261020-r1` and `/work/AI/experiments/strata-continuation-final-analysis-20261020-r1`; the original failed campaign remains separate from its recovered result.

All model execution was on dev8. There was no quantization, weight update, service change, or Mac/oMLX/Sev interruption. This gate validates inference state and trace alignment; it does not qualify training, semantic completion, recurrence gains, a persistent server, or the full advertised checkpoint context window.
