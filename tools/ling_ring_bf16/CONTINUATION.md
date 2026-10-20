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

The initial implementation passed CPU integration controls and all four builds. Live qualification and fresh held-out results are recorded separately under `/work/AI/experiments/strata-continuation-qualification-20261020-r1` and `/work/AI/experiments/strata-continuation-heldout-20261020-r1`; consult their final result files rather than treating this preparation note as a GPU pass. No production serving or capability-gain claim follows from state extraction alone.
