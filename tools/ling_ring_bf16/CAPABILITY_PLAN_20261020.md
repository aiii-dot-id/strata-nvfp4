# Ring and Ling capability improvements on the RTX 4070 Ti

AI Identity Incorported <james@aiii.id>

Reviewed 20 October 2026. Target: original BF16 Ring-mini-2.0 and Ling-3.0-tiny through our Strata fork on dev8. No quantization. Sev's Mac runtime is outside this work.

The strongest route is to establish a useful reasoning and verification baseline, remove measured memory and transfer waste, and then test a small learned latent refiner. Recent work supports all three directions, but does not establish a capability gain on our checkpoints. Simply repeating layers, allocating a larger context, or extracting internal states is not evidence of improvement.

## What our implementation actually does

The frozen research and pilot source is branch `ling-ring-bf16-lmhead-20261020` at `763f5161f45271bb31e914474e2a0f1af30d1d28`. C++ qualification binds `ddacebeb4d12e23aa70f8a1ee6ee8aef378e1224`; the Python verification update binds `2410912468ba152cc072b19129b9fd58ca66eb32`. The two successor commits `e7aaa5eb` and `763f5161` change only qualification documentation. The fork is [published here](https://github.com/aiii-dot-id/strata-nvfp4/tree/ling-ring-bf16-lmhead-20261020).

The live hardware inventory found a Ryzen 9 7950X3D, approximately 123 GiB of system RAM, and an RTX 4070 Ti reporting 12,282 MiB of GPU memory with PCIe 4.0 x16 negotiated. Available memory varies with other work. Dawn and the existing GPU processes remain running. Timing from this shared host is observational.

| Property | Ring mini 2.0 | Ling 3.0 tiny |
| --- | --- | --- |
| Decoder | 20 conventional attention layers | 24 layers, with 18 KDA and six gated MLA layers |
| Routed experts | 256 per routed layer, top eight | 128 per routed layer, top eight |
| Original BF16 routed expert payload | 28.5 GiB total; 6 MiB per expert | 12.9375 GiB total; 4.5 MiB per expert |
| Current evaluated context limit | 4,096 total positions | 1,024 total positions |
| Cache state needed for continuation | K/V, position, and token history at every layer | MLA caches and positions plus all KDA recurrent and convolution states |
| Reference used in existing parity work | Original-source execution with stored BF16 weights widened to FP32 | Same reference precision convention |

These evaluation limits are properties of our bounded executable, not the checkpoints' advertised training windows. The [Ring repository](https://github.com/inclusionAI/Ring-V2) lists a 32K native window and a separately configured 128K YaRN extension. Local Ling configuration declares 131,072 positions. Neither is a locally qualified long-context limit in this executor.

The inference programs do not expose autograd or an optimizer. Most ordinary matrix applications still upload stored BF16 weights and use FP32 working arithmetic in the qualified profile. That preserves the current numerical target but does not exploit a native BF16 tensor-core path. The original-source FP32 comparison is not a native vendor-BF16 behavior qualification.

Two optional improvements are implemented and boundedly qualified: resident BF16 output weights, and one to four parallel whole-file hash workers. Defaults remain unchanged. The resident head adds 614 MiB on Ring or 460.5 MiB on Ling. Full checkpoint hashing remains mandatory before and after execution. Retained head-enabled layer/logit/route outputs matched the retained native path bitwise; Python controls passed 201 tests. This is tooling evidence, not a broad answer-quality result.

## Context management that fits this machine

**Start with reversible rendering and evidence retrieval.** [ReFold, published 6 October 2026](https://arxiv.org/abs/2610.07863), preserves the underlying interaction history while folding or replacing material in the rendered context. Chunked rendering also addresses prefix-cache invalidation. Its frontier-model results are not small-model evidence. The useful local proposal is narrower: retain immutable tool results, render bounded excerpts and references, and support exact restoration before making a context-saving claim.

**Judge compaction by what happens next.** [TRACE, revised 14 October 2026](https://arxiv.org/abs/2608.06503), tests compaction events with paired continuations from the same environment state. This matches our lessons program: a summary that sounds accurate can still lose a constraint that changes later behavior. Preserve permissions, unresolved failures, task constraints, and provenance; compare later verified task outcomes rather than summary similarity.

**Do not assume that adding memory tools teaches their use.** [ACM, 26 July 2026](https://arxiv.org/abs/2607.23809), combines context-editing tools with post-training. Its own small-model case study reports failure to sustain useful search trajectories before context pressure arises. [HyMem, 16 August 2026](https://arxiv.org/abs/2608.15703), separates planning, execution, and analysis contexts. Both motivate structured context, but neither supplies Ring/Ling qualification. Begin with operator-controlled task boundaries and deterministic retrieval; learn compression decisions only if ordinary tool use works.

[Recursive Language Models](https://arxiv.org/abs/2512.24601) provide another applicable distinction: context can be explored through bounded programmatic calls rather than placed in one prompt. That is a harness-level recursive workflow. It does not introduce recurrent transformer depth or prove that a small model can plan the required calls reliably.

**Proposed first context experiment:** replay fresh, multi-step tasks with an identical model, tools, deadline, and response budget under full history versus reversible rendering. Keep raw history accessible to both policies. Include delayed references to old failures and permissions, restore-required cases, misleading duplicate text, and changed tool outputs. Count final verified completions, restoration failures, repeated actions, tokens, prefill time, and total time. No history deletion or automatic deployment is part of this proposal.

## KV cache and complete recurrent state

The useful modern references are the [vLLM hybrid cache manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/), [SGLang's hierarchical cache design](https://docs.sglang.io/docs/advanced_features/hicache_design), and its [session radix cache](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/session_radix_cache.mdx). These are current design/source references, not installed Strata capabilities or proof that every main-branch facility is in a released package. Their central lesson for Ling is that token-indexed cache and recurrent state require coordinated ownership.

Our Ring K/V and Ling attention states currently live in host vectors. Paging them therefore primarily changes host allocation, sharing, and transfer behavior in this implementation; it does not automatically recover an equivalent amount of GPU VRAM. Generic token eviction is especially unsafe for KDA: a recurrent state is not a list of independently removable token entries.

The first necessary interface is a complete continuation snapshot. Bind model/config/tokenizer, exact prefix tokens and position, executable and arithmetic profile, every attention layer's state, and any research adapter. Loading is transactional: reject partial or mismatched state and leave the current session untouched. Forked branches must not mutate one another.

**Qualification:** uninterrupted decode versus pause/save/reload/continue; empty and long prefixes; branch A then rollback then B; independent sessions; corrupted/missing components; source, model, profile, and position mismatches. Compare subsequent hidden states, ordered routes, logits, and generated IDs using the existing numerical domain. A low-level KDA copy test is not a full-model continuation test.

Ling currently expands MLA into per-head K/V arrays. From the source geometry, these six layers require 120 KiB per cached position in FP32. A compact representation of a 512-dimensional latent plus 64 rotary dimensions across six layers would require 13.5 KiB per position, an 8.89-fold storage ratio before overhead. This is a design calculation, not an implemented reduction: changing where projections and sums occur can change rounding, routing, and outputs. Exact expanded-state snapshots should come first; compressed MLA needs a separate operator and quality qualification.

The same source audit found an accounting gap: Ling's `host_attention_state_bytes` uses the number of positions actually evaluated, while MLA vectors reserve the generation capacity. It therefore understates reserved logical payload when generation ends early. Preserve that historical live-use estimate, add separately named live and allocated-capacity fields, and check their difference on early-EOS controls. The KDA payload is a constant 19.6875 MiB in this implementation. These payload figures exclude allocator overhead and temporary buffers.

Increasing a limit is not enough. Extend qualified context geometrically and test both continuation and actual use of distant constraints. Include retrieval, conflicting updates, multi-hop dependencies, and failure recovery. Do not advertise a checkpoint window as locally usable merely because allocation succeeds.

## Expert residency and transfers

[SeqMoE, 11 September 2026](https://arxiv.org/abs/2609.12978), combines learned multi-step expert prediction, deadline scheduling, future-aware eviction, and graph-compatible execution. Its hardware and residency assumptions differ from ours. [MoE-CORE, 1 October 2026](https://arxiv.org/abs/2610.01950), coordinates prefill staging, layer-dependent residency, routing-history replacement, and prefetch. It evaluates quantized models on an NPU and includes a separate approximate substitution option. We can borrow scheduling ideas while excluding quantization, substituted experts, and transferred headline speedups.

First measure current routing traces. Simulate the existing LRU at one and two GiB, verify that simulated counters reproduce real receipts, and compute a clairvoyant miss lower bound. That tells us whether cache policy has meaningful headroom before training a predictor. Count transferred BF16 bytes, wasted prefetch bytes, active-lease stalls, GPU memory, and wall time—not hit rate alone.

That offline analysis is now complete on three retained traces. The one-GiB simulation reproduces native hits, misses, evictions, and resident bytes exactly. At fixed routes, the counterfactual two-GiB LRU reduces misses from 111,124 to 92,049 on Ring, from 23,638 to 18,610 on the first Ling trace, and from 40,534 to 30,417 on the second. These are roughly 17–25% fewer misses. They are not measured latency improvements. Clairvoyant replacement leaves further theoretical headroom but uses future knowledge and cannot be deployed as that algorithm.

Next compare the existing one and two GiB configurations at fixed token traces and unchanged arithmetic. If capacity helps, prefer that simple setting to a new predictor. If misses remain costly, separate host and GPU residency so a GPU eviction need not repeat the host copy. Bound and account both pools. Prefetch may change placement and timing, never router decisions, selected experts, reduction order, or correctness on a miss.

The source also shows a broader transfer boundary: ordinary matrix calls synchronously upload weights and activations and return results to the CPU. Resident experts and the output head avoid weight uploads but retain the activation/result transfers. Optional Torch operators likewise bridge host vectors and device tensors, and Ring attention transfers its accumulated prefix. A routed Ling layer has six or seven attention matrix calls followed by 28 router/expert/shared-expert matrix calls. This motivates profiling synchronization and launch overhead as well as bytes. SeqMoE's graph execution is not a drop-in change to this host-orchestrated path.

The observed two-case Ling cache smoke preserved tokens and ran faster, but both answers were wrong. Ring timing was interrupted by host contention. Those results do not establish more verified work per hour. Full-file verification also has a real cost; faster hashing retains the same integrity work rather than omitting it.

A persistent admitted worker may ultimately save more time than another small kernel change by retaining model residency across requests. It would need a new explicit lifetime for full checkpoint verification, independent session state, cancellation, and concurrency. Do not quietly remove the current per-run before/after hashing or call a one-shot research executable a serving implementation.

## Recurrence and looping

There are four different mechanisms to keep distinct:

1. Repeating generation and tool verification revises an answer through ordinary tokens.
2. A latent refiner revises a small workspace, then the original decoder consumes it.
3. Repeating transformer blocks adds recurrent depth inside the decoder.
4. Ling's KDA recurrence already carries memory across tokens; it does not automatically provide repeated deliberation at a fixed token position.

**Most promising learned architectural candidate:** a small frozen-backbone latent refiner. [LRT, 1 September 2026](https://arxiv.org/abs/2609.01117), uses an 11.2M-parameter proposer/refiner with a frozen Qwen3-8B decoder. It is trained by task family, and its principal prompted comparisons use non-thinking mode; this limits transfer of its gains. [INTACT, 14 October 2026](https://arxiv.org/abs/2610.19568), trains an external encoder through distillation and correctness reinforcement and tests multi-domain transfer. This supports a bounded auxiliary module as a research candidate, not an established upgrade for Ling or Ring.

Freezing backbone weights does not remove the need for gradients through backbone activations when learning injected latents. INTACT explicitly retains those activations and reports training on an H100 NVL. Our 12 GiB card and inference-only Strata executor do not yet qualify that training path. First implement a disabled-by-default injection boundary with an exact no-op control, then qualify finite differences/backward, checkpointing/offload, one update, exact resume, export/reload, and answer quality. A small parameter count alone is not a memory-feasibility result.

There is also a forward-only research option. [MeZO](https://arxiv.org/abs/2305.17333) established memory-efficient zeroth-order tuning, while [CoRP, 29 May 2026](https://arxiv.org/abs/2605.31494), consolidates rewarded perturbations using a held-out gate. These methods can avoid backbone backpropagation, but trade it for repeated evaluations; neither establishes success on our MoEs. A defensible local starting point would tune a few explicit damping or adapter parameters on development tasks, with untouched BF16 checkpoint files, rather than perturb billions of checkpoint parameters. Cost every rollout and compare against spending that time on ordinary verified repair.

**Closest no-training candidate:** [Training-Free Looped Transformers, 22 May 2026](https://arxiv.org/abs/2605.23872). It reports that naive block repetition usually degrades performance and instead uses damped substeps. Sparse MoE and MLA evidence makes it relevant, but not a Ring/Ling recipe. Repeated passes also repeat our host-device transfers. Ring is the simpler initial block-loop target; Ling needs a deliberate KDA/MLA state policy.

**Trained conversion:** [Retrofitted Recurrence](https://arxiv.org/abs/2511.07384), [Huginn](https://arxiv.org/abs/2502.05171), and [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) establish distinct trained designs. Huginn is pretrained for recurrence; MoR learns token-level depth allocation and cache behavior. They do not show that arbitrary repetition of a frozen block improves our models.

Before any loop treatment, define what is held fixed, what is revised, whether each loop has its own cache, and which state is committed. Advancing KDA history repeatedly for the same token without an explicit rule silently changes the modeled sequence. A safe experimental alternative is a transactional branch from a frozen pre-token snapshot, with one final committed state. That is our proposed semantics, not a validated model architecture.

## Cascade correlation and completion detection

The historical [recurrent cascade-correlation architecture](https://papers.neurips.cc/paper_files/paper/1990/hash/fe73f687e5bc5280214e0486b273a5f9-Abstract.html) grows units during learning and freezes previously learned structure. Adding more arbitrary MoE experts is not equivalent: the existing router, normalization, and residual scales were trained for a particular expert set.

A plausible modern analogue is to add one small residual refiner, train it on verified failure families, freeze it only after independent acceptance, and add another only if residual errors justify it. Compare against an equally sized fixed module, LoRA, and ordinary repair. This is an original research hypothesis; no surveyed result establishes broad Ring/Ling gains from it. Growing capacity must pay for its extra training, inference, and interference risk.

Stable state is not solved state. [Prediction Dynamics in Depth-Recurrent Language Models, 18 September 2026](https://arxiv.org/abs/2609.21383), analyzes completed trajectories retrospectively. It does not provide a prospective, correctness-guaranteeing stopping controller. Contraction can establish numerical settling under its assumptions, not semantic correctness.

Use external tests for completion where available. Treat internal signals—state drift, logit margins, entropy trends, and routing margins—as candidate predictors of whether more reasoning or a repair will help. Require out-of-sample improvement over cheaper signals such as a failed test, output length, and remaining time. J-space state capture is instrumentation; it does not by itself reveal a faithful interpretation of private reasoning or causal problem resolution.

## Hardware compatible kernels and training

The [official FlashAttention repository](https://github.com/Dao-AILab/flash-attention) supports Ada with FlashAttention-2's FP16/BF16 path. FlashAttention-3 targets Hopper; FlashAttention-4 is optimized for Hopper/Blackwell. A higher version number is not a reason to install an unsuitable kernel on this GPU. Switching our FP32 working profile to BF16 attention is an arithmetic change, despite retaining original BF16 weights; it needs a newly declared numerical and answer-quality gate.

[RoutePack, August 2026](https://arxiv.org/abs/2608.12146), includes Ling training experiments, but addresses expert placement and rollout scheduling in a distributed training setting. It is useful once a correct trainer exists; it is not a single-card Strata RL implementation. Rejection sampling and verified repair remain required controls before a costly RL program.

## Execution order and acceptance

| Order | Deliverable | Decisive acceptance |
| --- | --- | --- |
| 1 | Fresh paired Ling thinking baseline; preserve Ring's separate quality result | Exact valid receipts, externally graded answers, every timeout counted; no global budget change |
| 2 | Profile actual work and analyze exact expert traces | Reproduce native counters; separate verification, mapping/copies, compute, attention and tools |
| 3 | Complete continuation snapshot, fork and rollback | Whole-model continuation equality; corrupt/mismatched/partial state rejected transactionally |
| 4 | Reversible context rendering and bounded verifier repair | Fresh matched-time tasks; final tests hidden from all arms; retain raw history and failures |
| 5 | Existing cache capacity and hardware-compatible kernel variants | Unchanged reference gates plus independent answer-quality validation; measure verified work/hour |
| 6 | One small latent-refiner candidate | Real no-op, gradient/resume and reload gates; beats best repair baseline under equal total time and memory |
| 7 | Damped block loop, then optional learned growth | Additional externally correct answers beyond the stronger baseline; reject stable-wrong and stalled loops |

The established promotion objective remains at least 15% more verified work per wall-clock hour with a predeclared held-out quality gate. Count a completion only when execution is valid and every withheld final task test passes. The denominator is total charged wall time for the assigned task set, including setup, loading, verification, tools, failed attempts, timeouts, and queue time; declare how common costs are allocated before starting. Separately declare an acceptable correctness difference and confidence rule before seeing the independent validation results. If the baseline solves nothing, a percentage speedup is undefined: report absolute completions and time instead. The current six-case mode pilot is directional evidence only, not that independent promotion test. Keep original weights, failed attempts, numerical limits, and non-promoted candidates identifiable throughout.

## Current execution record

The six fresh paired Ling tasks are complete under frozen prompts, graders, source and executable hashes, equal 768-token allowances, native/whole-case deadlines, full checkpoint verification, and a GPU process identity check. Both mode arms used Strata at the frozen source, and the service PID remained unchanged.

| Policy | Strict task passes | Valid EOS receipts | Native timeouts | Charged wall time |
| --- | --- | --- | --- | --- |
| Thinking off | 0 of 6 | 6 of 6 | 0 | 297.01 seconds |
| Thinking on | 4 of 6 | 5 of 6 | 1 | 944.32 seconds |

The predeclared directional gate failed: it required all 12 valid EOS completions and at least five thinking-on passes. Thinking on produced one wrong string transformation and timed out on the digit-arrangement task. Two thinking-off failures had substantively correct content in the wrong output format; they remain failures under the frozen interface contract. Thus this is evidence for better usable answers with thinking in a small sample, with unresolved reliability, rather than a general reasoning gain or an architecture improvement.

Observed thinking-on throughput was 15.25 strict completions per charged wall hour on the shared host; the off arm completed zero strictly valid tasks. A percentage improvement over that zero baseline is undefined. All 11 completed provenance records match the previously qualified full checkpoint digests, manifest and executable exactly. The timeout arm has no verified receipt and receives zero credit. No promotion follows.

The frozen RESULT's original timeout counter incorrectly counted only the outer process deadline. An independent audit identifies the one native 350-second timeout from the wrapper's exact error marker. The raw record is preserved. A separate future driver copy now distinguishes native and outer timeouts and passes seven CPU controls; it was not substituted into the completed experiment.

Evidence bindings:

- Frozen plan SHA-256: `53140e8c8bec42a4e10881a19ab32099170844757911f256474a4abff93c67ca`.
- Initial runtime freeze SHA-256: `94bf5dd2619cf3806884acef22d5b11dbfbdf71e97ede23aa664ddcaeddb8464`.
- Original result SHA-256: `8a0e9991311688d15fda39c15e8a5de66ae81c05e10f17bfa5a56a55f27b4d08`.
- Independent final audit SHA-256: `ab3df76add6d8fb47bdd4988232cb9e5f15d143a4b7db16572b439d982fc0dd7`.

The freeze records the GPU processes' PIDs, process start times, resolved executables and executable digests. Those identities were checked throughout the campaign, alongside the service identity. This is stronger than retaining process-name strings across a host restart.

The model implementation, checkpoint payloads, and main services remain unchanged by this scan. Our new BF16 residency and verification tools are published on the research branch. No recurrent model, trained refiner, context controller, or production workflow has been promoted by this review.

## Upstream synchronization and tool delivery

The integration source `22942f42038083cf063c48d8ae13093d94c9e2c7` incorporates upstream `origin/main` at `57293b8ec57203a741dc7b2abd69f24f077037ab`, the existing primary fork branch, and our frozen BF16 research branch. Upstream was checked again immediately before publication. The primary publication target is [ling-ring-bf16](https://github.com/aiii-dot-id/strata-nvfp4/tree/ling-ring-bf16); the frozen pilot branch remains unchanged at `763f5161`. The publication receipt records the final documentation commit separately from the tested implementation source.

The merged implementation passed 227 CPU tests and nine subtests on a clean tree. Four CUDA builds passed their CPU CTests: Ring without/with Torch (6/7), and Ling without/with Torch (2/3). Six selected upstream framework groups also passed with CUDA disabled and isolated temporary state: fork-update mocks, release configuration, GPU fit, no-acquisition controls, launch plans, and local model inventory/import plans. The no-acquisition control now explicitly permits the existing literal loopback UI health check, and a mocked test pins that URL. These checks did not start the browser, replace a serving runtime, install anything, or exercise unrelated quantized-model paths.

The merged Ring/Ling engine and wrapper implementations remain byte-identical to the frozen research versions. Their shared BF16 header gained an unused eight-line `bf16_gemv_pair` declaration. Other upstream shared-kernel and root-build changes are recorded, but were not activated in our standalone executor. The new executables have different hashes, so their qualification was not inferred from the older binaries.

A new bounded GPU replay passed on both rebuilt Torch executables: every layer state, final hidden state, full vocabulary logits, and ordered route bytes at four retained positions matched the prior native outputs bitwise. This covers 76 routed-layer positions on Ring and 92 on Ling. Both wrappers performed full before/after checkpoint hashing with four workers; checkpoint and manifest digests matched the older admitted records. The replay did not rerun the full 1,031/195/358-position comparisons, generation-quality suite, training, or serving integration on the new binaries.

Replay result SHA-256: `a4354c1a716a806485ac0b3c8fd4e77ea9cfb743cc84cf17682aabc5494e301d`. Ring executable: `63c206ea61b07229c3a847df393fe1fba3578d626587a79073958cd83b24985b`. Ling executable: `dd1a638dedae9bfb761792b7265280166305c4695624377c4a9160965d27a8e9`.

The calibrated offline cache analyzer is now part of the fork at `tools/ling_ring_bf16/cache_trace/`. It reproduces native one-GiB cache counters exactly before displaying counterfactual policies, rejects malformed routes and receipts, and passes six controls. Its two-GiB results predict fewer misses, not measured latency. No cache policy, precision, model weights, service, or production defaults were promoted by this delivery. Dawn's service identity and existing GPU process identities stayed unchanged; Sev's Mac runtime was not touched.
