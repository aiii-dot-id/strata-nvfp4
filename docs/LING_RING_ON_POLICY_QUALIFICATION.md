# Strata on-policy qualification operator guide

Author: AI Identity Incorported <james@aiii.id>

This guide describes the native Strata self-OPD and verifier-reward policy-gradient research qualification flow. The exact training implementation qualified on dev8 is commit 4ac6b3a14964b19529e77802ec14f7745004472f.

Across the linked qualified runs, each model completed two real verifier-reward policy-gradient updates and two real Self-OPD updates, with exact training-state restoration and adapter export/reload. Ling q3 imports the qualified two-update RLVR endpoint from Ling q2 and trains a fresh Self-OPD student. The qualified scope is one final-layer expert, not the whole model. These are usable research training mechanisms; capability improvement has not been established and no candidate was promoted.

## What the run measures

RLVR trains one terminal expert adapter with sequence-level verifier reward. Each fresh rollout is sampled from the current native policy. Rewards are grouped by original prompt, and the leave-one-out advantage is multiplied by the sequence-summed sampled log probability. Invalid or incomplete answers receive the task grader's zero reward and remain in the batch. The update uses the current policy logits and the recorded native sampling log probabilities.

Self-OPD uses a fresh student initialized from the same base model and the same six trainable terminal-expert tensors. It first requires a frozen, qualified RLVR endpoint. The round-zero student samples with a zero adapter. After the runner proves zero-adapter native logits and sampling equality, it reuses those round-zero records for both RLVR and Self-OPD; they are therefore on-policy records from the initial policy, not independent teacher/student draws. Later rounds sample fresh trajectories from the updated student. For each sampled trajectory, the teacher is run with the identical original rendered prompt and the sampled prefix, with the frozen RLVR adapter active. The teacher receives no reference answer or other prompt augmentation. The objective is full-vocabulary reverse KL at the sampled response states, using FP64 objective computation while model and native logits remain FP32. This is paper-inspired fixed-prefix semi-gradient training: sampling is outside the gradient and no gradient flows through sampled tokens.

The trainable scope is one final-layer routed expert: gate, up, and down low-rank adapter matrices. Rank is 8 and alpha is 16. Base weights, buffers, and all other experts remain frozen. The native forward path reads the original BF16 checkpoint and uses FP32 computation; qualification is configured with the recorded fp32-cublas, Torch CUDA RMS/pointwise, and model-specific native profile options.

## Exact objective and gradient boundary

For a batch of N sampled trajectories, RLVR minimizes
-(1/N) * sum_i A_i * sum_t log p_theta(y_it | x_i, y_i,<t),
where A_i is the verifier reward minus the mean reward of the other trajectories with the same original prompt. There is one optimizer step per newly collected batch: no corrected-action labels, cross-entropy target-answer substitution, PPO clipping, or repeated policy optimization epochs. The saved-batch replay after restoring a snapshot is a restoration test.

Self-OPD minimizes the mean of KL(p_student(. | s) || p_frozen_RLVR_teacher(. | s)) over sampled response-prefix states. Both distributions cover the full model vocabulary. Teacher logits and sampled prefixes are detached; the trainable student logits receive the gradient.

Captured terminal inputs support this restricted gradient because the adapter is in the final layer's expert feed-forward block. For a fixed sampled token prefix, the attention memory and router inputs are upstream of this final expert adapter. They do not depend on its parameters. Native parity is checked at each update, and later rollout tokens are generated through the current adapter. This qualification does not establish that the same cached-feature gradient would be valid for adapters in earlier layers.

## Training runner configuration and limits

The runner configuration is recorded in PLAN.json and currently specifies:

- two update rounds and four samples per training prompt;
- temperature 1.0 and 32 maximum new tokens;
- learning rate 0.001;
- initialization seed 61021, rollout seed 41021, evaluation seed 41022;
- up to three RLVR batches per round when a verifier-reward group is initially uninformative;
- 300-second native-call timeout and 14,400-second qualification-job limit;
- four training, eight held-out, and four retention tasks;
- bounded native capacity of 1,024 total positions for Ling and 4,096 for Ring; these are executor limits, not checkpoint training windows;
- one shared GPU lock at /work/AI/experiments/strata-dev8-gpu.lock;
- at least 10 GiB free disk required by the native wrapper.

The native wrapper requires paired positive-finite temperature and uint64 seed arguments. Every stochastic receipt records the categorical full-vocabulary method, temperature, seed, sampled log probabilities, and uniform draws. It audits those values against the native full-vocabulary logits before training uses the rollout.

## Normal command

From the repository root on dev8, a fresh run is started with a new output directory:

~~~sh
/work/AI/experiments/ling-strata-20261016/reference-env/bin/python \
  tools/ling_ring_bf16/training/on_policy_runner.py \
  --model ling \
  --suite /work/AI/experiments/strata-on-policy-20261021-r1/SUITE.json \
  --output /work/AI/experiments/strata-on-policy-20261021-r1/ling-new-run
~~~

The recorded Ling q3 run additionally used --qualified-rlvr with the qualified Ling q2 directory; this is an optional admitted import, not required for a new campaign. A generic Ring command uses --model ring, the same suite, and a fresh distinct output directory without --qualified-rlvr.

The process takes the non-blocking exclusive GPU lock, records checkpoint bindings before execution, checks foreign GPU process identity, and records bindings and foreign-process status on return. A changed checkpoint or foreign process causes the lifecycle gate to fail. No service action, Mac change, Sev action, deployment, or commit is part of this flow.

## Evidence and restore checks

Each native stage writes an execution receipt, generation receipt, payload hashes, and optional terminal capture. Selected response rows are copied into retained payloads before transient float files are removed. RUN.json, ROLLOUT.json, and RETENTION.json bind the data used by training.

Each objective saves a step-one snapshot containing adapter parameters, Adam state, CPU/CUDA RNG state, optimizer groups, source and executable bindings, parent checkpoint digest, adapter hash, and the round-zero batch digest. It restores that snapshot, replays the second batch, and requires exact parameter, optimizer, RNG, loss, gradient, and metric equality with the uninterrupted path. The uninterrupted and resumed exports must be byte-identical. After parameters are overwritten, reloading the export must restore them exactly, and native forward logits must pass parity on a training trajectory with maximal target-expert exposure.

When a prior RLVR subgate is admitted after a purposeful interruption, the import path independently checks the q2 plan pins, binding-before/after records, interruption receipt, foreign GPU preservation, qualified RLVR record, adapter sidecars and payload hashes, snapshot payload and adapter binding, and the allowed source-change set. The imported endpoint is copied into the new run and then the student is reset to its fresh zero-adapter parameters before Self-OPD begins.

## Evaluation and admission

The full suite is loaded and split before training, while held-out and retention model evaluation is deferred until both candidate adapters are frozen. Baseline, RLVR, and Self-OPD candidates use rotated evaluation order, paired evaluation seeds, and the same answer-token allowance. Results include correctness, incomplete responses, wall time, and verified per-inference-hour values. The run records mechanics qualification and a small paired qualification result; it does not promote a model or claim broad capability improvement.

Admission requires exact native default-control parity against the prior validated native output, zero-adapter sampling parity, valid full-vocabulary sampling audits, finite adapter gradients with a nonzero aggregate norm, successful native export/reload parity, exact snapshot restore, unchanged frozen base tensors, exact rollback to the original native path, unchanged checkpoint bindings, and preservation of foreign GPU processes. A completed run must also retain the plan, return, qualification, adapter, snapshot, rollout, alignment, retention, and final result records.

## Limitations

This is a mechanics qualification on a small fixed task suite. It is not full-model training, a broad capability benchmark, or a matched training-cost comparison when RLVR requires extra rollout groups. The adaptive collection rule stops once an informative reward group is observed, so the resulting conditional sampling procedure is not claimed to be a fixed-budget unbiased estimator. Self-OPD uses a frozen qualified RLVR endpoint as its teacher while the student uses its own sampled prefixes; this distinction must be reported with any result. The objective is paper-inspired and is not claimed to reproduce an exact paper recipe. The current native engines reject stochastic sampling with continuation state and reject adapters with continuation state. Rollouts therefore start from complete prompts with fresh generation and cache state.

Native rollout wall times include shared-host process startup, model loading, cache setup, payload validation, and output writing; they are instrumented qualification measurements, not production serving benchmarks. For any new campaign, use a fresh output directory, freeze the suite and plan before rollout, and record current outcomes separately from this scope guide. Reusing the existing suite reproduces this qualification; independent capability validation needs fresh held-out tasks. Qualification outcomes belong to their named result and return receipts. No research adapter was promoted.


## Recorded results and exact scope

The completed Ling q3 and Ring q1 runs each qualified two actual RLVR updates and two actual Self-OPD updates. Ling q3 imports only the already-qualified RLVR endpoint from Ling q2, after validating that interrupted run's admitted source changes and preserved checkpoint. Self-OPD resets to a fresh initial student; it is not additional RLVR training and does not use corrected-answer supervised labels.

| Model | Updated expert | Trainable parameters | RLVR | Self-OPD | Snapshot and export |
|---|---:|---:|---|---|---|
| Ling-3.0-tiny | final-layer expert 47 | 49,152 | 2 qualified updates | 2 qualified updates | exact restoration and resumed export |
| Ring-mini-2.0 | final-layer expert 194 | 61,440 | 2 qualified updates | 2 qualified updates | exact restoration and resumed export |

The original checkpoint hashes were unchanged. Native default rollback was exact on 27 Ling files and 23 Ring files. Native adapter forward checks had maximum absolute logit error at most 3.814697265625e-6 and identical argmax. This tolerance-based forward comparison is distinct from the bitwise restoration/export checks.

Ling's original 32-token paired evaluation produced 2/8 correct held-out answers and 0/4 retention answers in each arm. All 12 generated token sequences were identical across baseline and both adapters; no answer-quality gain was observed. One answer in each split was incomplete.

Ring's original 32-token evaluation was inconclusive: every one of the 36 baseline/candidate trajectories reached the token limit. Its zero scores cannot establish quality equality or safety. A separate follow-up records any repaired inference protocol and new task results; the original failure remains part of the evidence.

## Why the numerical repair matters

An early self-OPD diagnostic exposed cancellation when computing very small reverse-KL differences in FP32. On the retained 4 by 157,184 logit fixture, the FP32 calculation produced a negative KL sum of -5.5186035297e-8; the FP64 objective gave positive 1.61173814015e-10 and analytic-gradient agreement within 7.8737675559e-15. The repair changes objective arithmetic, not model weights, native activation precision, or checkpoint format. The interrupted run was not relabeled as an overall pass.

## Evidence locations

The complete campaign remains on dev8 under /work/AI/experiments/strata-on-policy-20261021-r1. Key entries are CPU-ADMISSION-R3.json, ling-q3/, ring-q1/, the sibling RETURN receipts, REVIEW-METHODS-FINAL.json, and the final campaign summary. Full tensor evidence remains there; compact mirrored receipts are sufficient for reviewing outcomes but do not replace the retained training batches for exact replay.

The CPU source admission passed 274 tests. The unchanged native component suites passed 3 Ling and 9 Ring tests. Separate Ring inference-controller tests, where used, are reported independently from training qualification.


## Reconcile prior work before adding experiments

ADOPTION-AND-GAPS.json is the current evidence inventory. It separates the older inference-only Strata executor from the attached terminal adapter trainer qualified here. Teacher-prefix scoring, current-policy sampled-log-probability checks, and exact training-state restoration are no longer wholly unknown on dev8; their qualified scope remains one terminal expert. Historical Mac measurements are reported separately and are not a CPU/CUDA numerical equivalence claim.

The relevant RLVR admission criterion is informative verifier reward within sampled prompt groups and finite gradients with a nonzero aggregate norm. A zero score in an older greedy or truncated evaluation is a warning about task/protocol choice, not proof that every new sampled group has zero variance. The runner retains uniform-reward groups, draws additional current-policy records under its declared cap, and stops without manufacturing an update if the gate cannot be met.

The qualified runs call the native executors directly. They do not use the Strata serving prompt-rewriting path. Prompt identity is still pinned: the original Ring training protocol closes the reasoning block in its rendered prefix, whereas the separate quality follow-up allows a bounded reasoning stage and records a uniform final-answer instruction. Those are distinct protocols and must not be reported as a before/after capability improvement.

## Separate Ring inference follow-up

The training-only controller probe used 128 reasoning tokens and up to 96 final-answer tokens with greedy decoding. If the reasoning allowance ended without a closing delimiter, the controller appended an explicit closing delimiter and replayed the full prefix with the same adapter. This is budget management, not natural completion detection. The probe completed both tasks in the requested format but answered only one correctly.

The subsequent fresh comparison freezes that same controller for baseline, RLVR and Self-OPD. Its 8 held-out and 4 retention tasks were created after the candidate adapters were frozen, with no further training. Complete EOS, strict answer format and correctness are reported separately. The controller has 29 passing CPU checks.

Adapters with native continuation state are not yet admitted. The controller therefore uses full-prefix replay, which repeats prefill work. Its measured wall time includes that cost, process startup, validation and output writing; it is not a persistent-serving speed measurement.


## What the retained reward groups actually taught

REWARD-SIGNAL-DIAGNOSTIC.json inspects the saved training batches after qualification; it did not change the running evaluation. Every arithmetic reward group in both models and both RLVR rounds was uniformly zero. The leave-one-out advantages for those arithmetic trajectories were therefore zero. The nonzero direct RLVR advantages came only from an extraction task: tr-extract-01 for Ling and tr-extract-02 for Ring.

This is a real policy-gradient update, but it is not evidence that the RLVR stage learned arithmetic. Self-OPD subsequently scored student arithmetic prefixes too; that does not establish arithmetic improvement. A nonzero global gradient is insufficient as an admission rule for a capability campaign. Future training-only calibration should establish informative reward coverage for every target task family and verify actual adapter exposure and update sensitivity before freezing a new capability trial. It must not tune against this campaign's held-out cases.

The 2-update single-expert runs establish a usable, restorable training mechanism. They do not establish that this narrow adapter has sufficient capacity, receives enough useful gradient, or that the qualified RLVR teacher is better at the intended downstream tasks. Increasing steps or adapter scope is a new experiment, with new gradient validation if the scope moves upstream.


## Next capability admission

The next campaign should first qualify a training completion policy on development data only. Ring's 32-token training allowance produced many incomplete answers, and every arithmetic reward group had zero direct RLVR advantage. More updates on those same uniform-reward groups would not supply the missing policy-gradient signal.

Before a larger run:
1. Freeze the generation contract and a bounded total compute budget. If a controller inserts a reasoning delimiter, retain its tokens separately from sampled actions, and requalify teacher-prefix alignment and policy log probabilities for that exact contract.
2. On training/development tasks, require both correct and incorrect completed samples within each target family under the declared sampling allowance. Report format failures, truncations, reward variance, target-expert exposure, aggregate gradient norm and actual update sensitivity separately. Do not discard failures to manufacture a positive reward rate.
3. Choose the adapter scope and update count using development evidence. Any move to earlier layers needs a new backward path; the terminal cached-feature proof does not cover it.
4. Freeze candidates and compare original, direct RLVR and Self-OPD on fresh, disjoint held-out tasks with matched inference budgets. Include the RLVR teacher's quality and total training cost; a teacher that changes logits without improving the target tasks has not established a useful capability to distill.

The present experiment is an implemented training foundation. Its small fixed evaluations do not authorize a capability-improvement or production-promotion claim.


## Efficiency candidate after the capability protocol is repaired

The current bottleneck is native rollouts and frozen-teacher scoring, not the small optimizer update. Ring q1 recorded 584.10 seconds of additional RLVR training rollouts, 290.71 seconds of fresh Self-OPD rollouts, and 547.11 seconds of teacher scoring. The two RLVR optimizer updates took 12.49 seconds and the two Self-OPD updates took 8.35 seconds. Shared initial rollouts cost 296.44 seconds once; evaluation costs are separate. These are instrumented qualification timings, with no startup/prefill/I/O decomposition.

TEACHER-CAPTURE-REUSE-ASSESSMENT.json identifies a narrow candidate: reconstruct the frozen teacher's terminal output from the student's existing capture for the same sampled prefix and same expert. The engine records pre-adapter terminal values, so the necessary upstream data may already exist. This remains unimplemented and unqualified. Full-vocabulary agreement, the actual tiny KL objective, route/capture identity, and present/absent-expert cases must pass before replacing authoritative native teacher scoring. No speedup is claimed. The same argument does not cover earlier-layer adaptation.


## Completed fresh Ring comparison

The frozen 36-row comparison and closing checks completed successfully. Independent CPU auditing reproduced all scores, exact sampled token sequences, prompt/adapter bindings and summary timings. There were no missing or extra rows. Both trained arms reproduced the baseline sampled token sequence on all 12 tasks.

| Arm | Held-out correct | Retention correct | Incomplete | Invalid-format held-out |
|---|---:|---:|---:|---:|
| Original | 5/8 | 2/4 | 0/12 | 1/8 |
| RLVR | 5/8 | 2/4 | 0/12 | 1/8 |
| Self-OPD | 5/8 | 2/4 | 0/12 | 1/8 |

No capability gain or efficiency advantage is established. Total instrumented wall time was 3,566.40 seconds. The held-out arms each took approximately 799–802 seconds; retention took 384–387 seconds. These small timing differences on a shared host are not a speedup result.

Strict grading caveats are retained. All arms answered uppercase R for the third letter of harbor, while the frozen exact-string grader expects lowercase r. The prompt did not explicitly require lowercase; this should not be described as a demonstrated reasoning error. A separate retention case returned 7 as the last digit of 9076, a genuine wrong value. One held-out answer gave a correct arithmetic expression but violated the explicitly required bare-number format. Primary scores were not changed after inspection.

The quality process exited, its dedicated 32.5 GB read-only memory holder was released, and the GPU lease was available at the final check. The original checkpoint was unchanged, existing GPU processes were preserved, and no Mac model/service change or research promotion occurred. FINAL-RESTORATION.json records the check.

The evidence audit itself accepted a healthy copied fixture and rejected eight deliberate corruptions. That supports the result-accounting checks; it is not an additional model capability score.
