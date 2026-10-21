# Bounded research updates

AI Identity Incorported <james@aiii.id>

This is an opt-in research candidate. It has CPU validation; CUDA execution and capability improvement remain unqualified. It is separate from the running Ling/Ring campaign.

## Why it exists

Ling's third update in the 2026-10-21 capability pilot passed native/trainer parity but produced repeated digits and failed to terminate. On one identical retained response prefix, the end-token probability fell from 0.9999898 after update one to 0.0736274 after update three. This is a local regression witness, not a general quality statistic. The evidence is in `/work/AI/experiments/strata-capability-20261021-r2/diagnosis/ROOT-MATCHED-PREFIX.json`.

Numerical parity establishes that the trainer and runtime implement the same adapter. It does not establish that the update improves the policy. This option adds a separate update-admission check.

## Behavior

Without `--bounded-update-policy`, the runner calls its existing update function and retains the original learning rates and update receipt fields.

With the option, supply a JSON file containing exactly `learning_rates` (with positive finite `rlvr` and `self_opd` values), `max_mean_kl`, `max_max_kl`, and `max_selected_log_ratio` (finite nonnegative values). All settings are explicit; there are no recommended or implicitly selected bounds. Duplicate JSON keys, unknown fields, and nonfinite values are rejected.

The wrapper performs the existing RLOO or full-vocabulary Self-OPD update once. It then measures KL(old policy || proposed policy), both mean and maximum across the sampled action states, and the maximum absolute change in selected-action log probability. It uses FP64 reductions at the declared sampling temperature. Feature and vocabulary transfers are chunked, and the check runs without gradients. It does not substitute supervised learning, normalize trajectory lengths, alter rewards, or reuse an old batch for another accepted update.

A rejected update restores trainable parameters, their gradients, optimizer state and groups, and CPU or the actual model device's CUDA random state. It does not increment the accepted-step count or export an adapter. A JSON receipt records the rejection. Frozen buffers are hash-checked, not cloned; mutation of those buffers aborts the experiment and is not reported as a successful rollback. This path supports one CPU or CUDA device in code; only CPU behavior is currently tested.

Accepted updates still require the existing native parity, exact resumed-step comparison, export/reload, teacher-selection, and fresh evaluation gates. Numerical admission alone cannot promote a teacher.

## Limits and validation

The distribution bound applies only to the observed action states. It is not a guarantee about unseen prefixes or task correctness. The diagnostic check preserves the post-update random state, so observation cannot advance subsequent sampling randomness.

The historical calibration importer deliberately rejects the new source closure. Bounded runs require fresh calibration; this revision does not weaken that import guard. Keep training, teacher selection, held-out evaluation, and retention cases separate. Select bounds before sampling the next campaign and retain failed trials.

CPU controls cover update dispatch, strict configuration parsing, actual objective calls, exact rejection restoration with initialized Adam state, frozen-buffer mutation, nonfinite state, bounded metric batches, random-state isolation, and positive and negative historical calibration controls. CUDA acceptance/rejection, real-model resumed updates, and fresh answer quality must still be qualified before this feature can be described as ready.
