# Scale-ladder held-job false-success incident (2026-08-19)

## Summary

The probe-first scale-ladder launcher reported that its infrastructure jobs
had been released, while Slurm still reported them as `PENDING` with reason
`JobHeldUser`.

The observed campaign was `slad_flat_primary_v4_7937c22`:

- job `250111`, default-partition environment probe;
- job `250112`, Scaglione environment probe;
- job `250113`, activation controller.

The launcher had persisted `released=true` after `scontrol release` returned
success. A subsequent `squeue` observation showed that all three jobs were
still held. An identity-checked manual release then moved job `250112` to
`RUNNING`, job `250111` to a non-held pending state, and job `250113` to its
valid dependency wait.

This was a liveness and state-reporting failure. It did **not** authorize or
corrupt scientific results: the activation and scientific dependency barriers
prevented the 138 scientific and diagnostic tasks from starting prematurely.

## Root cause

`src/launch_scale_ladder.py` treated the process exit status of
`scontrol release` as the release postcondition. On return code zero, the
launcher immediately wrote `released=true` and later reported the activation
as released.

That assumption is invalid for scheduler mutations. A successful command
submission means that Slurm accepted the request; it does not prove that a
fresh, exact observation of the bound job has left `JobHeldUser`.

Two related design gaps made the incident harder to recover from:

1. A restart trusted cached `released=true` fields instead of re-observing the
   scheduler.
2. The activation dependency list could omit probe jobs based on those cached
   Boolean fields.

## Corrected contract

The launcher now applies an observed-postcondition state machine:

1. Resolve the exact job through the approved scheduler tools (including the
   approved-user live query) and verify its job ID, name, partition, and
   comment before any mutation.
2. Treat only live `PENDING/JobHeldUser` as eligible for a release command.
3. After each release request, poll the exact job up to five times at
   one-second intervals. A still-held response is allowed to be a stale read;
   it does not immediately trigger another command.
4. Reissue the idempotent release only when a complete verification window
   observes no non-held state and contains at least one fresh exact held
   observation. A window containing only query failures stops fail-closed.
   At most three command attempts are allowed.
5. Accept success only after observing an exact non-held live state or an
   allowed exact terminal state. The role-specific rules reject invalid probe
   dependency waits, `DependencyNeverSatisfied`, and failed activation
   terminals.
6. Fail closed on identity mismatch, persistent query failure, or exhausted
   held-state verification. A raw return code of zero is never sufficient.
7. Persist the exact verified observation and release-command attempt count
   alongside `released=true`, so the manifest retains the evidence behind the
   claim.
8. On every restart, clear all cached probe and activation release claims in
   one durable manifest write, then revalidate every exact scheduler state.
9. Always retain both exact probe job IDs in the activation controller's
   `afterany` dependency barrier. Terminal dependencies are valid and safer
   than deriving safety from cached release flags.

The bounded worst case is three release requests and fifteen one-second
verification waits per job, plus bounded scheduler-query time.

## Regression coverage

`tests/test_scale_ladder_campaign.py` now covers:

- stale held reads followed by a verified release;
- a full stale-read window before an idempotent reissue;
- return code zero with no observed postcondition;
- nonzero release status followed by a verified scheduler transition;
- transient and persistent scheduler-query failures;
- identity mismatch during post-release verification;
- role-specific pending, active, and terminal state classifications;
- restart-time revalidation of cached release claims;
- prevention of activation release after probe verification failure; and
- preservation of both probe IDs in activation dependencies.

## Rules for future Slurm code

- Name a scheduler mutation and its observation separately: “request
  accepted” is not “state changed.”
- Persist success only after a fresh, identity-bound scheduler observation.
- Make restart decisions from scheduler evidence, not cached Boolean flags.
- Keep dependency barriers independent of optimistic client-side state.
- Test false-success, stale-read, partial-write, timeout, identity-mismatch,
  and restart paths before a launcher is called cluster-ready.
- A launch summary must distinguish infrastructure armed, scientific jobs
  submitted, scientific jobs running, and results validated.

## Scope and follow-up

The audit found the same exit-code assumption in the scale-ladder scientific
gate reconciler. That path was safety-preserving—the scientific arrays still
depended on successful gate completion, and `submitted=true` still required a
certified `COMPLETED/0:0` gate—but it could stall indefinitely after a false
successful release request. The incident patch therefore applies the same
observed-postcondition algorithm to `src/reconcile_scale_ladder_gate.py` and
removes the unused raw-release helper from `src/launch_scale_ladder.py`.

This patch changes future reviewed commits only. It cannot alter a detached
campaign that was already launched from commit `7937c22`; that campaign's
dependency barriers preserve safety, but its gate should be monitored for the
same liveness symptom.

Do not infer that sibling launchers are fixed automatically. The legacy
`scripts/arm_scale_ladder_fresh_probes.sh` path and tariff-response release
paths should be audited against the same observed-postcondition rule before
future use. Prefer the reviewed probe-first scale-ladder launcher until that
audit is complete.

Known non-blocking residuals:

- a terminal non-`COMPLETED` scientific gate safely leaves `submitted=false`,
  but the terminal observation is not yet copied into `campaign.json`; inspect
  the exact Slurm record when diagnosing that state;
- the `scontrol` and `sacct` fallback fingerprints do not independently bind
  the username (the live `squeue` query is approved-user scoped);
- bounded polling deliberately stops fail-closed under unusually long Slurm
  controller lag, so an operator may still need to reconcile after inspection;
  and
- the legacy and tariff-response release paths remain separate audit items.
