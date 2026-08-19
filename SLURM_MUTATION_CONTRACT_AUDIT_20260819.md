# Slurm mutation contract audit (2026-08-19)

## Scope and governing contract

This audit is based on reviewed commit
`1d80402d79d1cbb4b786b780f7287c12b02d3621` and the tracked diff
`7937c22..1d80402`. No live Slurm command was run while preparing it.

The contract is:

1. A command return code reports only whether a request was accepted.
2. Before a mutation, resolve one exact job and bind job ID, approved user,
   job name, partition, immutable execution comment, and campaign role.
3. After a mutation, observe the exact postcondition through bounded queries.
   A live controller record takes precedence over lagging accounting.
4. Bounded stale reads are tolerated. A retry is permitted only after a full
   observation window still proves the same idempotent precondition.
5. Query failure, identity mismatch, missing exit code, or an unobserved
   transition fails closed.
6. Persist evidence before persisting a success label. On restart, re-observe
   the scheduler instead of trusting cached `released`/`submitted` strings.
7. Never remove a reservation unless the intended jobs' disposition is proved.

`src/slurm_state_contract.py` is the shared implementation used by the active
tariff-response and MIP-statistics launchers. Its synthetic fixtures exercise
the same `squeue`, `scontrol -o`, and `sacct -P` parsers used in production.

## Active paths fixed by reviewed commit 1d80402

| Mutation site | Pre-fix behavior | Severity | Required/implemented postcondition | Restart behavior | Disposition |
|---|---|---:|---|---|---|
| `src/launch_scale_ladder.py`: held probe and activation submissions | Parsed IDs were historically treated as enough; cached release flags could survive restart. | Blocker | Exact user/ID/name/partition/comment receipt; release accepted only after exact non-`JobHeldUser` live state or allowed terminal observation. | Clear cached release evidence and re-observe each exact infrastructure job. | Fixed in `1d80402`. |
| `src/launch_scale_ladder.py`: probe and activation `scontrol release` | `rc=0` was recorded as released. | Blocker | Up to three idempotent commands, each followed by five bounded observations; `rc!=0` is acceptable if the exact transition is observed. | Exact comment discovery recovers accepted-before-record infrastructure jobs; ambiguous absence never authorizes replacement. | Fixed in `1d80402`. |
| `src/reconcile_scale_ladder_gate.py`: held scientific gate release | Command status could be mistaken for state change and stall the gate. | High | Exact gate fingerprint plus observed non-held state; `COMPLETED` requires `ExitCode=0:0`. | Reconstruct held gate/arrays from exact comments and controller records; persist submission intents before `sbatch`. | Fixed in `1d80402`; terminal-failure persistence completed here. |
| `src/reconcile_scale_ladder_gate.py`: held gate and missing held arrays | Accepted-before-record windows could otherwise duplicate work. | Blocker | Validate exact parent, user, name, partition, comment, task coverage, dependencies, and held state before accepting or resuming. | Bounded discovery; an unresolved durable intent forbids replacement. | Fixed in `1d80402`. |

## Active paths fixed in this branch

| Mutation site | Pre-fix behavior | Severity | Required postcondition | Restart behavior | Disposition |
|---|---|---:|---|---|---|
| `src/launch_tariff_response_pilot.py`: held gate `sbatch` | The gate had no plan-derived comment and a parsed numeric ID was not checked against the scheduler. | Blocker | Exact held receipt for role `tariff_response_release_gate`, approved user, ID, name `TRG<plan>`, `default_partition`, comment `TRSPG:<plan-prefix>`, state `PENDING`, reason `JobHeldUser`. | A durable intent is discovered by exact comment. Unresolved acceptance remains ambiguous and no replacement is submitted. | Fixed here. |
| `src/launch_tariff_response_pilot.py`: gate release | Raw `scontrol release` status directly produced `gate_state=released`. | Blocker | Verify the exact precondition, request release idempotently, and observe a non-held valid state. Persist `gate_release_verification` before `released_verified`. | Every invocation re-observes. Cached `released` is not evidence. | Fixed here. |
| `src/reconcile_tariff_response_gate.py` | Queried only `JobIDRaw,State`, trusted a narrow cached-state allowlist, and accepted `COMPLETED` without `ExitCode`. | Blocker | Bind exact user/ID/name/partition/comment/role. Terminal success is only `COMPLETED/0:0`; terminal failure records source, state, exit code, and `submitted=false`. | Handles exact new states and accepted-before-record gate intents. Legacy manifests are preserved as `legacy_unverified`, never resubmitted. | Fixed here. |
| `src/assemble_tariff_response_campaign.py` and `src/validate_tariff_response_archive.py` | Accepted the strings `released` or `released_reconciled`. | Blocker | Require exact persisted release evidence and exact terminal `COMPLETED/0:0` evidence, both matching the approved plan-derived gate specification. | Old artifacts remain readable JSON but are explicitly rejected as unverified scientific evidence. | Fixed here. |
| `src/launch_mip_statistics_campaign.py`: individual held `sbatch` receipts | A numeric string was accepted as the job receipt. | High | Resolve every exact user/ID/name/partition/execution-comment tuple and prove `PENDING/JobHeldUser` before it joins the release set. | Durable per-cell intents are recovered by exact execution comment. Ambiguous absence forbids replacement. | Fixed here. |
| `src/launch_mip_statistics_campaign.py`: grouped release | One comma-separated `scontrol release` return code marked every job released and was described as atomic. | Blocker | Pre-resolve every intended job; release and persist verification separately. Commands are issued only for exact jobs still observed `PENDING/JobHeldUser`. `submitted=true` only after all release verifications exist. | A restart discards cached release claims, re-observes all IDs, and retries only exact held jobs; already-released jobs require no command. | Fixed here. |
| `src/launch_mip_statistics_campaign.py`: cleanup cancellation | `scancel rc=0` deleted all execution reservations without proving cancellation. | Blocker | Pre-resolve all exact held jobs; cancel each separately; accept only exact terminal `CANCELLED` with an exit code. | Partial/ambiguous/live/mismatched cancellation is durable and all reservations remain. | Fixed here. |
| `src/launch_mip_statistics_campaign.py`: direct four-task array | A parsed parent ID immediately set every task and the campaign to submitted. | Blocker | Validate exact parent and complete task coverage, including split controller records, user, name, partition, comment, state, and terminal exit code where applicable. | Same-campaign restart rediscovers the exact parent by execution comment and revalidates controller/accounting evidence. No exact discovery means no replacement. | Fixed here. |
| `src/summarize_mip_statistics.py` | Trusted `submitted=true`, `submitted_array`, or legacy release strings. | High | Require per-job release evidence or complete exact array-receipt evidence before reading scientific outputs. | Legacy evidence is labeled unverified and rejected rather than silently upgraded. | Fixed here. |
| `src/reconcile_scale_ladder_gate.py`: terminal scientific gate | Raised on terminal non-success without copying the observation to `campaign.json`. | Medium | Before raising, persist `gate_state=terminal_failed`, exact observation/source/state/exit code, and `submitted=false`. Missing/nonzero exit codes are retained as failure evidence. | Repeated reconciliation reads the durable terminal failure instead of a stale optimistic state. | Fixed here. |
| Tariff/MIP same-campaign mutation serialization | Concurrent submission/reconciliation processes could both advance one campaign manifest. | Blocker | An OS-backed lock outside the not-yet-created campaign root covers initialization, recovery, every manifest transition, scheduler mutation, cancellation, and release. | A waiting process re-enters through exact recovery after the first process publishes its state; it cannot race a second `sbatch`. | Fixed after adversarial review. |
| MIP execution reservations and deduplication | Sequential `O_EXCL` files could be stranded before their set was published, and recovery could submit planned jobs without repeating a failed execution-comment query. | Blocker | Persist the expected one-to-one cell/digest/path/hash transaction first; publish each fully fsynced reservation by atomic no-replace link; adopt only byte-identical same-plan files; and persist/re-run successful `squeue` plus `sacct` comment deduplication before every planned recovery submission. | A crash-created exact subset is idempotently adopted and completed. Unknown scheduler state, conflicting bytes, symlinks, duplicate digests, or wrong paths fail closed before `sbatch`. | Fixed after adversarial review. |

## Sibling-path inventory and disposition

| Site | Mutation(s) | Current contract/disposition | Risk before reuse |
|---|---|---|---:|
| `scripts/arm_scale_ladder_fresh_probes.sh` | Historical probe `sbatch`, gate dependency `scontrol update`, and gate release. | Formally retired here. The executable now exits 64 before any query or mutation and points to `reconcile_scale_ladder_gate.py`. | Blocker if the historical body were restored. |
| `scripts/replace_stalled_scale_ladder_20260819.sh` | Seven exact `scancel` requests. | Historical one-campaign closeout. Independently verifies fixed IDs/fingerprints, publishes a pre-cancel archive, ignores request status as proof, polls boundedly, requires exact zero-runtime `CANCELLED` accounting, and publishes a hashed receipt before replacement. Retired from general use by its embedded campaign/plan/IDs. | Low for its one frozen incident; not a reusable API. |
| `scripts/closeout_scale_ladder_a27eed66.sh` | Grouped `scancel` of seven embedded IDs. | Historical and formally non-reusable: fixed campaign/plan/job IDs and archived closeout only. It checks exact pre-state, bounded post-cancel queue absence, and accounting/no-runtime evidence before allowing its paired replacement. | Medium if copied: grouped cancellation is not independently persisted per job. |
| `src/slurm_campaign.py` | Held array `sbatch`, per-task `scontrol update`, parent `scontrol release`. | Deferred. It is not used by the scale-ladder, tariff-response, or MIP-statistics next campaigns. It still treats command status as rename/release success and lacks exact receipt/restart recovery. It must be fixed before reuse. | Blocker. |
| `src/submit_exact_dive.sub`, `src/submit_exact_peaks.sub` | Best-effort per-task `scontrol update JobName`. | Deferred with `src/slurm_campaign.py`. Worker-side rename is cosmetic and warns on failure, but no observed postcondition is retained. Do not reactivate these campaigns until the parent launcher is hardened. | Low for science, high for provenance/operations. |
| `src/cluster_campaign.py` | Direct MIP `sbatch`. | Deferred as the next standalone MIP hardening task, as requested. It writes an intent but marks submitted after parsing text; it has no immutable execution comment, exact scheduler receipt, or accepted-before-record discovery. | High duplicate-job/provenance risk. |
| `src/reconcile_scale_ladder_probe_artifacts.py` | `scontrol show` only. | Query-only; no scheduler mutation. It verifies probe/gate identity and artifact publication. | None in this mutation audit. |
| `src/submit_scale_ladder.sub` | Resolves an approved `scontrol` binary but does not mutate scheduler state. | Worker-side query/configuration support only. | None in this mutation audit. |

No other `scontrol release`, `scontrol update`, `scancel`, or held `sbatch`
site exists under tracked `src/` or `scripts/` at this commit.

## State semantics exercised by synthetic fixtures

The tests distinguish:

- `PENDING/JobHeldUser`: exact idempotent mutation precondition;
- `PENDING/Dependency`: valid only for a role whose approved dependency is
  expected;
- `PENDING/DependencyNeverSatisfied`: always invalid;
- other non-held pending reasons such as `Resources`: a release transition;
- `CONFIGURING`, `RUNNING`, `COMPLETING`: released/live;
- terminal success: `COMPLETED` and exactly `0:0` where role success matters;
- terminal non-success or missing exit code: durable failure, never success;
- scheduler-query failure: unknown, not absent and not cancelled.

The adversarial transcripts cover stale held reads, bounded retries, nonzero
mutation return with an observed transition, zero return with no transition,
identity mismatch before and after mutation, split array records, partial
release, partial cancellation, restart after partial persistence, and
ambiguous acceptance with no duplicate submission.

## Residual risks

1. Slurm controller/accounting visibility can exceed the bounded windows.
   The chosen response is an operator-visible, durable fail-closed state.
2. Legacy tariff and MIP manifests do not contain enough immutable scheduler
   identity to be upgraded automatically. They remain readable but unverified.
3. `src/slurm_campaign.py` and `src/cluster_campaign.py` remain unsafe for new
   submission until their standalone audits are complete.
4. The one-off closeout scripts are evidence-preserving historical tools, not
   reusable lifecycle libraries.
5. Scheduler evidence proves lifecycle facts only. Scientific completion still
   requires worker completion records, artifact hashes, and domain validators.
