# Independent audit of commit f85798c ("Harden exact CG preemption recovery")

Date: 2026-08-11. Auditor: independent review requested before launching the
task-22/24/32 recovery on Unicorn. Scope: correctness, race conditions,
provenance gaps, preemption behavior, Slurm requeue behavior, and unnecessary
complexity of `f85798cbf172ddb463f083636abcf139d066d1df` on `peel-and-price`.

Files audited: `src/migrate_legacy_exact_pool.py`, `src/durable_io.py`,
`src/exact_pricer_expanded.py`, `src/launch_legacy_bigtariff_recovery.sh`,
`src/submit_legacy_bigtariff_recovery.sub`, `src/run_cg_snapshot_control.py`,
`src/campaign_artifact_status.py`, the three recovery test modules, and the
`f4e31c3` versions of the pricer, config, and campaign submit file.

## 1. Confirmed correctness

- **Model semantics are unchanged.** `git diff f4e31c3 f85798c` touches no
  routing, charging, tariff, battery, or objective code:
  `audit_giro_known_columns.py`, `utils_v2.py`, `config.py`, and
  `pricing_dp_og.py` are byte-identical to the legacy commit. Only
  `master_lp_scipy.py` changed (LP result hygiene + the deliberate,
  pre-existing 1e-7 -> 1e-6 tolerance from 2cc40d9/47a9a49). HiGHS remains
  the CG master; nothing switches to Gurobi.
- **The legacy failure mechanism is what the runbook claims.** The `f4e31c3`
  resume loop does `json.loads(line)` per journal line with no repair, and it
  crashes *before* opening the journal in append mode, so the damaged
  journals were never modified after the crash.
- **The witness design is sound against the real legacy artifacts.** The
  `f4e31c3` *terminal* result writer stores `provenance.instance_sha256` /
  `prices_sha256` plus every model field the migration checks
  (`MODEL_FIELDS`), while its *periodic* partials store the model fields but
  no provenance — exactly the gap the split-status witness reconstruction is
  built for. Legacy snapshots cannot self-witness (no
  `snapshot_mark_minutes`, no provenance), and the damaged statuses
  themselves are excluded (`stop_reason="running"` is not a stable witness).
- **Fail-closed behavior verified end-to-end** (real CLI, no mocks): missing
  witness exits 2 with the `no authenticated .* witness` message the `.sub`
  classifies as WAIT; conflicting witnesses exit 2 with a message that does
  *not* match the WAIT pattern (fatal, as intended); interior journal
  corruption refuses without touching any source byte; sources are hashed
  before and after both preview and copy, so concurrent modification of the
  legacy artifacts aborts the migration.
- **Copy-only isolation verified.** After `--apply`, the legacy status,
  journal, iters, and log bytes are unchanged; the raw archive under
  `*.legacy_raw/` is byte-identical to the sources on a distinct inode; the
  repaired tail bytes are quarantined (`*_changed_tail.bin`); repairs happen
  only on staged working copies.
- **Idempotency and preemption-mid-append verified.** Re-running `--apply`
  after (a) success, (b) the continuation extending the journal, and (c) a
  simulated preemption that truncated the final appended record all validate
  the migrated prefix hashes and return "already prepared" without writing.
  The pricer's narrow tail repair truncates exactly back to the last complete
  record, which can never reach inside the migrated prefix (the prefix ends
  on a complete, newline-terminated record by construction).
- **Publication ordering is crash-safe.** The migration writes raw archive ->
  journal -> iters -> attestation -> status (commit marker) with fsynced
  atomic renames; the pricer publishes identity (`initializing` /
  `resume_starting`) before opening append handles; journal appends are
  fsynced before the status that counts them; immutable snapshots publish the
  journal copy before the discoverable status JSON, and the orphan-recovery
  path re-publishes an interrupted snapshot status only from a prior status
  with the matching `snapshot_mN` stop reason, column count, and positive
  routes.
- **Locking verified across processes.** `exclusive_output_lock` refuses a
  concurrent second owner (with owner diagnostics in the message), and a
  requeue acquires the lock after the previous owner is SIGKILLed. A live
  kill-and-resume run on the committed TwoDuty instance resumed the pool,
  kept `iters.csv` elapsed monotone across attempts, refused a rerun without
  `--resume`, and refused a concurrent duplicate.
- **Resume identity checks fire before any artifact is modified**, including
  the git-commit discipline: a legacy status resumed directly (bypassing
  migration) is refused for missing provenance hashes and/or commit mismatch
  before any repair side effect; an attested legacy migration additionally
  hard-requires commit identity on both sides.
- **Requeue behavior:** `--requeue`, `--open-mode=append`, explicit
  `-t 7-00:00:00`, and re-verification of both checkout commits on every
  (re)start; a requeued allocation re-runs the idempotent migration, skips
  terminal results via `campaign_artifact_status --require-terminal`, and
  resumes the pricer. Exit-0 WAIT (no witness) and SKIP states cannot loop.
- **Original array mapping is correct.** Tasks 22/24/32 map to
  k30_r2/peak18, k30_r4/peak18, k30_r2/sek under the original
  `submit_cg_bigtariffs.sub` arithmetic, and the recovery `.sub` reproduces
  the legacy campaign's identity-checked parameters (soc-step 15,
  block-min 10, partition master, 300/300/0 defaults).
- Full test suite at f85798c: **204 passed, 23 subtests passed** (Python
  3.12.3, pinned requirements).

## 2. Bugs fixed on this branch (`cursor/recovery-audit`)

1. **Wall-limit expiry mid-master was mislabeled `master_failed`**
   (`src/exact_pricer_expanded.py`). If the cumulative wall budget ran out
   *between* master method attempts (the in-loop `_remaining_wall_s(30) <= 0`
   break), `lp` stayed `None` and the run recorded
   `stop_reason="master_failed"` even when the wall — not the solver — ended
   the run (in the worst case without attempting any method that iteration).
   That violates the honest-labeling rule and misclassifies a resumable timed
   stop as a solver failure. Not reachable by the recovery jobs (they set no
   `--wall-limit-s`) but reachable by every wall-limited campaign sharing
   `run_cg`, e.g. the snapshot-control continuations, where `master_failed`
   vs `wall_limit` changes `control_result_complete`. Fixed to label the stop
   `wall_limit`; genuine three-method failures keep `master_failed`.
   Regression tests: `test_wall_expiry_between_master_attempts_is_labeled_wall_limit`
   (fails against f85798c, passes with the fix) and
   `test_exhausting_all_master_methods_stays_labeled_master_failed`.
2. **The `.sub` truncated the flock diagnostic record before losing the
   race** (`src/submit_legacy_bigtariff_recovery.sub`). `exec 9>` opens with
   O_TRUNC, so a duplicate submission erased the current owner's
   `job=... host=... restart=...` line even though it then correctly failed
   the `flock -n` and exited. Kernel locking was never at risk; only the
   diagnostics that explain a SKIP_LOCKED. Fixed to `exec 9>>` (append).
   Guarded by `test_job_lock_fd_uses_append_mode`.

Also added `tests/test_legacy_recovery_scripts.py`: the launcher and `.sub`
previously had zero test coverage. It now pins bash parseability, the
requeue/time-limit/partition flags, append-mode locking, the legacy identity
parameters, the WAIT-grep classification against the real `MigrationError`
messages (missing witness matches; conflict must not), the task->cell mapping
against the original array arithmetic, and the dual commit pinning.

## 3. Operational risks needing Unicorn verification (not code changes)

1. **flock semantics on the results filesystem.** Both `flock(1)` in the
   `.sub` and `fcntl.flock` in `durable_io` assume real cross-node locking.
   On NFSv4 this holds; on Lustre mounted with `localflock` (or `noflock`)
   two jobs on different nodes could both "acquire" the lock. One-time check
   before `--submit`: run the `.sub`'s flock line from two different nodes
   against a scratch file under the same filesystem as
   `src/results/`, or check `mount | grep -E 'flock|nfs'`. The runbook's
   single-launcher discipline plus semantic job-name dedup makes collisions
   unlikely regardless.
2. **Witness availability.** Tasks 22/32 need a hashed *terminal* k30_r2
   status and a peak18/sek prices witness in `~/EVSP-DR/src/results`; task 24
   needs k30_r4. Which of the 8 completed 867334 cells (or older terminal
   runs) qualify is only knowable on the cluster; WAIT_NO_WITNESS handles
   absences cleanly. Expect the witness scan (recursive JSON load of the
   legacy results tree, four times per job start) to take minutes on NFS.
3. **Do not resubmit the failed legacy indices** (`sbatch --array=22,24,32
   submit_cg_bigtariffs.sub` per the old HANDOFF A1 advice). The old reader
   would crash again without modifying journals, but the crash tracebacks
   would extend `cg-bigtar_867334_{22,24,32}.out/.err`, changing archived-log
   hashes and hence the `migration_id`; an already-applied migration would
   then fail closed and need operator attention.
4. **Preemption during the short migration window itself** (between the raw
   archive landing and the status commit marker) fails closed on requeue
   ("destination status is missing but migration artifacts exist") and
   requires a human to preserve the orphans and pick a new output path. This
   is by design (fail-closed provenance) — just be aware the recovery is
   self-healing for pricer preemption, not for migration preemption.
5. **Python 3.12 conda fallback order.** The `.sub` activates the first env
   that exists and *then* asserts 3.12; if `/home/nc437/evsp_env` is still a
   3.10 env the job fatals immediately. The runbook already treats the first
   submissions as the real 3.12 check against the damaged tails.
6. **Real damaged-tail classes.** Local tests cover truncated JSON,
   concatenated-then-truncated records, non-UTF8, and non-object tails; the
   actual task-22/24/32 bytes are only on Unicorn. A refusal there is
   evidence to preserve, per the runbook.

## 4. Optional suggestions (do not block the recovery launch)

- Dead code in `exact_pricer_expanded.py`: the unreachable second `return`
  block at the end of `min_reduced_cost_route` (stale pre-refactor shape),
  and the `if iters_csv: iters_csv.close()` / `if journal: journal.close()`
  calls inside the snapshot-reconciliation section, where both handles are
  still `None` since f85798c moved handle-opening after that block.
- `_common_prefix_size` is a pure-Python byte loop; on multi-MB journals the
  preview+apply pair costs a few seconds. Fine for this recovery; consider
  `os.path.commonprefix`-style chunking if reused on bigger artifacts.
- The task->cell mapping lives in both the launcher and the `.sub` (each
  validates independently). The new mapping test pins them together; merging
  them into one source of truth would still be cleaner.
- `run_cg_snapshot_control.TERMINAL_STOP_REASONS` excludes `master_failed`,
  so a deterministic master failure would be re-attempted every runner pass.
  With fix 1, wall-exhausted stops no longer masquerade as `master_failed`,
  which removes the common case; a retry cap would remove the rest.
- The `.sub`'s fallback `#SBATCH -o src/cluster_logs/legacy_recovery/...`
  path only exists relative to the repo; direct `sbatch` submission from
  elsewhere (bypassing the launcher, which overrides `--output`) would fail
  to start. Harmless given the documented launcher-only flow.

## 5. Exact tests run and results

Environment: Python 3.12.3, `numpy==2.2.6 pandas==2.3.3 scipy==1.15.3
matplotlib==3.10.9` (pinned per `requirements-unicorn.txt`), pytest 9.x,
Linux; repository clean at each run.

1. `cd tests && python3 -m pytest -q` at `f85798c` (pre-change baseline):
   **204 passed, 23 subtests passed** in 20.2s.
2. End-to-end migration CLI script (real `main()`, real `_tool_identity`,
   synthetic legacy tree): read-only plan leaves sources byte-identical;
   `--apply` archives byte-identical raw copies + quarantined tails; re-apply
   idempotent; idempotent after journal extension; idempotent after simulated
   preemption mid-append plus pricer-style repair; missing witness exits 2
   matching the WAIT grep; conflicting witnesses exit 2 NOT matching the WAIT
   grep. **All passed.**
3. Cross-process lock script: contention refused with owner diagnostics;
   SIGKILL of the holder releases the lock for a requeue. **Passed.**
4. Live preemption simulation on `Practice_Custom_TwoDuty_13301_13302.csv`:
   fresh run SIGKILLed mid-CG (journal 433 columns), `--resume` completed
   (417 iterations, 7,813 columns, `final_lp_source=final_pool_resolve`),
   `iters.csv` elapsed monotone across attempts, rerun without `--resume`
   refused, concurrent duplicate refused by the output lock. **Passed.**
5. `bash -n` and `shellcheck -S warning` on both recovery scripts: **clean**.
6. Regression validity: `test_wall_expiry_between_master_attempts_is_labeled_wall_limit`
   **fails against unmodified f85798c** (observed `master_failed`) and passes
   with fix 1.
7. `cd tests && python3 -m pytest -q` after the fixes on
   `cursor/recovery-audit`: **213 passed, 28 subtests passed** in 20.7s
   (baseline 204 + 2 wall-limit labeling tests + 7 recovery-script tests).
