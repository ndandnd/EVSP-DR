# Ladder-lite — Cursor work order (self-contained)

**Date:** 2026-08-19 · **Author:** Claude (macbook2, no cluster access)
**Operator:** Nathan (macbook1 → Unicorn login `unicorn-login-01`)
**Status of prior campaign:** the gated `7937c22` campaign is **abandoned**; see §0.3.

This file is self-contained. You do not need any other document, and you do not
need cluster access — all live Slurm facts you need are in §1.

---

## 0. Why this exists

### 0.1 The measurement

The frozen 138-task scale ladder is **~325 job-hours** total:

| Group | Tasks | Job-hours | Longest task |
|---|---:|---:|---:|
| PREFLIGHT | 22 | ~2 | 4 h cap, minutes real |
| SEED | 21 | ~2 | 4 h cap, minutes real |
| CG (primary) | 23 | 210 | 24 h |
| CG_SENSITIVITY | 30 | 60 | 2 h |
| MIP_RAW | 21 | 27 | 4 h |
| MIP_KNOWN | 21 | 27 | 4 h |

Unicorn has **14,538 CPUs across 226 nodes** on `default_partition`, with **64
nodes fully idle** at the time of writing. At 20-way concurrency the entire
ladder finishes in **one overnight window**. We have produced **zero** rows in a
week. The bottleneck is the submission path, nothing else.

### 0.2 Root cause

`src/submit_scale_ladder.sub` will not run one CG iteration until **13**
fail-closed preconditions all pass in a single shot on a heterogeneous cluster:
approved-plan sha256; `python3` binary sha256; `scontrol` binary sha256; worker
self-hash; exact detached commit + clean tree + no untracked `*.py`; per-file
sha256 over ~22 reviewed sources; `tariff_response_environment --compare-plan`
portable-identity equality; instance/tariff sha256; a successful `scontrol
update JobName=`; two released-and-passed held probes; a held activation
controller that must *observe* the arrays inside a 6-second window; a held
scientific gate released with an observed postcondition; unclaimed no-clobber
reservations. `src/launch_scale_ladder.py` is 2822 lines of that. Any single
failure stops all 138 tasks.

The science underneath is twelve lines of bash.

### 0.3 Track A is closed — do not try to recover the gated campaign

Confirmed on the cluster today. Gate `250838` is, per `scontrol show job`:

```
JobId=250838 JobName=LDG373dd  JobState=PENDING Reason=JobHeldUser  Priority=0
```

That is precisely the state the recovery path exists to handle. It still fails:

```
File ".../src/reconcile_scale_ladder_gate.py", line 225, in _require_gate_held
    raise ValueError("gate is not proven held by the user")
```

The reconciler's own proof-of-held check refuses the exact scheduler state it
was written to recover, and both the activation controller (job `250113`,
FAILED after 13 s) and any direct reconcile invocation land on the same line.
**Do not debug `reconcile_scale_ladder_gate.py`. Do not touch job `250838`** —
it stays held as evidence. Ladder-lite is now the only path to data.

### 0.4 The design: keep the plan, replace the launcher

The valuable reviewed part of `launch_scale_ladder.py` is `build_plan()`: it
freezes all 138 cell definitions (budgets, snapshot marks, SOC step, block
minutes, threads, partition, output paths) and validates the instance manifest
against `data/scale_ladder/known_membership_preflight.json`. **It already runs
without submission**, via `--plan-out` + `--matrix-out`.

```
launch_scale_ladder.py --plan-out   →  approved-plan.json     unchanged, reviewed
scripts/ladder_lite/submit.sh       →  plain sbatch arrays    new
scripts/ladder_lite/run_cell.sh     →  index → job → exec     new
src/summarize_scale_ladder.py       →  the same CSVs          unchanged path
```

Both paths read the same `approved-plan.json`, so ladder-lite rows are
**scientifically identical** to what the gated campaign would have produced.
The only difference is provenance, which we label honestly as
`ladder_lite_direct_array`.

Of the 13 guards, keep exactly **two**, because each is one command and
`src/run_exact_pool_mip.py:113-128` already enforces them under Slurm:

- detached checkout at a named commit, clean tracked tree;
- `EVSP_EXPECTED_COMMIT` equal to that commit.

The other eleven are **absent from the code path**, not disabled.

---

## 1. Live cluster facts (measured 2026-08-19T23:52Z — do not re-derive)

Do not run Slurm commands. These are measured; build against them.

| Fact | Value | Consequence for your code |
|---|---|---|
| `default_partition` | `MaxTime=UNLIMITED`, `DefaultTime=04:00:00`, 226 nodes, 14538 CPUs | 24 h CG cells submit fine. **No clamping will ever trigger** — still write the clamp, it is 6 lines and inert. |
| `scaglione` | `MaxTime=UNLIMITED`, 6 nodes, 304 CPUs, 128 GB/node | MIP target partition, as the plan already specifies. |
| `MaxArraySize` | `1001` | Our largest group is 30. No chunking needed. |
| `MaxJobCount` | `40000`; no per-user `MaxSubmit`/`MaxJobs` | No throttling required. Use `%16` concurrency as courtesy. |
| **`DefMemPerCPU = 1000`**, `SelectType=select/cons_tres`, `SelectTypeParameters=CR_CORE_MEMORY` | **Memory is enforced per job** | **This is the highest-risk finding.** Without an explicit `--mem`, a `-c 2` CG task gets **2 GB** and will be killed on the larger rungs. `--mem` is mandatory in `submit.sh`. |
| Node memory pool | 99 nodes at ~32 GB, 64 **idle** at ~31.9 GB, 36 nodes at 257 GB+ | Keep default requests **≤ 24 G** so cells schedule onto the large idle pool. Escalate to 64 G only on a real OOM. |
| Approved Python | `/home/nc437/evsp_env/bin/python3.12` | Default `LL_PYTHON`. `/usr/bin/python3` is also 3.12.3 but is not the plan-approved env. |
| Gurobi licence | `/share/apps/software/gurobi/gurobi.lic`, mode 644, readable | The path the reviewed worker expects. Keep it. |
| Account / QOS | `Account=scaglione`, `QOS=normal`, no limits | No `-A` flag needed. |
| Home filesystem | 92 TB free | Artifact volume is a non-issue. |

**Memory table to implement** (4-line case statement, not a config file):

| Cell | `--mem` default |
|---|---|
| any phase, `scale ≤ 13` | `16G` |
| CG / CG_SENSITIVITY, `scale ≥ 20` | `24G` |
| MIP (any scale, 8 threads) | `24G` |

Escalation, documented in the README as an operator action, not automated:
if a cell dies with exit 137 / `slurmstepd: error: ... Exceeded job memory
limit`, resubmit that group with `--mem 64G`, which lands on the 257 GB nodes.
`status.sh` must surface OOM kills distinctly (see §2.4).

---

## 2. What to build

### Branch

Base `origin/cursor/duty-grid-transition-audit-2969` (`86b0a42`).
New branch **`cursor/ladder-lite-20260819-2969`**. Push to `origin`.

Do not merge, rebase, or cherry-pick any other branch. Do not touch
`peel-and-price`.

### Hard constraints

1. **Do not modify any existing file** except `src/summarize_scale_ladder.py`,
   and only if §2.5 requires it, additively, ≤40 changed lines. If you believe
   another existing file must change, **stop and say so** instead of changing it.
2. **Total new shell + Python ≤ 400 lines**, excluding README, tests, and CSV
   headers. If your design exceeds that, the design is wrong. Simplify.
3. **Forbidden. Do not write, import, or call any of these:**
   - environment probes; `tariff_response_environment --compare-plan`; portable
     environment identity comparison of any kind;
   - activation controllers, gate jobs, held jobs, `scontrol release`,
     `scontrol update`;
   - reservations, campaign locks, no-clobber directory claims;
   - sha256 verification of `python3`, `scontrol`, the worker script, or the
     ~22 reviewed source paths;
   - `--dependency` between arrays, or any inter-array dependency;
   - retry state machines, submission intents, restart reconcilers;
   - new abstraction layers, new config formats, new manifest schemas.
4. **No `set -e` anywhere.** In operator-facing scripts, no `set -euo pipefail`
   at all — a pasted `set -e` has already killed this operator's SSH session. In
   `run_cell.sh` (executed by Slurm, not a human) `set -uo pipefail` is fine;
   `set -e` is not, because a failed cell must write a `.failed` marker rather
   than die silently.
5. Every operator action is **one committed script plus one command**. No
   multi-line paste blocks in the README.

### 2.1 `scripts/ladder_lite/plan.sh`

```
usage: plan.sh
env:   LL_ROOT=$HOME/ladder-lite   LL_PYTHON=/home/nc437/evsp_env/bin/python3.12
       LL_CAMPAIGN=ll_$(date -u +%Y%m%d)
```

1. `REPO` = git root containing this script. Require detached HEAD and empty
   `git status --porcelain --untracked-files=no`. These two checks stay — they
   are the guards we deliberately kept. Print the commit.
2. `mkdir -p "$LL_ROOT/campaign"`.
3. Run, unchanged and **without `--submit`**:
   `"$LL_PYTHON" -B src/launch_scale_ladder.py --campaign "$LL_CAMPAIGN" --reservation-root "$LL_ROOT" --plan-out "$LL_ROOT/campaign/approved-plan.json" --matrix-out "$LL_ROOT/campaign/task_matrix.csv"`
   - If `build_plan()` refuses for a reason unrelated to submission (Python
     version, detached checkout, manifest hash), **satisfy the demand**; do not
     patch `launch_scale_ladder.py`.
   - If it refuses for a submission-only reason, report the exact message and
     stop. Do not work around it by copying `build_plan` into new code.
4. Stage instance inputs by calling the existing helper — do not reimplement:
   ```
   "$LL_PYTHON" -B -c 'import sys,json;sys.path.insert(0,"src");import launch_scale_ladder as L;L._stage_scientific_inputs(json.load(open(sys.argv[1])))' "$LL_ROOT/campaign/approved-plan.json"
   ```
5. Write `$LL_ROOT/campaign/campaign.json` with:
   `approval_sha256` (sha256 of the plan file bytes), `execution_mode:
   "ladder_lite_direct_array"`, `campaign`, `commit`, `created_utc`, plus any
   additional keys `src/summarize_scale_ladder.py` actually reads — read the
   normalizer and add only those.
6. Print plan path, plan sha256, and per-group task counts read back from the
   plan. **Assert the total is 138** and fail loudly if not.

### 2.2 `scripts/ladder_lite/run_cell.sh`

```
usage: run_cell.sh <PLAN_JSON> <GROUP>      (reads $SLURM_ARRAY_TASK_ID)
```

1. Resolve the job exactly as `src/submit_scale_ladder.sub` does:
   `plan["task_groups"][GROUP][SLURM_ARRAY_TASK_ID]` → `job_key` → the single
   matching entry in `plan["jobs"]`. Copy that resolution logic; it is correct
   and short. Extract at minimum `job_key, phase, arm, scale,
   selection_replicate, cg_replicate, budget_s, threads, soc_step, block_min,
   snapshot_minutes, instance.path, instance.relative_path,
   instance.instance_file_sha256, output, progress_dir, telemetry,
   dependency_cg, dependency_seed, dependency_preflight`.
2. `OUT` = the job's `output`. If `"$OUT.done"` exists → print `SKIP <job_key>`,
   exit 0. **This is the only idempotency mechanism. Do not add others.**
3. Install the staged instance under `data/` via the existing
   `src/install_exact_cg_profile_input.py`, with the same arguments
   `submit_scale_ladder.sub` uses.
4. Export `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` /
   `NUMEXPR_NUM_THREADS` = 1 for CG phases, = `threads` for MIP. Export
   `EVSP_EXPECTED_COMMIT="$(git -C "$REPO" rev-parse HEAD)"`.
   **Do not export `EVSP_REQUIRE_DETACHED`** — the checkout is already detached
   and a second enforcement path buys nothing.
5. Dispatch by phase using the **exact same commands** as
   `src/submit_scale_ladder.sub`:
   - `PREFLIGHT` → `src/audit_scale_ladder_known_membership.py`
   - `SEED` → `src/prepare_scale_ladder_known_partition.py`
   - `CG` / `CG_SENSITIVITY` → `src/exact_pricer_expanded.py` with the plan's
     values: `--csv <instance.relative_path> --prices_csv hourly_prices_flat.csv
     --g-kwh 300 --charge-kw 300 --min-soc-frac 0 --soc-step <soc_step>
     --block-min <block_min> --master-sense partition --initial-pool singletons
     --wall-limit-s $((budget_s + 60)) --checkpoint-every 25 --resume
     --snapshot-at-minutes <snapshot_minutes> --out <output>`, plus
     `--phase-telemetry <telemetry>` when `telemetry` is non-null.
   - `MIP` → `src/run_exact_pool_mip.py --result <dep CG output> --two-stage
     --threads <threads> --timelimit <budget_s> --mipgap 0.0001 --progress-dir
     <progress_dir> --out <output>`, plus `--initial-partition-routes <SEED
     output>` when `arm == "KNOWN-PARTITION"`.
6. MIP-only environment. `run_exact_pool_mip.py` **requires** these under
   Slurm — compute them from the real files with `sha256sum`, never from a
   manifest: `EVSP_MIP_EXPECTED_RESULT_SHA256`,
   `EVSP_MIP_EXPECTED_JOURNAL_SHA256`, and (KNOWN arm only)
   `EVSP_MIP_EXPECTED_INITIAL_PARTITION_SHA256`. Resolve the journal from the CG
   status `columns_journal` field relative to the status file's directory, as
   the reviewed worker does. Export
   `GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic` and
   `unset LM_LICENSE_FILE`.
7. MIP preconditions: dependency CG `output` and its `.done` marker exist; SEED
   output exists for the KNOWN arm. If not, write `"$OUT.blocked"` with the
   reason and **exit 0** — a not-yet-ready MIP is not a failure, and we
   resubmit the group later.
8. Forward `SIGUSR1`/`SIGTERM` to the child as the reviewed worker does. Keep
   this; it is why `--resume` works after a walltime kill.
9. On child exit 0 → `touch "$OUT.done"`. On nonzero → write `"$OUT.failed"`
   containing the exit code, `job_key`, `$SLURM_JOB_ID`, `$SLURMD_NODENAME`, and
   the last 40 lines of stderr if you captured it; exit with the child's code.
10. **Do not** write `worker-completion.json`, do not hash the artifact set, do
    not validate snapshot availability. That is the normalizer's job.
11. Optional `LL_BUDGET_OVERRIDE_S`: applies **only** to `--wall-limit-s` /
    `--timelimit`, and the override value **must be recorded in the emitted
    status JSON** (or, if the pricer will not accept an extra field, in a
    sibling `"$OUT.override.json"`) so an overridden smoke run can never be
    mistaken for a real cell.

### 2.3 `scripts/ladder_lite/submit.sh`

```
usage: submit.sh <GROUP> [--scales 2,3,5] [--concurrency N] [--partition P]
                         [--mem 24G] [--dry-run]
GROUP ∈ PREFLIGHT SEED CG CG_SENSITIVITY MIP_RAW MIP_KNOWN
env:   LL_ROOT (default $HOME/ladder-lite)
```

1. Read `$LL_ROOT/campaign/approved-plan.json`. Build the ordered index list for
   `GROUP`, preserving index ↔ `task_groups` position exactly — that mapping is
   the contract with `run_cell.sh`.
2. `--scales` filters on the job's `scale`.
3. **Partition indices by `budget_s`** and submit one `sbatch` per distinct
   budget with an explicit comma index list (`--array=0,3,7%N`). Walltime =
   `budget_s + 1800`, formatted `H:MM:SS`.
4. `partition` and `threads` come from the plan's jobs (already
   `default_partition` for CG, `scaglione` for MIP; threads 2 and 8).
   `--partition`, `--concurrency`, `--mem` override. `--mem` default from the
   §1 table. Default concurrency `%16`.
5. Clamp walltime to the partition's `MaxTime` if smaller, print a loud warning
   naming affected cells, and **continue**. (On this cluster `MaxTime` is
   `UNLIMITED`, so this never fires. Write it anyway; do not refuse a
   submission.) A censored 24 h cell that ran 12 h is data; a cell that never
   ran is not.
6. `sbatch --requeue --parsable`, `-J ll_<GROUP>_k<scales>`, logs to
   `$LL_ROOT/logs/ll_<GROUP>_%A_%a.out` / `.err`.
7. Append one line per submitted array to `$LL_ROOT/submitted.tsv`:
   `utc  group  array_id  budget_s  partition  mem  n_tasks  index_list`.
8. Print the array IDs and total task count. Nothing else.
9. `--dry-run` prints the exact `sbatch` command lines and exits 0.

### 2.4 `scripts/ladder_lite/status.sh`

```
usage: status.sh [GROUP]
```

Prints, in this order, and nothing more:

1. one line per group: `GROUP  planned  done  failed  blocked  running  missing`
   (markers from disk, `running` from `squeue -u $USER -n ll_*`);
2. compact `squeue` for the user;
3. for the three most recently modified CG `*.iters.csv`: file name and last line;
4. first 5 lines of every `*.failed` marker;
5. **an explicit OOM line**: count of tasks whose `sacct` `State` is
   `OUT_OF_MEMORY` or whose `.err` log contains `Exceeded job memory limit`,
   with their cell IDs. Given `DefMemPerCPU=1000` this is the failure mode most
   likely to bite, and it must not hide inside a generic failure count.

### 2.5 `scripts/ladder_lite/normalize.sh`

Calls `src/summarize_scale_ladder.py --campaign-root "$LL_ROOT/campaign"
--out-dir "$LL_ROOT/normalized"`.

Read the normalizer first. It validates a "completed verified scheduler
contract" (~line 205) and the campaign approval hash (~line 345). If it refuses
a ladder-lite campaign, add **one** additive, narrowly scoped escape:
`--execution-mode ladder-lite-direct`, which skips **only** the
scheduler-receipt validation and stamps `provenance=ladder_lite_direct_array` on
every emitted row. ≤40 changed lines. Do not refactor the normalizer, do not
touch its schemas, **do not weaken the approval-hash check**.

### 2.6 `records/` — durable, append-only project record

The operator has asked for durable records of bugs, decisions, and results, in
CSV. Seed files already exist in the repo root at `records/` (I created them
with headers and today's rows). Your job is to keep them working, not to
redesign them:

- `records/DECISION_LOG.csv` — one row per irreversible or scientifically
  material decision.
- `records/BUG_LOG.csv` — one row per defect, with root cause and fix commit.
- `records/RESULTS_LOG.csv` — one row per completed cell, the durable
  human-readable summary. Column list is fixed by the seed file's header; do not
  add or reorder columns.
- `records/README.md` — 20 lines: what each file is, and the rule that rows are
  **append-only and never edited in place** (corrections get a new row with
  `supersedes` set).

Add `scripts/ladder_lite/record_results.sh`:

```
usage: record_results.sh <RUN_ID>
```

1. Copy `$LL_ROOT/normalized/*.csv` into `records/runs/<RUN_ID>/` inside the
   repo.
2. Append one row per cell to `records/RESULTS_LOG.csv`, joining
   `cg_run_summary.csv` and `mip_run_summary.csv` on `job_key`, filling the
   fixed header. Include `execution_mode`, `commit`, and the artifact sha256.
3. Deduplicate on `(run_id, cell_id)` — appending twice must not create
   duplicate rows.
4. Print the number of rows appended and the number skipped as duplicates.

≤60 lines. This counts toward the 400-line budget.

### 2.7 `scripts/ladder_lite/README.md`

≤40 lines: the operator's exact command sequence, the `--mem 64G` OOM
escalation, and nothing else. No rationale, no architecture notes.

### 2.8 `tests/test_ladder_lite.py`

Small and real:

1. Array index → `job_key` resolution agrees with `plan["task_groups"]` for all
   138 cells — drive the real resolution snippet as a subprocess against a plan
   built in a tmpdir.
2. **Write this one first.** For one cell of each phase, the command line
   `run_cell.sh` builds is **string-identical** (modulo paths) to the command
   `src/submit_scale_ladder.sub` builds for the same job. This is the test that
   protects scientific equivalence between ladder-lite and the reviewed worker.
3. `submit.sh --dry-run` groups indices by `budget_s` correctly and emits the
   right walltimes and `--mem` values for a synthetic 3-budget, 2-scale group.
4. `--scales` filtering selects exactly the expected `job_key`s.
5. `record_results.sh` is idempotent: running it twice appends no duplicates.

Do not mock `sbatch`, `squeue`, or `scontrol` beyond what item 3 needs.
Mock-heavy scheduler tests are exactly what failed last week: hundreds of tests
passed while every live launch died on real cluster behaviour.

---

## 3. Acceptance — what you must report back

Do not report "ready", "packaged", "armed", or "tests pass". Report **executed
output**:

1. `git log --oneline -1` and the pushed branch name.
2. `bash scripts/ladder_lite/plan.sh` output, including the 138-task assertion.
3. **A real local CG run.** Run `run_cell.sh` directly, no Slurm, for the
   PREFLIGHT k2 s1 cell and the CG k2 s1 c1 cell with
   `LL_BUDGET_OVERRIDE_S=180`. Paste:
   - the PREFLIGHT output JSON;
   - the last 5 lines of the CG `*.iters.csv`;
   - the CG status JSON's `route_weight`, `min_reduced_cost`, `stop_reason`,
     `n_columns`;
   - proof the override was recorded in the artifact.
4. `bash scripts/ladder_lite/submit.sh CG --scales 2,3,5 --dry-run` output —
   I will check the `--mem`, `-c`, walltime, and index lists by hand before
   Nathan submits anything.
5. `pytest tests/test_ladder_lite.py -q` output.
6. `wc -l scripts/ladder_lite/*.sh` plus the diff stat for
   `src/summarize_scale_ladder.py` if you touched it.
7. Any row you added to `records/BUG_LOG.csv` or `records/DECISION_LOG.csv`
   while doing this work.

**If item 3 does not produce a real CG iteration on your machine, say so plainly
and stop.** Do not proceed to polish anything else. One executed k2 iteration is
worth more than the entire rest of this work order.

---

## 4. Standing rules for this project

1. **Report executed output, never readiness.** The only reportable states are:
   *scientific tasks submitted → running → outputs on disk → normalized CSVs*.
2. **One scientific worker before any infrastructure work.** If a launch path
   cannot produce one k2 CG iteration, hardening around it is worthless.
3. **Fail open on liveness, fail closed on interpretation.** Never block a run
   because a label might be wrong. Run it, label it honestly
   (`ladder_lite_direct_array`, `legacy_scheduler_unverified`,
   `combined-cost-master route weight`), and let the normalizer carry the caveat.
4. **`route_weight` is `combined-cost-master route weight`**, not a fleet LP
   lower bound, until the three-phase lexicographic master exists. Every axis
   label and table header says so.
5. **RAW and KNOWN never share a row.** KNOWN is a plumbing positive control,
   not algorithmic recovery.
6. **Every work order carries a line budget and a forbidden list.** The failure
   mode here has never been incompetence; it is unbounded scope.
