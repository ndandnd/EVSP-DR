# Ladder-lite directive — 2026-08-19

> **AMENDED 2026-08-19T23:52Z after cluster recon. Read this banner first.**
>
> 1. **Track A (§1.5, Block 1) is CLOSED. Do not run Block 1.** Gate `250838` is
>    verifiably `PENDING/JobHeldUser`, and
>    `reconcile_scale_ladder_gate.py:225` still raises
>    `ValueError: gate is not proven held by the user`. The recovery path refuses
>    the exact state it exists to recover; activation job `250113` died on the
>    same line. Leave `250838` held as evidence. Ladder-lite is the only path.
> 2. **Branch is `cursor/ladder-lite-20260819-2969`** (Cursor's naming
>    convention), not `cursor/ladder-lite-20260819`.
> 3. **Cursor's authoritative work order is now
>    `EVSP-DR/LADDER_LITE_CURSOR_WORKORDER_20260819.md`** — self-contained, with
>    measured cluster facts folded in. Part 2 below is superseded by it.
> 4. **New mandatory finding:** `DefMemPerCPU=1000` with `CR_CORE_MEMORY` means
>    an omitted `--mem` caps a 2-CPU CG task at 2 GB. `submit.sh` must always
>    pass `--mem` (16G for scale ≤ 13, 24G for scale ≥ 20 and MIP; 64G on OOM).
> 5. `default_partition` and `scaglione` are both `MaxTime=UNLIMITED`, so the
>    24 h cells submit unclamped. `MaxArraySize=1001`. Approved Python is
>    `/home/nc437/evsp_env/bin/python3.12`.
> 6. Durable records now live in `EVSP-DR/records/` (`DECISION_LOG.csv`,
>    `BUG_LOG.csv`, `RESULTS_LOG.csv`).

**Author:** Claude (macbook2, no cluster access)
**Operator:** Nathan (macbook1 → Unicorn)
**Implementer:** Cursor (pushes to `origin`)

This document has three parts:

- **Part 1 — Diagnosis and decision.** Why the cluster is idle and what we do instead.
- **Part 2 — Cursor work order.** Paste Part 2 into Cursor verbatim.
- **Part 3 — Operator bash.** Run these on macbook1 against Unicorn, in order.

---

## Part 1 — Diagnosis and decision

### 1.1 The measurement that matters

The whole frozen ladder is **138 tasks ≈ 325 job-hours**:

| Group | Tasks | Job-hours | Longest single task |
|---|---:|---:|---:|
| PREFLIGHT | 22 | ~2 | 4 h cap, minutes real |
| SEED | 21 | ~2 | 4 h cap, minutes real |
| CG (primary) | 23 | 210 | 24 h |
| CG_SENSITIVITY | 30 | 60 | 2 h |
| MIP_RAW | 21 | 27 | 4 h |
| MIP_KNOWN | 21 | 27 | 4 h |

If Unicorn runs even 20 tasks concurrently, **the entire ladder finishes in one
overnight window (~26 h wall)**. We have spent a week and produced zero rows on
something that is one night of compute. The bottleneck is not the cluster, the
solver, or the science. It is the submission path.

### 1.2 Root cause

`src/submit_scale_ladder.sub` (the reviewed worker) will not run a single CG
iteration until **all** of the following succeed, in one shot, on a
heterogeneous cluster:

1. approved-plan sha256 match;
2. `python3` binary sha256 match;
3. `scontrol` binary sha256 match;
4. worker-script self-hash match;
5. exact detached commit + clean tracked tree + no untracked `*.py`;
6. per-file sha256 over ~22 reviewed source paths;
7. `tariff_response_environment --compare-plan` portable-identity equality
   (this is what NumPy SIMD differences broke);
8. instance / tariff sha256 match;
9. a successful `scontrol update JobName=…` call;
10. two held environment probes released and passed;
11. a held activation controller that must *observe* the arrays within a
    6-second window;
12. a held scientific gate released with an observed postcondition;
13. no-clobber reservations not already taken.

Every one of those is a total stop. `launch_scale_ladder.py` is 2822 lines of
that. Meanwhile the actual science command is this:

```
python -u src/exact_pricer_expanded.py \
  --csv <rel.csv> --prices_csv hourly_prices_flat.csv \
  --g-kwh 300 --charge-kw 300 --min-soc-frac 0 \
  --soc-step 15 --block-min 10 \
  --master-sense partition --initial-pool singletons \
  --wall-limit-s <budget+60> --checkpoint-every 25 --resume \
  --snapshot-at-minutes <marks> --out <out.json> [--phase-telemetry <t.jsonl>]
```

Twelve lines. Guards 1–13 exist to protect provenance. Provenance is worth
protecting, but **provenance of nothing is worth nothing.** We are paying a
100% liveness tax for a property we can also get by pinning one detached commit.

### 1.3 What we keep and what we delete

Of guards 1–13, exactly two carry real scientific weight and both are cheap:

- **detached checkout at a named commit, clean tracked tree** — one
  `git checkout --detach`. `src/run_exact_pool_mip.py` already enforces this
  under Slurm via `EVSP_EXPECTED_COMMIT`, so we satisfy it rather than fight it.
- **input/output hashes recorded in the artifacts** — the pricer and the MIP
  already record instance/prices/reference/deadhead sha256 internally. We do
  not need a second hashing layer at submit time.

Everything else — probes, activation, gate, reservations, binary hashing,
portable-identity equality, dependency chains — goes away. Not "is disabled":
is **not in the code path at all**.

### 1.4 The design: reuse the plan, replace the launcher

The valuable, reviewed part of `launch_scale_ladder.py` is `build_plan()`: it
freezes 138 cell definitions (budgets, snapshot marks, SOC step, block minutes,
threads, partition, output paths) and validates the instance manifest against
`known_membership_preflight.json`. That part is already exposed **without
submission**: `--plan-out` + `--matrix-out` work with no `--submit`.

So:

```
launch_scale_ladder.py --plan-out        →  approved-plan.json   (unchanged, reviewed)
scripts/ladder_lite/submit.sh            →  plain sbatch arrays  (new, ~120 lines)
scripts/ladder_lite/run_cell.sh          →  index → job → exec   (new, ~90 lines)
src/summarize_scale_ladder.py            →  the same CSVs        (unchanged path)
```

Because both paths read the same `approved-plan.json`, the ladder-lite rows are
**scientifically identical** to what the gated campaign would have produced.
The only difference is the provenance label, and we label it honestly:
`execution_mode = ladder_lite_direct_array`.

### 1.5 Two tracks tonight

- **Track A (10 minutes, one shot, Nathan).** Try the `7937c22` reconciler
  once. If it records six arrays, we are done and Track B is a bonus.
  **Hard stop:** if it does not print six array IDs on the first attempt, do not
  debug it. Do not read its traceback for more than one minute. Go to Track B.
- **Track B (primary, Cursor tonight → Nathan submits).** Ladder-lite as above.

Track A costs one paste block. Track B is what we are actually betting on.
Track B writes to a **different root** (`$HOME/ladder-lite/`) so the two cannot
collide, and `submit.sh` takes a `--scales` filter so we never pay for the same
k30 cell twice.

### 1.6 Success criterion — the only one that counts

> **Within 10 minutes of the first `submit.sh`: array IDs printed, and at least
> one PREFLIGHT output JSON exists on disk. Within 3 hours: nine CG status JSONs
> at k2/k3/k5 with `iters.csv` rows.**

"Packaged", "pushed", "tests pass", "campaign armed" are not reportable states.
Do not report anything else as progress.

---

## Part 2 — Cursor work order

> Paste everything from here to the end of Part 2 into Cursor.

### Task

Add a minimal direct-array launch path (`ladder-lite`) for the existing frozen
138-task scale ladder. It must reuse `approved-plan.json` produced by the
existing reviewed `src/launch_scale_ladder.py`, and must contain **no** probe,
activation, gate, reservation, binary-hash, or portable-environment-identity
logic.

### Branch

Base: `origin/cursor/duty-grid-transition-audit-2969` (`86b0a42`).
New branch: `cursor/ladder-lite-20260819`. Push to `origin`.

Do not merge, rebase, or cherry-pick any other branch. Do not touch
`peel-and-price`.

### Hard constraints

1. **Do not modify any existing file** except:
   - `src/summarize_scale_ladder.py` — only if required by item 6 below, and
     only additively, ≤40 changed lines.
   - No other existing file may change. If you believe one must, stop and say
     so instead of changing it.
2. **Total new shell + Python ≤ 350 lines** excluding the README and tests. If
   your design exceeds that, the design is wrong; simplify it.
3. **Forbidden — do not write, import, or call any of these:**
   - environment probes; `tariff_response_environment --compare-plan`; portable
     identity comparison of any kind;
   - activation controllers, gate jobs, held jobs, `scontrol release`,
     `scontrol update`;
   - reservations, campaign locks, no-clobber directory claims;
   - sha256 verification of `python3`, `scontrol`, the worker script, or the
     ~22 reviewed source paths;
   - `--dependency` between arrays; any inter-array dependency;
   - retry state machines, submission intents, restart reconcilers;
   - new abstraction layers, new config formats, new manifest schemas.
4. **No `set -euo pipefail` in any script an operator pastes or sources.**
   Inside `run_cell.sh` (which Slurm executes, not a human) `set -uo pipefail`
   is fine; `set -e` is not — we want explicit exit-code handling so a failed
   cell writes a `.failed` marker instead of dying silently.
5. Every operator-facing action must be **one committed script + one command**.
   No pasted multi-line blocks in the README.

### Files to add

#### 1. `scripts/ladder_lite/plan.sh`

Creates the lite campaign root and the plan, using existing reviewed code.

```
usage: plan.sh            (env: LL_ROOT, LL_PYTHON, LL_CAMPAIGN)
defaults: LL_ROOT=$HOME/ladder-lite  LL_PYTHON=python3  LL_CAMPAIGN=ll_$(date +%Y%m%d)
```

Steps:

1. `REPO` = the git root containing this script; require detached HEAD and a
   clean tracked tree (`git status --porcelain --untracked-files=no` empty).
   These two checks stay — they are the guards we chose to keep. Print the
   commit.
2. `mkdir -p "$LL_ROOT/campaign"`.
3. Run, unchanged:
   `"$LL_PYTHON" -B src/launch_scale_ladder.py --campaign "$LL_CAMPAIGN" --reservation-root "$LL_ROOT" --plan-out "$LL_ROOT/campaign/approved-plan.json" --matrix-out "$LL_ROOT/campaign/task_matrix.csv"`
   (no `--submit`).
   - If `build_plan()` refuses for a reason unrelated to submission (e.g. it
     demands a detached checkout, or a Python version), satisfy the demand;
     do not patch `launch_scale_ladder.py`.
   - If it refuses for a submission-only reason, report the exact message and
     stop. Do not work around it by copying `build_plan` into new code.
4. Stage the instance inputs by calling the existing helper — do not
   reimplement:
   ```
   "$LL_PYTHON" -B -c 'import sys,json;sys.path.insert(0,"src");import launch_scale_ladder as L;L._stage_scientific_inputs(json.load(open(sys.argv[1])))' "$LL_ROOT/campaign/approved-plan.json"
   ```
5. Write `$LL_ROOT/campaign/campaign.json` containing exactly:
   `{"approval_sha256": <sha256 of the plan file bytes>, "execution_mode": "ladder_lite_direct_array", "campaign": "<LL_CAMPAIGN>", "commit": "<HEAD>", "created_utc": "<ISO8601>"}`
   plus any additional keys `src/summarize_scale_ladder.py` requires in order to
   load the campaign (read the normalizer and add only what it actually reads).
6. Print: plan path, plan sha256, and the per-group task counts read back from
   the plan. Assert the total is 138 and fail loudly if not.

#### 2. `scripts/ladder_lite/run_cell.sh`

The Slurm worker. One array task = one cell.

```
usage: run_cell.sh <PLAN_JSON> <GROUP>      (reads $SLURM_ARRAY_TASK_ID)
```

1. Resolve `job` from the plan exactly the way `src/submit_scale_ladder.sub`
   does: `plan["task_groups"][GROUP][SLURM_ARRAY_TASK_ID]` → `job_key` → the
   single matching entry in `plan["jobs"]`. Copy that resolution logic; it is
   correct and short. Extract at minimum: `job_key, phase, arm, scale,
   selection_replicate, cg_replicate, budget_s, threads, soc_step, block_min,
   snapshot_minutes, instance.path, instance.relative_path,
   instance.instance_file_sha256, output, progress_dir, telemetry,
   dependency_cg, dependency_seed, dependency_preflight`.
2. `OUT="$job.output"`. If `"$OUT.done"` exists → print `SKIP <job_key>`,
   exit 0. This is the only idempotency mechanism. Do not add others.
3. Install the staged instance under `data/` using the existing
   `src/install_exact_cg_profile_input.py` (unchanged call, same arguments as
   `submit_scale_ladder.sub` uses).
4. Export: `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/
   `NUMEXPR_NUM_THREADS` = 1 for CG phases, = `threads` for MIP;
   `EVSP_EXPECTED_COMMIT="$(git -C "$REPO" rev-parse HEAD)"`.
   **Do not export `EVSP_REQUIRE_DETACHED`** — the checkout is already detached
   and we do not want a second enforcement path.
5. Dispatch by phase using the **exact same commands** as
   `src/submit_scale_ladder.sub`:
   - `PREFLIGHT` → `src/audit_scale_ladder_known_membership.py`
   - `SEED` → `src/prepare_scale_ladder_known_partition.py`
   - `CG` / `CG_SENSITIVITY` → `src/exact_pricer_expanded.py` with the flags
     from the plan (`--wall-limit-s $((budget_s + 60))`, `--checkpoint-every 25
     --resume`, `--snapshot-at-minutes` from `snapshot_minutes`,
     `--phase-telemetry` when `telemetry` is non-null)
   - `MIP` → `src/run_exact_pool_mip.py --result <dep CG output> --two-stage
     --threads <threads> --timelimit <budget_s> --mipgap 0.0001
     --progress-dir <progress_dir> --out <output>`, plus
     `--initial-partition-routes <SEED output>` when `arm == "KNOWN-PARTITION"`.
   For MIP, `run_exact_pool_mip.py` **requires** these under Slurm — set them
   from the real files with `sha256sum`, not from any manifest:
   `EVSP_MIP_EXPECTED_RESULT_SHA256`, `EVSP_MIP_EXPECTED_JOURNAL_SHA256`, and
   `EVSP_MIP_EXPECTED_INITIAL_PARTITION_SHA256` (KNOWN arm only). Resolve the
   journal path from the CG status `columns_journal` field relative to the
   status file's directory, as the reviewed worker does. Export
   `GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic` and `unset
   LM_LICENSE_FILE`.
   Preconditions for MIP: dependency CG `output` exists and its `.done` marker
   exists; SEED output exists for the KNOWN arm. If not, write
   `"$OUT.blocked"` with the reason and exit 0 (not nonzero — a not-yet-ready
   MIP is not a failure, and we resubmit that group later).
6. Forward `SIGUSR1`/`SIGTERM` to the child, as the reviewed worker does, so
   checkpointing on timeout still works. Keep this; it is why `--resume` works.
7. On child exit 0: `touch "$OUT.done"`. On nonzero: write
   `"$OUT.failed"` containing the exit code, the job key, `$SLURM_JOB_ID`, and
   the last 40 lines of the child's stderr if you captured it; exit with the
   child's code.
8. Do **not** write a `worker-completion.json`, do not hash the artifact set,
   do not validate snapshot availability. That belongs to the normalizer.

#### 3. `scripts/ladder_lite/submit.sh`

The operator entry point. Plain `sbatch`, no dependencies.

```
usage: submit.sh <GROUP> [--scales 2,3,5] [--concurrency N] [--partition P]
                         [--mem 16G] [--dry-run]
GROUP ∈ PREFLIGHT SEED CG CG_SENSITIVITY MIP_RAW MIP_KNOWN
env: LL_ROOT (default $HOME/ladder-lite)
```

Behavior:

1. Read `$LL_ROOT/campaign/approved-plan.json`. Build the ordered index list for
   `GROUP`; keep index ↔ `task_groups` position exact (that mapping is the
   contract with `run_cell.sh`).
2. Apply `--scales` as a filter on the job's `scale`.
3. **Partition indices by `budget_s`** and submit one `sbatch` per distinct
   budget, using an explicit comma index list
   (`--array=0,3,7%N`). Walltime = `budget_s + 1800`, formatted `H:MM:SS`.
4. Read `partition` and `threads` from the plan's jobs (they are already
   `default_partition` for CG and `scaglione` for MIP, threads 2/8);
   `--partition` and `--concurrency` override. `--mem` default: `16G` for
   scale ≤ 13, `32G` for scale ≥ 20 and for all MIP. Make the default table a
   4-line case statement, not a config file.
5. If the target partition's `MaxTime` (from `scontrol show partition`) is less
   than the requested walltime, clamp to `MaxTime`, print a loud warning naming
   the affected cells, and continue. Do not refuse. A censored 24 h cell that
   ran for 12 h is data; a cell that never ran is not.
6. `sbatch --requeue`, `-J ll_<GROUP>_k<scales>`, output to
   `$LL_ROOT/logs/ll_<GROUP>_%A_%a.out` / `.err`, `--parsable`.
7. Append one line per submitted array to `$LL_ROOT/submitted.tsv`:
   `utc  group  array_id  budget_s  partition  n_tasks  index_list`.
8. Print the array IDs and the total task count. Nothing else.
9. `--dry-run` prints the exact `sbatch` command lines and exits 0.

#### 4. `scripts/ladder_lite/status.sh`

```
usage: status.sh [GROUP]
```

Prints, in this order, and nothing more:

1. one line per group: `GROUP  planned  done  failed  blocked  running  missing`
   (`done`/`failed`/`blocked` from the marker files, `running` from
   `squeue -u $USER -n ll_*`);
2. `squeue` for the user, compact format;
3. for the three most recently modified CG `*.iters.csv`: the file name and its
   last line;
4. the first 5 lines of every `*.failed` marker.

#### 5. `scripts/ladder_lite/normalize.sh`

Calls `src/summarize_scale_ladder.py --campaign-root "$LL_ROOT/campaign"
--out-dir "$LL_ROOT/normalized"`.

Read the normalizer first. It validates a "completed verified scheduler
contract" (~line 205) and the campaign approval hash (~line 345). If it refuses
a ladder-lite campaign, add **one** additive, narrowly scoped escape:
`--execution-mode ladder-lite-direct`, which skips **only** the scheduler-receipt
validation and stamps `provenance=ladder_lite_direct_array` on every emitted
row. ≤40 changed lines. Do not refactor the normalizer, do not touch its
schemas, do not weaken the approval-hash check.

#### 6. `scripts/ladder_lite/README.md`

≤40 lines. The operator's exact command sequence and nothing else. No
rationale, no architecture notes.

#### 7. `tests/test_ladder_lite.py`

Small and real. Assert:

1. array index → `job_key` resolution in `run_cell.sh` agrees with
   `plan["task_groups"]` for all 138 cells (drive the resolution snippet as a
   subprocess against a real plan built in a tmpdir).
2. For one cell of each phase, the command line `run_cell.sh` would build is
   **string-identical** (modulo paths) to the command
   `src/submit_scale_ladder.sub` builds for the same job. This is the test that
   protects scientific equivalence — write it first.
3. `submit.sh --dry-run` groups indices by `budget_s` correctly and emits the
   right walltimes for a synthetic 3-budget group.
4. `--scales` filtering selects exactly the expected `job_key`s.

Do not add mocks of `sbatch`, `squeue`, or `scontrol` beyond what item 3 needs.
Mock-heavy scheduler tests are precisely what failed us last week.

### Acceptance — what you must report back

Do not report "ready", "packaged", or "tests pass". Report **executed output**:

1. `git log --oneline -1` and the branch name pushed.
2. `bash scripts/ladder_lite/plan.sh` output on your machine, including the
   138-task assertion.
3. **A real local CG run.** Run `run_cell.sh` directly (no Slurm) for the
   PREFLIGHT k2 s1 cell and for the CG k2 s1 c1 cell with
   `LL_BUDGET_OVERRIDE_S=180` (add that optional env override; it must apply
   only to `--wall-limit-s`, and it must be recorded in the status JSON so an
   overridden run can never be mistaken for a real cell). Paste:
   - the PREFLIGHT output JSON;
   - the last 5 lines of the CG `*.iters.csv`;
   - the CG status JSON's `route_weight`, `min_reduced_cost`, `stop_reason`,
     `n_columns`.
4. `bash scripts/ladder_lite/submit.sh CG --scales 2,3,5 --dry-run` output.
5. `pytest tests/test_ladder_lite.py -q` output.
6. Line counts: `wc -l scripts/ladder_lite/*.sh` and the diff stat for
   `src/summarize_scale_ladder.py` if you touched it.

If item 3 does not produce a real CG iteration on your machine, say so plainly
and stop. Do not proceed to polish anything else.

---

## Part 3 — Operator bash (macbook1 → Unicorn)

Run these in order. Each is one paste. None uses `set -e`, so none can kill your
login shell. Everything in Blocks 0 and 1 is read-only except the one explicitly
named recovery call.

### Block 0 — recon (read-only, run now, before Cursor finishes)

I need these facts to size the arrays. Paste the whole block and send me the
output.

```bash
bash <<'BASH'
main() {
  echo "===== identity ====="
  id -un; hostname; date -u +%FT%TZ

  echo; echo "===== partitions ====="
  sinfo -o '%20P %6a %12l %6D %8t %10z %12m %N' 2>/dev/null | head -30

  echo; echo "===== partition maxtime / limits ====="
  scontrol show partition 2>/dev/null | grep -E 'PartitionName|MaxTime|DefaultTime|TotalCPUs|TotalNodes|MaxMemPerNode|State='

  echo; echo "===== scheduler config ====="
  scontrol show config 2>/dev/null | grep -iE 'MaxArraySize|MaxJobCount|DefMemPerCPU|MaxMemPerCPU|SchedulerType|SelectType'

  echo; echo "===== my association limits ====="
  sacctmgr -nP show assoc where user="$(id -un)" \
    format=Partition,QOS,MaxJobs,MaxSubmit,GrpTRES,MaxTRESPerJob 2>/dev/null | head -10

  echo; echo "===== my live jobs ====="
  squeue -u "$(id -un)" -o '%.12i %.24j %.16P %.2t %.10M %.10L %.6D %R' 2>/dev/null

  echo; echo "===== my last 3 days ====="
  sacct -u "$(id -un)" -S now-3days -X -P \
    -o JobID,JobName%34,Partition,State,Elapsed,MaxRSS,ExitCode 2>/dev/null | tail -30

  echo; echo "===== gate 250838 ====="
  scontrol show job 250838 2>/dev/null | head -14 || echo "gate no longer in scheduler"

  echo; echo "===== approved python from the frozen plan ====="
  PLAN="$HOME/EVSP-DR-scale-ladder-7937c22fef77/src/results/scale_ladder/slad_flat_primary_v4_7937c22/approved-plan.json"
  if [ -s "$PLAN" ]; then
    python3 - "$PLAN" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print("plan python :", p["python"]["path"])
print("groups      :", {k:len(v) for k,v in sorted(p["task_groups"].items())})
print("total tasks :", sum(len(v) for v in p["task_groups"].values()))
PY
  else
    echo "frozen plan not found at $PLAN"
  fi

  echo; echo "===== candidate pythons ====="
  for p in /usr/bin/python3 "$HOME/.evspdr-envs/py312/bin/python3" \
           /share/apps/software/python*/bin/python3 \
           /share/apps/software/*/*/bin/python3.12; do
    [ -x "$p" ] && printf '%-60s %s\n' "$p" "$("$p" -V 2>&1)"
  done

  echo; echo "===== gurobi ====="
  ls -l /share/apps/software/gurobi/gurobi.lic 2>/dev/null \
    || echo "NO gurobi.lic at the path the reviewed worker expects"

  echo; echo "===== existing checkouts and space ====="
  ls -d "$HOME"/EVSP-DR* "$HOME"/ladder-lite 2>/dev/null
  df -h "$HOME" | tail -2
  quota -s 2>/dev/null | tail -4

  echo; echo "===== activation failure log (retain this) ====="
  LOGS="$HOME/EVSP-DR-scale-ladder-7937c22fef77/src/results/scale_ladder/slad_flat_primary_v4_7937c22/logs"
  tail -n 40 "$LOGS/activation_a1_250113.err" 2>/dev/null || echo "no .err"
  tail -n 20 "$LOGS/activation_a1_250113.out" 2>/dev/null || echo "no .out"
}
main
BASH
```

### Block 1 — Track A: one shot at the frozen `7937c22` campaign

Run this **once**. If the `submitted_arrays` object at the end does not contain
six entries, stop. Do not debug, do not resubmit, do not cancel anything. Send
me the output and move on to Block 2.

```bash
bash <<'BASH'
main() {
  RUN_ROOT="$HOME/EVSP-DR-scale-ladder-7937c22fef77"
  CR="$RUN_ROOT/src/results/scale_ladder/slad_flat_primary_v4_7937c22"
  PLAN="$CR/approved-plan.json"; MANIFEST="$CR/campaign.json"
  COMMIT=7937c22fef7771e2f74dd03569ea852cbd805e1c

  [ -s "$PLAN" ] && [ -s "$MANIFEST" ] || { echo "SKIP: plan/manifest missing"; return 1; }
  [ "$(git -C "$RUN_ROOT" rev-parse HEAD 2>/dev/null)" = "$COMMIT" ] \
    || { echo "SKIP: checkout is not $COMMIT"; return 1; }

  PLAN_SHA=$(sha256sum "$PLAN" | awk '{print $1}')
  PYTHON=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["python"]["path"])' "$PLAN")
  echo "plan_sha=$PLAN_SHA"; echo "python=$PYTHON"

  echo; echo "--- manifest before ---"
  python3 -c 'import json,sys;m=json.load(open(sys.argv[1]));print(json.dumps({k:m.get(k) for k in ("submission_state","probe_state","gate_state","gate_job_id","submitted_arrays")},indent=2))' "$MANIFEST"

  echo; echo "--- reconcile (submit six arrays, request gate release) ---"
  env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
      PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py" "$COMMIT" \
    reconcile_scale_ladder_gate.py \
    --campaign-root "$CR" --approved-plan-sha256 "$PLAN_SHA" \
    --resume-missing-arrays --release-held-gate
  echo "reconcile_rc=$?"

  echo; echo "--- manifest after ---"
  python3 -c 'import json,sys;m=json.load(open(sys.argv[1]));a=m.get("submitted_arrays") or {};print(json.dumps({k:m.get(k) for k in ("submission_state","gate_state","gate_job_id","release_error")},indent=2));print("ARRAY_COUNT=%d"%len(a));print(json.dumps(a,indent=2))' "$MANIFEST"

  GATE=$(python3 -c 'import json,sys;print((json.load(open(sys.argv[1])).get("gate_job_id") or ""))' "$MANIFEST")
  [ -n "$GATE" ] && { echo; echo "--- gate $GATE ---"; scontrol show job "$GATE" 2>/dev/null | head -8; }

  IDS=$(python3 -c 'import json,sys;a=json.load(open(sys.argv[1])).get("submitted_arrays") or {};print(",".join(str(v) for v in a.values()))' "$MANIFEST")
  [ -n "$IDS" ] && { echo; echo "--- live arrays ---"; squeue -j "$IDS" -o '%.14i %.24j %.2t %.10M %R' 2>/dev/null; }
}
main
BASH
```

If — and only if — the block reports six arrays but the gate is still
`JobHeldUser`, run this single line:

```bash
scontrol release 250838; sleep 3; squeue -j 250838 -o '%.12i %.20j %.2t %R'
```

Then stop touching Track A either way.

### Block 2 — Track B setup (after Cursor pushes `cursor/ladder-lite-20260819`)

Creates a fresh detached checkout in a new root. Does not touch any existing
`EVSP-DR*` directory.

```bash
bash <<'BASH'
main() {
  BR=cursor/ladder-lite-20260819-2969
  ROOT="$HOME/ladder-lite/repo"
  mkdir -p "$HOME/ladder-lite"
  if [ -d "$ROOT/.git" ]; then
    git -C "$ROOT" fetch --prune origin "$BR" || { echo "fetch failed"; return 1; }
  else
    git clone --no-checkout https://github.com/ndandnd/EVSP-DR.git "$ROOT" || return 1
    git -C "$ROOT" fetch origin "$BR" || return 1
  fi
  git -C "$ROOT" checkout --detach FETCH_HEAD || { echo "checkout failed"; return 1; }
  echo "HEAD: $(git -C "$ROOT" rev-parse HEAD)"
  echo "detached: $(git -C "$ROOT" symbolic-ref -q HEAD >/dev/null 2>&1 && echo NO || echo YES)"
  echo "dirty tracked: $(git -C "$ROOT" status --porcelain --untracked-files=no | wc -l)"
  ls -1 "$ROOT/scripts/ladder_lite/"
}
main
BASH
```

Then build the plan (one command; substitute the Python 3.12 path Block 0
reports):

```bash
LL_ROOT=$HOME/ladder-lite LL_PYTHON=<py312-path-from-block-0> \
  bash $HOME/ladder-lite/repo/scripts/ladder_lite/plan.sh
```

### Block 3 — Track B submission ladder

Run these one at a time, in order, checking `status.sh` between steps. `LL` is
just a shorthand.

```bash
LL=$HOME/ladder-lite/repo/scripts/ladder_lite
```

```bash
# 1. cheap groups first — these must produce files within minutes
bash $LL/submit.sh PREFLIGHT
bash $LL/submit.sh SEED
bash $LL/status.sh
```

**Gate: do not run step 2 until `status.sh` shows PREFLIGHT `done` > 0.** That
is the 10-minute success criterion. If PREFLIGHT fails, send me one `.failed`
marker and the matching `.err` log; that is a real diagnostic and I can act on it.

```bash
# 2. the small CG rungs — nine 2-hour cells, results tonight
bash $LL/submit.sh CG --scales 2,3,5
bash $LL/status.sh
```

```bash
# 3. once step 2 has at least one .done, launch everything else that is cheap
bash $LL/submit.sh CG_SENSITIVITY
bash $LL/submit.sh CG --scales 8,13
bash $LL/status.sh
```

```bash
# 4. the long rungs — start these before you go to bed
bash $LL/submit.sh CG --scales 20,30,40
bash $LL/status.sh
```

```bash
# 5. MIPs, per scale, only after that scale's CG cells show .done
bash $LL/submit.sh MIP_RAW   --scales 2,3,5,8
bash $LL/submit.sh MIP_KNOWN --scales 2,3,5,8
bash $LL/status.sh
```

Repeat step 5 with `--scales 13`, then `20`, then `30` as the CG rungs land.
Cells whose CG is not ready write a `.blocked` marker and exit cleanly, so
resubmitting a group is always safe.

```bash
# 6. normalize whatever exists — run this even when cells are still missing
bash $LL/normalize.sh
ls -l $HOME/ladder-lite/normalized/
head -3 $HOME/ladder-lite/normalized/cg_run_summary.csv
```

Send me `cg_run_summary.csv`, `scale_progress_summary.csv`, and `status.sh`
output. Missing cells stay explicit `missing`/`censored` rows — that is correct
and expected; do not wait for a complete matrix before normalizing.

---

## Part 4 — Standing rules for this project

1. **Report executed output, never readiness.** The reportable states are:
   *scientific tasks submitted → scientific tasks running → outputs on disk →
   normalized CSVs*. "Prepared", "packaged", "armed", "tests pass" are not
   states.
2. **One scientific worker before any infrastructure work.** If a launch path
   cannot produce one k2 CG iteration, no amount of hardening around it matters.
3. **Fail open on liveness, fail closed on interpretation.** Never block a run
   because a label might be wrong. Run it, label it honestly
   (`legacy_scheduler_unverified`, `ladder_lite_direct_array`,
   `combined-cost-master route weight`), and let the normalizer carry the caveat.
4. **`route_weight` is `combined-cost-master route weight`,** not a fleet LP
   lower bound, until the three-phase lexicographic master exists. Every plot
   axis and table header says so.
5. **RAW and KNOWN never share a row.** KNOWN is a plumbing positive control,
   not algorithmic recovery.
6. **Cursor gets a line budget and a forbidden list in every work order.** The
   failure mode is not incompetence, it is unbounded scope.
