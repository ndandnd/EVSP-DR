#!/bin/bash
set -uo pipefail

main() {
  [ "$#" -eq 2 ] || { echo "usage: run_cell.sh PLAN_JSON GROUP" >&2; return 2; }
  PLAN=$1; GROUP=$2
  LL_ROOT=${LL_ROOT:-"$HOME/ladder-lite"}
  PYTHON=${LL_PYTHON:-/home/nc437/evsp_env/bin/python3.12}
  unset PYTHONPATH PYTHONHOME PYTHONSTARTUP LD_LIBRARY_PATH
  export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
  TASK=${SLURM_ARRAY_TASK_ID:-}
  [[ "$TASK" =~ ^[0-9]+$ ]] || { echo "SLURM_ARRAY_TASK_ID missing" >&2; return 2; }
  readarray -t F < <("$PYTHON" -B - "$PLAN" "$GROUP" "$TASK" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); group=sys.argv[2]; i=int(sys.argv[3])
keys=p["task_groups"].get(group)
if not isinstance(keys,list) or not 0<=i<len(keys): raise SystemExit("invalid group/task")
m=[j for j in p["jobs"] if j["job_key"]==keys[i]]
if len(m)!=1: raise SystemExit("task does not map to exactly one job")
j=m[0]; by={x["job_key"]:x for x in p["jobs"]}
def out(name): return (by[j[name]]["output"] if j.get(name) else "")
for v in (p["checkout_identity"]["commit"],j["job_key"],j["phase"],j["arm"],
 j["scale"],j["selection_replicate"],j["cg_replicate"],j["budget_s"],j["threads"],
 j["instance"]["path"],j["instance"]["instance_file_sha256"],
 j["instance"]["relative_path"],j["output"],j["progress_dir"],j["telemetry"],
 j["soc_step"],j["block_min"],",".join(map(str,j["snapshot_minutes"])),
 j.get("g_kwh",300),j.get("charge_kw",300),j.get("min_soc_frac",0),
 j.get("columns_per_iter",30),j.get("max_iters",2000),
 j.get("diversify_rounds",0),j.get("initial_pool","singletons"),
 j.get("objective","combined-cost"),j.get("master_sense","partition"),
 j.get("checkpoint_every",25),
 out("dependency_preflight"),out("dependency_cg"),out("dependency_seed")):
 print("" if v is None else v)
PY
  ) || return 2
  COMMIT=${F[0]}; JOB_KEY=${F[1]}; PHASE=${F[2]}; ARM=${F[3]}
  SCALE=${F[4]}; SEL=${F[5]}; CGREP=${F[6]}; BUDGET=${F[7]}; THREADS=${F[8]}
  INSTANCE=${F[9]}; INSTANCE_SHA=${F[10]}; INSTANCE_REL=${F[11]}
  OUT=${F[12]}; PROGRESS=${F[13]}; TELEMETRY=${F[14]}
  SOC=${F[15]}; BLOCK=${F[16]}; SNAPSHOTS=${F[17]}
  G_KWH=${F[18]}; CHARGE_KW=${F[19]}; MIN_SOC=${F[20]}
  COLUMNS_PER_ITER=${F[21]}; MAX_ITERS=${F[22]}; DIVERSIFY=${F[23]}
  INITIAL_POOL=${F[24]}; OBJECTIVE=${F[25]}; MASTER_SENSE=${F[26]}
  CHECKPOINT_EVERY=${F[27]}
  PREFLIGHT=${F[28]}; CG_OUT=${F[29]}; SEED_OUT=${F[30]}
  MARKER="$OUT.done"
  record_failure() { rc=$?; trap - EXIT; if [ "$rc" -ne 0 ] && [ ! -s "$OUT.failed" ]; then mkdir -p "$(dirname "$OUT")" 2>/dev/null || true; printf 'exit_code=%s\njob_key=%s\nslurm_job_id=%s\nnode=%s\n' "$rc" "$JOB_KEY" "${SLURM_JOB_ID:-local}" "${SLURMD_NODENAME:-$(hostname)}" >"$OUT.failed" 2>/dev/null || true; fi; exit "$rc"; }; trap record_failure EXIT
  mkdir -p "$(dirname "$OUT")" || return 1
  rm -f "$OUT.failed" "$OUT.blocked"
  REPO=${LL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}; echo "[ll] host=$(hostname) task=$TASK repo=$REPO"
  [ -f "$REPO/src/exact_pricer_expanded.py" ] || { echo "unresolved repository root: missing $REPO/src/exact_pricer_expanded.py" >&2; return 2; }
  if [ -n "${LL_BUDGET_OVERRIDE_S:-}" ] && [[ "$PHASE" == CG* || "$PHASE" == MIP ]]; then
    MARKER="$OUT.smoke.done"
  fi
  if [ -e "$OUT.done" ] || [ -e "$MARKER" ]; then echo "SKIP $JOB_KEY"; return 0; fi
  EFFECTIVE=${LL_BUDGET_OVERRIDE_S:-$BUDGET}
  CG_LIMIT=$((BUDGET + 60))
  [ -z "${LL_BUDGET_OVERRIDE_S:-}" ] || CG_LIMIT=$EFFECTIVE
  [[ "$EFFECTIVE" =~ ^[0-9]+$ ]] || { echo "invalid budget override" >&2; return 2; }
  command=()
  case "$PHASE" in
    PREFLIGHT) command=("$PYTHON" -B "$REPO/src/audit_scale_ladder_known_membership.py"
      --instance "$REPO/data/$INSTANCE_REL" --instance-sha256 "$INSTANCE_SHA"
      --scale "$SCALE" --selection-replicate "$SEL" --out "$OUT"
      --csv-out "${OUT%.json}.csv") ;;
    SEED) command=("$PYTHON" -B "$REPO/src/prepare_scale_ladder_known_partition.py"
      --instance "$REPO/data/$INSTANCE_REL" --instance-sha256 "$INSTANCE_SHA"
      --out "$OUT") ;;
    CG|CG_SENSITIVITY)
      command=("$PYTHON" -u "$REPO/src/exact_pricer_expanded.py"
        --csv "$INSTANCE_REL" --prices_csv hourly_prices_flat.csv
        --g-kwh "$G_KWH" --charge-kw "$CHARGE_KW" --min-soc-frac "$MIN_SOC"
        --soc-step "$SOC" --block-min "$BLOCK" --master-sense "$MASTER_SENSE"
        --initial-pool "$INITIAL_POOL" --objective "$OBJECTIVE"
        --columns_per_iter "$COLUMNS_PER_ITER" --max-iters "$MAX_ITERS"
        --diversify-rounds "$DIVERSIFY" --wall-limit-s "$CG_LIMIT")
      if [ "$OBJECTIVE" = "combined-cost" ]; then
        command+=(--checkpoint-every "$CHECKPOINT_EVERY" --resume
          --snapshot-at-minutes "$SNAPSHOTS" --out "$OUT")
        [ -z "$TELEMETRY" ] || command+=(--phase-telemetry "$TELEMETRY")
      elif [ "$OBJECTIVE" = "lexicographic-fleet" ]; then
        # The three-phase runner is deliberately immutable: it does not resume
        # and writes a phase-specific iteration journal.  Snapshot and phase
        # telemetry options belong only to the resumable combined-cost path.
        command+=(--out "$OUT")
      else
        echo "unsupported plan objective: $OBJECTIVE" >&2; return 2
      fi ;;
    MIP)
      command=("$PYTHON" -u "$REPO/src/run_exact_pool_mip.py" --result "$CG_OUT"
        --two-stage --threads "$THREADS" --timelimit "$EFFECTIVE" --mipgap 0.0001
        --progress-dir "$PROGRESS" --out "$OUT")
      [ "$ARM" != "KNOWN-PARTITION" ] || command+=(--initial-partition-routes "$SEED_OUT") ;;
    *) echo "unknown phase: $PHASE" >&2; return 2 ;;
  esac
  if [ "${LL_PRINT_COMMAND:-0}" = 1 ]; then printf '%q ' "${command[@]}"; echo; return 0; fi
  OBSERVED=$(git -C "$REPO" rev-parse HEAD) || return 2
  [ "$OBSERVED" = "$COMMIT" ] || { echo "commit mismatch" >&2; return 2; }
  [ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || {
    echo "tracked checkout modifications" >&2; return 2;
  }
  "$PYTHON" -B "$REPO/src/install_exact_cg_profile_input.py" \
    --source "$INSTANCE" --data-root "$REPO/data" --relative "$INSTANCE_REL" \
    --sha256 "$INSTANCE_SHA" || return 2
  export EVSP_EXPECTED_COMMIT="$OBSERVED"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  if [ "$PHASE" = MIP ]; then
    if [ ! -s "$CG_OUT" ] || [ ! -e "$CG_OUT.done" ]; then
      printf 'dependent CG not ready: %s\n' "$CG_OUT" >"$OUT.blocked"; return 0
    fi
    if [ "$ARM" = "KNOWN-PARTITION" ] && [ ! -s "$SEED_OUT" ]; then
      printf 'known partition not ready: %s\n' "$SEED_OUT" >"$OUT.blocked"; return 0
    fi
    if [ -d "$PROGRESS" ]; then
      if [ -s "$PROGRESS/result_pending.json" ]; then
        "$PYTHON" -B "$REPO/src/recover_scale_ladder_mip_progress.py" \
          --progress-dir "$PROGRESS" >/dev/null 2>&1
        if [ -s "$PROGRESS/final.json" ]; then
          [ -e "$OUT" ] || ln "$PROGRESS/result_pending.json" "$OUT" 2>/dev/null
          if cmp -s "$PROGRESS/result_pending.json" "$OUT"; then
            touch "$MARKER"; echo "RECOVERED $JOB_KEY"; return 0
          fi
        fi
      fi
      UTC=$(date -u +%Y%m%dT%H%M%SZ)
      QUARANTINE="$PROGRESS.quarantine-${SLURM_JOB_ID:-local}-$UTC"
      mv "$PROGRESS" "$QUARANTINE" || return 2
      printf '%s\t%s\t%s\t%s\n' "$UTC" "$JOB_KEY" "$PROGRESS" "$QUARANTINE" \
        >>"$LL_ROOT/quarantine.tsv"
    fi
    JOURNAL=$("$PYTHON" -B -c \
      'import json,pathlib,sys;p=pathlib.Path(sys.argv[1]);q=pathlib.Path(json.load(p.open())["columns_journal"]);print((q if q.is_absolute() else p.resolve().parent/q).resolve())' \
      "$CG_OUT") || return 2
    [ -s "$JOURNAL" ] || { echo "dependent CG journal missing" >&2; return 2; }
    EVSP_MIP_EXPECTED_RESULT_SHA256=$(sha256sum "$CG_OUT" | awk '{print $1}') || return 2
    EVSP_MIP_EXPECTED_JOURNAL_SHA256=$(sha256sum "$JOURNAL" | awk '{print $1}') || return 2
    export EVSP_MIP_EXPECTED_RESULT_SHA256 EVSP_MIP_EXPECTED_JOURNAL_SHA256
    if [ "$ARM" = "KNOWN-PARTITION" ]; then
      EVSP_MIP_EXPECTED_INITIAL_PARTITION_SHA256=$(sha256sum "$SEED_OUT" | awk '{print $1}') || return 2
      export EVSP_MIP_EXPECTED_INITIAL_PARTITION_SHA256
    else
      unset EVSP_MIP_EXPECTED_INITIAL_PARTITION_SHA256 || true
    fi
    export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
    unset LM_LICENSE_FILE || true
    export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
    export MKL_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
  elif [ -n "$PREFLIGHT" ] && [ ! -s "$PREFLIGHT" ]; then
    printf 'membership preflight not ready: %s\n' "$PREFLIGHT" >"$OUT.blocked"; return 0
  fi
  if [ "$EFFECTIVE" != "$BUDGET" ] && [[ "$PHASE" == CG* || "$PHASE" == MIP ]]; then
    printf '{"job_key":"%s","plan_budget_s":%s,"effective_budget_s":%s,"label":"budget_overridden"}\n' \
      "$JOB_KEY" "$BUDGET" "$EFFECTIVE" >"$OUT.override.json"
  fi
  ERR="$OUT.stderr"; child=""
  forward() { [ -z "$child" ] || kill -USR1 "$child" 2>/dev/null || true; }
  trap forward USR1 TERM INT
  (cd "$REPO" && "${command[@]}") 2>"$ERR" & child=$!
  status=0
  while true; do
    wait_status=0; wait "$child" || wait_status=$?
    if kill -0 "$child" 2>/dev/null; then continue; fi
    status=$wait_status; break
  done
  child=""; trap - USR1 TERM INT
  if [ "$status" -eq 0 ]; then
    touch "$MARKER"
    echo "DONE $JOB_KEY scale=$SCALE selection=$SEL cg=$CGREP"; return 0
  fi
  {
    echo "exit_code=$status"; echo "job_key=$JOB_KEY"
    echo "slurm_job_id=${SLURM_JOB_ID:-local}"; echo "node=${SLURMD_NODENAME:-$(hostname)}"
    tail -n 40 "$ERR" 2>/dev/null
  } >"$OUT.failed"
  return "$status"
}
main "$@"
