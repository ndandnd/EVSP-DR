#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 1 ]] || evsp_die "usage: $0 [CAMPAIGN_ROOT]"
REPO=$(evsp_repo_root)
BRANCH=$(git -C "$REPO" branch --show-current)
WRAPPER_COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
SOLVER_COMMIT="$WRAPPER_COMMIT"
INPUT_COMMIT="$WRAPPER_COMMIT"
ROOT="${1:-$HOME/ladder-lite/threshold_9_15_event_20260904_${WRAPPER_COMMIT:0:7}}"

if squeue --me -h -o '%j' | grep -qE '^th04(net|cg)$'; then
  evsp_die "k9--k15 threshold campaign already active"
fi
[[ ! -e "$ROOT" ]] || evsp_die "campaign root already exists: $ROOT"

SOLVER_REPO=$(evsp_execution_checkout "$REPO" "$SOLVER_COMMIT")
INPUT_REPO="$SOLVER_REPO"
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
"$PYTHON_BIN" \
  "$INPUT_REPO/scripts/event_uniform_envelope/validate_small_threshold_inputs.py" \
  --repo "$INPUT_REPO" \
  --campaign threshold_9_15_20260904 \
  --schema evsp-dr-threshold-9-15-inputs-v1 \
  --seed 20260904 --scales 9 10 11 12 13 14 15 \
  --generator-script \
    scripts/event_uniform_envelope/build_threshold_9_15_inputs.py
HELP_TEXT=$("$PYTHON_BIN" "$SOLVER_REPO/src/exact_pricer_expanded.py" --help)
grep -q -- '--event-network-cache' <<< "$HELP_TEXT" \
  || evsp_die "solver lacks event-network caching"
grep -q -- '--column-selection' <<< "$HELP_TEXT" \
  || evsp_die "solver lacks controlled column selection"

"$PYTHON_BIN" \
  "$INPUT_REPO/scripts/event_uniform_envelope/prepare_threshold_9_15_event.py" \
  --input-repo "$INPUT_REPO" --solver-repo "$SOLVER_REPO" \
  --root "$ROOT" --input-commit "$INPUT_COMMIT" \
  --solver-commit "$SOLVER_COMMIT" --wrapper-commit "$WRAPPER_COMMIT"

COMMON="EVSP_EXECUTION_REPO=$SOLVER_REPO,EVSP_EXPECTED_COMMIT=$SOLVER_COMMIT,EVSP_CAMPAIGN_ROOT=$ROOT,EVSP_PYTHON=$PYTHON_BIN"
RECORD="$ROOT/jobs.tsv"
printf 'stage\tarm\tarray_job_id\tindices\tpartition\tmem\ttimelimit\tdependency\tcolumns_per_iter\tselection\tdiversity_weight\twrapper_commit\tsolver_commit\tinput_commit\n' > "$RECORD"

# Network construction is measured separately and gets a 24-hour Slurm
# envelope.  It has no internal checkpoint, so preemption restarts that task.
CACHE_JOB=$(evsp_submit_and_resolve th04net \
  --array='0-69%24' -p default_partition -c 1 --mem=96G \
  -t 24:30:00 --requeue --open-mode=append --signal=B:TERM@180 \
  --export="ALL,$COMMON" \
  -o "$ROOT/logs/cache/%x_%A_%a.out" \
  -e "$ROOT/logs/cache/%x_%A_%a.err" \
  "$SCRIPT_DIR/event_network_cache.sub")
printf 'cache\tNA\t%s\t0-69\tdefault_partition\t96G\t24:30:00\tNA\tNA\tNA\tNA\t%s\t%s\t%s\n' \
  "$CACHE_JOB" "$WRAPPER_COMMIT" "$SOLVER_COMMIT" "$INPUT_COMMIT" \
  >> "$RECORD"

# aftercorr pairs array index i with cache index i.  CG checkpoints every 25
# iterations and resumes accumulated columns after a Slurm requeue.
EXPORTS="$COMMON,EVSP_ARM=b030_reduced,EVSP_COLUMNS_PER_ITER=30,EVSP_COLUMN_SELECTION=reduced_cost,EVSP_COLUMN_DIVERSITY_WEIGHT=0.0"
CG_JOB=$(evsp_submit_and_resolve th04cg \
  --array='0-69%30' -p default_partition -c 1 --mem=96G \
  -t 12:15:00 --requeue --open-mode=append --signal=B:TERM@180 \
  --dependency="aftercorr:$CACHE_JOB" --export="ALL,$EXPORTS" \
  -o "$ROOT/logs/cg/%x_%A_%a.out" \
  -e "$ROOT/logs/cg/%x_%A_%a.err" \
  "$SCRIPT_DIR/cg_acceleration.sub")
printf 'cg\tb030_reduced\t%s\t0-69\tdefault_partition\t96G\t12:15:00\taftercorr:%s\t30\treduced_cost\t0.0\t%s\t%s\t%s\n' \
  "$CG_JOB" "$CACHE_JOB" "$WRAPPER_COMMIT" "$SOLVER_COMMIT" \
  "$INPUT_COMMIT" >> "$RECORD"

sha256sum "$ROOT/matrix.tsv" "$ROOT/input_selection_manifest.csv" \
  "$ROOT/execution_plan.json" "$RECORD" > "$ROOT/SUBMISSION_SHA256SUMS"
echo "k9--k15 threshold campaign: $ROOT"
echo "network-cache array: $CACHE_JOB (70 tasks, up to 24 concurrent)"
echo "baseline exact-CG array: $CG_JOB (70 paired tasks, up to 30 concurrent)"
echo "Machine-readable jobs: $RECORD"
echo "Audit after completion:"
echo "bash $SCRIPT_DIR/audit_cg_acceleration.sh $ROOT"
