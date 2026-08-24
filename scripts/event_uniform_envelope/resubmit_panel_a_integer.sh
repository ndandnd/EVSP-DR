#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# == 1 ]] || evsp_die "usage: $0 PANEL_A_ROOT"

ROOT=$(cd "$1" && pwd)
REPO=$(evsp_repo_root)
BRANCH=$(git -C "$REPO" branch --show-current)
[[ -n "$BRANCH" ]] || evsp_die "manager checkout must be on a named branch"
WRAPPER_COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
[[ -f "$ROOT/execution_provenance.json" ]] \
  || evsp_die "missing Panel A execution provenance"
COMMIT=$(
  "${EVSP_PYTHON:-$HOME/evsp_env/bin/python}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["checkout_commit"])' \
    "$ROOT/execution_provenance.json"
)
EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
MANIFEST="$ROOT/panel_a_integer_inputs.tsv"

if squeue --me -h -o '%j' | grep -qE '^eua24_(mipr|tfr)$'; then
  evsp_die "Panel A recovery jobs already exist"
fi
[[ ! -s "$ROOT/mip/A__k02_s1__event_2p5_event5.json" ]] \
  || evsp_die "Panel A MIP outputs already exist; audit before resubmitting"

"$PYTHON_BIN" "$SCRIPT_DIR/prepare_integer_manifest.py" \
  --root "$ROOT" --panel A --source-dir cg --out "$MANIFEST" \
  --provenance "$ROOT/integer_recovery_provenance.json" \
  --wrapper-commit "$WRAPPER_COMMIT" --solver-commit "$COMMIT"
sha256sum "$MANIFEST" > "$ROOT/panel_a_integer_inputs.sha256"

EXPORTS="ALL,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_CAMPAIGN_ROOT=$ROOT,EVSP_EXPECTED_COMMIT=$COMMIT,EVSP_PANEL=A,EVSP_INTEGER_MANIFEST=$MANIFEST,EVSP_MIP_OUTPUT_DIR=$ROOT/mip,EVSP_TARGET_OUTPUT_DIR=$ROOT/target,EVSP_PYTHON=$PYTHON_BIN"

MIP_JOB=$(evsp_submit_and_resolve eua24_mipr \
  --array=0-53%54 -p default_partition -c 8 --mem=24G -t 00:45:00 \
  --no-requeue --export="$EXPORTS" \
  -o "$ROOT/logs/mipr_%A_%a.out" -e "$ROOT/logs/mipr_%A_%a.err" \
  "$SCRIPT_DIR/pool_mip.sub")
sleep 1
TF_JOB=$(evsp_submit_and_resolve eua24_tfr \
  --array=0-53%54 -p default_partition -c 8 --mem=24G -t 00:45:00 \
  --no-requeue --export="$EXPORTS" \
  -o "$ROOT/logs/tfr_%A_%a.out" -e "$ROOT/logs/tfr_%A_%a.err" \
  "$SCRIPT_DIR/target_feasibility.sub")

{
  echo -e "stage\tarray_job_id\ttasks"
  echo -e "mip_recovery\t$MIP_JOB\t54"
  echo -e "target_recovery\t$TF_JOB\t54"
} | tee "$ROOT/integer_recovery_jobs.tsv"
