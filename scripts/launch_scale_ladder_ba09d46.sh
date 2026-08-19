#!/bin/bash

main() {
  COMMIT="ba09d4602ded98f9c9157f52af169ef511b5abf7"
  RUN_ROOT="$HOME/EVSP-DR-scale-ladder-ba09d46"
  PYTHON=$(readlink -f "$HOME/evsp_env/bin/python3.12" 2>/dev/null)
  STAMP=$(date -u +%Y%m%d_%H%M%S)
  CAMPAIGN="slad_${STAMP}_ba09d46"
  PLAN_ROOT="$HOME/evsp_scale_ladder_plans"
  RESERVATIONS="$HOME/evsp_scale_ladder_reservations"
  PLAN="$PLAN_ROOT/$CAMPAIGN.plan.json"
  MATRIX="$PLAN_ROOT/$CAMPAIGN.tasks.csv"
  DRY_OUT="$PLAN_ROOT/$CAMPAIGN.dryrun.out"
  DRY_ERR="$PLAN_ROOT/$CAMPAIGN.dryrun.err"
  SUBMIT_OUT="$PLAN_ROOT/$CAMPAIGN.submit.out"
  SUBMIT_ERR="$PLAN_ROOT/$CAMPAIGN.submit.err"

  echo "=== Prepare exact reviewed checkout ==="
  if [ ! -x "$PYTHON" ]; then
    echo "Approved Python 3.12 is unavailable; nothing submitted."
    return 1
  fi
  if [ ! -d "$RUN_ROOT/.git" ]; then
    git clone https://github.com/ndandnd/EVSP-DR.git "$RUN_ROOT" || return 1
  fi
  if [ -n "$(git -C "$RUN_ROOT" status --porcelain --untracked-files=all)" ]; then
    echo "Run checkout is dirty; nothing submitted: $RUN_ROOT"
    return 1
  fi
  git -C "$RUN_ROOT" fetch origin "$COMMIT" || return 1
  git -C "$RUN_ROOT" checkout --detach "$COMMIT" || return 1
  if [ "$(git -C "$RUN_ROOT" rev-parse HEAD)" != "$COMMIT" ] ||
     [ -n "$(git -C "$RUN_ROOT" status --porcelain --untracked-files=all)" ]; then
    echo "Exact detached checkout verification failed; nothing submitted."
    return 1
  fi

  echo "=== Generate and machine-check the approved 138-task contract ==="
  mkdir -p "$PLAN_ROOT" "$RESERVATIONS" || return 1
  env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py" \
    "$COMMIT" launch_scale_ladder.py \
    --campaign "$CAMPAIGN" --python "$PYTHON" \
    --reservation-root "$RESERVATIONS" \
    --plan-out "$PLAN" --matrix-out "$MATRIX" \
    >"$DRY_OUT" 2>"$DRY_ERR"
  DRY_STATUS=$?
  if [ "$DRY_STATUS" -ne 0 ] || [ ! -s "$PLAN" ] || [ ! -s "$MATRIX" ]; then
    echo "Dry run failed; nothing submitted."
    tail -n 100 "$DRY_OUT" "$DRY_ERR" 2>/dev/null
    return 1
  fi
  if ! jq -e --arg commit "$COMMIT" '
      .checkout_identity.commit == $commit and
      .checkout_identity.detached == true and
      .checkout_identity.tracked_clean == true and
      .task_count == 138 and
      .infrastructure_probe_task_count == 2 and
      .k40_mip_submission_count == 0 and
      ([.jobs[] | select(.scale == 40 and .phase == "MIP")] | length) == 0 and
      (.task_groups.PREFLIGHT | length) == 22 and
      (.task_groups.SEED | length) == 21 and
      (.task_groups.CG | length) == 23 and
      (.task_groups.CG_SENSITIVITY | length) == 30 and
      (.task_groups.MIP_RAW | length) == 21 and
      (.task_groups.MIP_KNOWN | length) == 21 and
      .physics.g_kwh == 300.0 and
      .physics.charge_kw == 300.0 and
      .physics.reserve_kwh == 0.0 and
      .physics.soc_step_kwh == 15.0 and
      .physics.block_min == 10 and
      .tariff.primary_tariff_sha256 ==
        "1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200" and
      (.python_identity.portable_identity_sha256 | type) == "string"
    ' "$PLAN" >/dev/null; then
    echo "Scientific plan contract mismatch; nothing submitted."
    return 1
  fi
  PLAN_SHA=$(sha256sum "$PLAN" | awk '{print $1}')
  echo "CAMPAIGN=$CAMPAIGN"
  echo "PLAN_SHA256=$PLAN_SHA"
  jq -r '
    ["group","tasks"],
    (.task_groups | to_entries[] | [.key, (.value | length)]) | @tsv
  ' "$PLAN"

  echo "=== Submit: six held arrays; exactly two official probes control release ==="
  env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py" \
    "$COMMIT" launch_scale_ladder.py \
    --campaign "$CAMPAIGN" --python "$PYTHON" \
    --reservation-root "$RESERVATIONS" \
    --approved-plan-sha256 "$PLAN_SHA" --submit \
    >"$SUBMIT_OUT" 2>"$SUBMIT_ERR"
  SUBMIT_STATUS=$?
  CAMPAIGN_ROOT=$(jq -r '.campaign_root' "$PLAN")
  MANIFEST="$CAMPAIGN_ROOT/campaign.json"
  if [ "$SUBMIT_STATUS" -ne 0 ]; then
    echo "Submission stopped fail-closed; do NOT submit this campaign again."
    if [ -s "$MANIFEST" ]; then
      jq '{probe_state,probe_results,gate_job_id,gate_state,submitted_arrays}' "$MANIFEST"
    fi
    tail -n 120 "$SUBMIT_OUT" "$SUBMIT_ERR" 2>/dev/null
    squeue --me -o '%.14i %.18j %.2t %.10M %R'
    return 1
  fi
  if ! jq -e --arg sha "$PLAN_SHA" '
      .approval_sha256 == $sha and
      .submitted == true and .gate_state == "released" and
      (.submitted_arrays | length) == 6 and
      ([.submitted_arrays[] | tostring | test("^[0-9]+$")] | all) and
      (.probe_results.default_partition.compatible == true) and
      (.probe_results.scaglione.compatible == true)
    ' "$MANIFEST" >/dev/null; then
    echo "Submission returned success but manifest validation failed."
    echo "Do not alter or resubmit this campaign."
    return 1
  fi

  echo "SCALE_LADDER_LAUNCHED=true"
  echo "CAMPAIGN_ROOT=$CAMPAIGN_ROOT"
  echo "PLAN=$PLAN"
  echo "MATRIX=$MATRIX"
  squeue --me -o '%.14i %.18j %.2t %.10M %R'
}

main
