#!/bin/bash

# Deterministic, rerunnable Unicorn entry point for the reviewed probe-first
# scale-ladder protocol.  It intentionally does not enable `set -e`, so a
# failure cannot close an interactive login shell that sourced/pasted it.

main() {
  COMMIT=${REVIEWED_COMMIT:-}
  CAMPAIGN=${LADDER_CAMPAIGN:-}
  SUBMIT_APPROVAL=${EVSP_LADDER_SUBMIT:-NO}
  RETRY_KIND=${EVSP_LADDER_RETRY:-}
  PYTHON_CANDIDATE=${EVSP_LADDER_PYTHON:-"$HOME/evsp_env/bin/python3.12"}
  PYTHON=$(readlink -f "$PYTHON_CANDIDATE" 2>/dev/null)

  if [[ ! "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Set REVIEWED_COMMIT to the exact 40-character reviewed commit." >&2
    return 1
  fi
  if [[ ! "$CAMPAIGN" =~ ^[a-z0-9][a-z0-9_-]{2,79}$ ]]; then
    echo "Set LADDER_CAMPAIGN once and reuse it on every retry." >&2
    return 1
  fi
  if [[ "$SUBMIT_APPROVAL" != "YES" && "$SUBMIT_APPROVAL" != "NO" ]]; then
    echo "EVSP_LADDER_SUBMIT must be YES or NO." >&2
    return 1
  fi
  if [[ -n "$RETRY_KIND" && \
        "$RETRY_KIND" != "failed_probes" && \
        "$RETRY_KIND" != "failed_activation" ]]; then
    echo "EVSP_LADDER_RETRY must be failed_probes or failed_activation." >&2
    return 1
  fi
  if [[ -n "$RETRY_KIND" && "$SUBMIT_APPROVAL" != "YES" ]]; then
    echo "A retry requires EVSP_LADDER_SUBMIT=YES." >&2
    return 1
  fi
  if [[ ! -x "$PYTHON" ]]; then
    echo "Approved Python 3.12 is unavailable; nothing submitted." >&2
    return 1
  fi

  SHORT_COMMIT=${COMMIT:0:12}
  RUN_ROOT=${EVSP_LADDER_RUN_ROOT:-"$HOME/EVSP-DR-scale-ladder-$SHORT_COMMIT"}
  PLAN_ROOT=${EVSP_LADDER_PLAN_ROOT:-"$HOME/evsp_scale_ladder_plans"}
  RESERVATIONS=${EVSP_LADDER_RESERVATIONS:-"$HOME/evsp_scale_ladder_reservations"}
  PLAN="$PLAN_ROOT/$CAMPAIGN.plan.json"
  MATRIX="$PLAN_ROOT/$CAMPAIGN.tasks.csv"
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  DRY_OUT="$PLAN_ROOT/$CAMPAIGN.$STAMP.dryrun.out"
  DRY_ERR="$PLAN_ROOT/$CAMPAIGN.$STAMP.dryrun.err"
  SUBMIT_OUT="$PLAN_ROOT/$CAMPAIGN.$STAMP.submit.out"
  SUBMIT_ERR="$PLAN_ROOT/$CAMPAIGN.$STAMP.submit.err"

  echo "=== exact reviewed checkout ==="
  if [[ ! -d "$RUN_ROOT/.git" ]]; then
    git clone https://github.com/ndandnd/EVSP-DR.git "$RUN_ROOT" || return 1
  fi
  if [[ -n "$(git -C "$RUN_ROOT" status --porcelain --untracked-files=all)" ]]; then
    echo "Run checkout is dirty; refusing: $RUN_ROOT" >&2
    return 1
  fi
  git -C "$RUN_ROOT" fetch origin "$COMMIT" || return 1
  git -C "$RUN_ROOT" checkout --detach "$COMMIT" || return 1
  if [[ "$(git -C "$RUN_ROOT" rev-parse HEAD)" != "$COMMIT" || \
        -n "$(git -C "$RUN_ROOT" status --porcelain --untracked-files=all)" ]]; then
    echo "Exact detached checkout verification failed." >&2
    return 1
  fi

  mkdir -p "$PLAN_ROOT" "$RESERVATIONS" || return 1
  if [[ -e "$PLAN" && ! -s "$PLAN" ]] || \
     [[ -e "$MATRIX" && ! -s "$MATRIX" ]] || \
     [[ -e "$PLAN" && ! -e "$MATRIX" ]] || \
     [[ ! -e "$PLAN" && -e "$MATRIX" ]]; then
    echo "Plan/matrix publication is incomplete; refusing to replace it." >&2
    return 1
  fi

  echo "=== dry run and immutable contract check ==="
  DRY_ARGS=(
    --campaign "$CAMPAIGN"
    --python "$PYTHON"
    --reservation-root "$RESERVATIONS"
  )
  if [[ ! -e "$PLAN" ]]; then
    DRY_ARGS+=(--plan-out "$PLAN" --matrix-out "$MATRIX")
  fi
  env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py" \
    "$COMMIT" launch_scale_ladder.py "${DRY_ARGS[@]}" \
    >"$DRY_OUT" 2>"$DRY_ERR"
  DRY_STATUS=$?
  if [[ "$DRY_STATUS" -ne 0 || ! -s "$PLAN" || ! -s "$MATRIX" ]]; then
    echo "Dry run failed; no jobs were submitted." >&2
    tail -n 100 "$DRY_OUT" "$DRY_ERR" 2>/dev/null
    return 1
  fi
  APPROVAL_LINES=$(grep -c '^\[approval-sha256\] [0-9a-f]\{64\}$' "$DRY_OUT")
  LAUNCHER_SHA=$(sed -n 's/^\[approval-sha256\] \([0-9a-f]\{64\}\)$/\1/p' "$DRY_OUT")
  FILE_SHA=$(sha256sum "$PLAN" | awk '{print $1}')
  if [[ "$APPROVAL_LINES" -ne 1 || "$LAUNCHER_SHA" != "$FILE_SHA" ]]; then
    echo "Launcher approval SHA and saved-plan SHA do not match." >&2
    return 1
  fi
  if ! jq -e --arg commit "$COMMIT" '
      .checkout_identity.commit == $commit and
      .checkout_identity.detached == true and
      .checkout_identity.tracked_clean == true and
      .submission_protocol == "probe_first_activation_v1" and
      .task_count == 138 and
      .infrastructure_probe_task_count == 2 and
      .infrastructure_activation_task_count == 1 and
      .infrastructure_task_count == 3 and
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
      (.input_manifest_sha256 | test("^[0-9a-f]{64}$")) and
      (.instance_manifest_sha256 | test("^[0-9a-f]{64}$")) and
      (.membership_preflight_sha256 | test("^[0-9a-f]{64}$")) and
      (.worker_sha256 | test("^[0-9a-f]{64}$")) and
      (.probe_worker_sha256 | test("^[0-9a-f]{64}$")) and
      (.activation_worker_sha256 | test("^[0-9a-f]{64}$")) and
      ([.code_hashes[] | test("^[0-9a-f]{64}$")] | all) and
      (.python_identity.portable_identity_sha256 | test("^[0-9a-f]{64}$"))
    ' "$PLAN" >/dev/null; then
    echo "Reviewed probe-first plan contract mismatch; nothing submitted." >&2
    return 1
  fi
  MATRIX_ROWS=$(awk 'END {print NR-1}' "$MATRIX")
  if [[ "$MATRIX_ROWS" -ne 138 ]]; then
    echo "Task matrix does not contain exactly 138 rows." >&2
    return 1
  fi

  echo "CAMPAIGN=$CAMPAIGN"
  echo "PLAN_SHA256=$FILE_SHA"
  echo "SCIENCE_TASKS=138"
  echo "INFRASTRUCTURE_TASKS=3 (2 probes + 1 activation controller)"
  echo "K40_MIP_TASKS=0"
  if [[ "$SUBMIT_APPROVAL" != "YES" ]]; then
    echo "DRY_RUN_READY=true"
    echo "Set EVSP_LADDER_SUBMIT=YES and rerun this same campaign to launch."
    return 0
  fi

  CAMPAIGN_ROOT=$(jq -r '.campaign_root' "$PLAN")
  MANIFEST="$CAMPAIGN_ROOT/campaign.json"
  RETRY_ARG=()
  if [[ "$RETRY_KIND" == "failed_probes" ]]; then
    RETRY_ARG=(--retry-failed-probes)
  elif [[ "$RETRY_KIND" == "failed_activation" ]]; then
    RETRY_ARG=(--retry-failed-activation)
  fi

  # Always re-enter the exact idempotent launcher.  It recovers accepted jobs
  # by their bound identity and, critically, diagnoses a terminal controller
  # instead of inferring success from stale `released` flags.
  echo "=== submit probe-first infrastructure ==="
  env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py" \
    "$COMMIT" launch_scale_ladder.py \
    --campaign "$CAMPAIGN" --python "$PYTHON" \
    --reservation-root "$RESERVATIONS" \
    --approved-plan-sha256 "$FILE_SHA" --submit \
    "${RETRY_ARG[@]}" >"$SUBMIT_OUT" 2>"$SUBMIT_ERR"
  SUBMIT_STATUS=$?
  if [[ "$SUBMIT_STATUS" -ne 0 ]]; then
    echo "Probe-first submission stopped fail-closed." >&2
    echo "Inspect the manifest/logs; use an explicit retry flag only for a proven terminal infrastructure failure." >&2
    [[ -s "$MANIFEST" ]] && jq '{submission_state,probe_state,infrastructure_retry,activation,gate_state,gate_job_id,submitted_arrays}' "$MANIFEST"
    tail -n 120 "$SUBMIT_OUT" "$SUBMIT_ERR" 2>/dev/null
    squeue --me -o '%.14i %.18j %.2t %.10M %R' 2>/dev/null
    return 1
  fi

  if [[ ! -s "$MANIFEST" ]] || ! jq -e --arg sha "$FILE_SHA" '
      .approval_sha256 == $sha and
      .submission_protocol == "probe_first_activation_v1" and
      ([.infrastructure_probes.default_partition.job_id,
        .infrastructure_probes.scaglione.job_id,
        .activation.job_id] | all(type == "string" and test("^[0-9]+$"))) and
      .infrastructure_probes.default_partition.released == true and
      .infrastructure_probes.scaglione.released == true and
      .activation.released == true and
      .infrastructure_probes.default_partition.release_verification.verified == true and
      .infrastructure_probes.default_partition.release_verification.job_id ==
        .infrastructure_probes.default_partition.job_id and
      .infrastructure_probes.default_partition.release_verification.observation.job_id ==
        .infrastructure_probes.default_partition.job_id and
      .infrastructure_probes.scaglione.release_verification.verified == true and
      .infrastructure_probes.scaglione.release_verification.job_id ==
        .infrastructure_probes.scaglione.job_id and
      .infrastructure_probes.scaglione.release_verification.observation.job_id ==
        .infrastructure_probes.scaglione.job_id and
      .activation.release_verification.verified == true and
      .activation.release_verification.job_id == .activation.job_id and
      .activation.release_verification.observation.job_id == .activation.job_id and
      .activation.probe_job_ids.default_partition ==
        .infrastructure_probes.default_partition.job_id and
      .activation.probe_job_ids.scaglione ==
        .infrastructure_probes.scaglione.job_id and
      (if .gate_state == "not_created" then
         .submitted == false and .gate_job_id == null and
         (.submitted_arrays | length) == 0 and
         (.reservations | length) == 0
       else true end)
    ' "$MANIFEST" >/dev/null; then
    echo "Infrastructure returned without a valid durable manifest." >&2
    return 1
  fi

  echo "INFRASTRUCTURE_ARMED=true"
  echo "SCIENCE_ONLY_AFTER_BOTH_PROBES_VALIDATE=true"
  jq '{submission_state,probe_state,probe_results,activation,gate_state,gate_job_id,submitted_arrays}' "$MANIFEST"
  squeue --me -o '%.14i %.18j %.2t %.10M %R' 2>/dev/null
  echo "CAMPAIGN_ROOT=$CAMPAIGN_ROOT"
  echo "PLAN=$PLAN"
  echo "MATRIX=$MATRIX"
}

main
