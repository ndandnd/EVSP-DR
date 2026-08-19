#!/bin/bash

main() {
  OLD_COMMIT="a27eed66ad9973e91c3d638acbaf9d34c47f0c86"
  OLD_PLAN_SHA="5fc295995df1119dd5aaf408422154649c4b0c9657a4417e24839bdb58e3b7d5"
  OLD_RUN_ROOT="$HOME/EVSP-DR-scale-ladder-a27eed66"
  OLD_CAMPAIGN="slad_20260819_023048_a27eed66"
  OLD_ROOT="$OLD_RUN_ROOT/src/results/scale_ladder/$OLD_CAMPAIGN"
  OLD_PLAN="$OLD_ROOT/approved-plan.json"
  OLD_MANIFEST="$OLD_ROOT/campaign.json"
  SCIENCE_IDS="217103,217104,217105,217106,217107,217108"
  ALL_IDS="217102,217103,217104,217105,217106,217107,217108,217109,217110,217145"
  EXPECTED_ARRAY_IDS="217103,217104,217105,217106,217107,217108"
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  AUDIT="$HOME/evsp_scale_ladder_closeout/${OLD_CAMPAIGN}_${STAMP}"
  PRE_ARCHIVE="$HOME/evsp_archives/${OLD_CAMPAIGN}_${STAMP}_before_cancel.tar.gz"
  FINAL_ARCHIVE="$HOME/evsp_archives/${OLD_CAMPAIGN}_${STAMP}_closed.tar.gz"

  refresh_audit_hashes() {
    (
      cd "$AUDIT" || return 1
      find . -type f \
        ! -name archive_contents.sha256 \
        ! -name archive_contents.sha256.tmp -print0 |
        sort -z |
        xargs -0 sha256sum >archive_contents.sha256.tmp || return 1
      mv archive_contents.sha256.tmp archive_contents.sha256 || return 1
      sha256sum -c archive_contents.sha256 || return 1
    )
  }

  echo "=== Validate the exact failed campaign ==="
  if [ ! -s "$OLD_PLAN" ] || [ ! -s "$OLD_MANIFEST" ]; then
    echo "Required old plan/manifest is missing; nothing cancelled."
    return 1
  fi
  if [ "$(git -C "$OLD_RUN_ROOT" rev-parse HEAD 2>/dev/null)" != "$OLD_COMMIT" ]; then
    echo "Old checkout commit mismatch; nothing cancelled."
    return 1
  fi
  if [ "$(sha256sum "$OLD_PLAN" | awk '{print $1}')" != "$OLD_PLAN_SHA" ]; then
    echo "Old approved-plan hash mismatch; nothing cancelled."
    return 1
  fi
  MANIFEST_OK=$(jq -r --arg sha "$OLD_PLAN_SHA" \
    --arg campaign "$OLD_CAMPAIGN" --arg root "$OLD_ROOT" \
    --arg commit "$OLD_COMMIT" '
    (.approval_sha256 == $sha) and
    (.campaign == $campaign) and (.campaign_root == $root) and
    (.checkout_identity.commit == $commit) and
    (.task_count == 138) and ((.jobs | length) == 138) and
    ((.gate_job_id | tostring) == "217102") and
    (.gate_state == "held_probe_failure") and
    ((.reservations | length) == 138) and
    ((.submitted_arrays | keys | sort) ==
      ["CG","CG_SENSITIVITY","MIP_KNOWN","MIP_RAW","PREFLIGHT","SEED"]) and
    ((.submitted_arrays.PREFLIGHT | tostring) == "217103") and
    ((.submitted_arrays.SEED | tostring) == "217104") and
    ((.submitted_arrays.CG | tostring) == "217105") and
    ((.submitted_arrays.CG_SENSITIVITY | tostring) == "217106") and
    ((.submitted_arrays.MIP_RAW | tostring) == "217107") and
    ((.submitted_arrays.MIP_KNOWN | tostring) == "217108") and
    ((.infrastructure_probes.default_partition.job_id | tostring) == "217109") and
    ((.infrastructure_probes.scaglione.job_id | tostring) == "217110")
  ' "$OLD_MANIFEST" 2>/dev/null)
  if [ "$MANIFEST_OK" != "true" ]; then
    echo "Old manifest identity mismatch; nothing cancelled."
    return 1
  fi
  if ! jq -e --arg campaign "$OLD_CAMPAIGN" --arg root "$OLD_ROOT" \
      --arg commit "$OLD_COMMIT" '
      .campaign == $campaign and .campaign_root == $root and
      .checkout_identity.commit == $commit and .task_count == 138 and
      (.jobs | length) == 138 and
      (.task_groups.PREFLIGHT | length) == 22 and
      (.task_groups.SEED | length) == 21 and
      (.task_groups.CG | length) == 23 and
      (.task_groups.CG_SENSITIVITY | length) == 30 and
      (.task_groups.MIP_RAW | length) == 21 and
      (.task_groups.MIP_KNOWN | length) == 21
    ' "$OLD_PLAN" >/dev/null; then
    echo "Old approved-plan scientific contract mismatch; nothing cancelled."
    return 1
  fi

  mkdir -p "$AUDIT/reservations" "$HOME/evsp_archives" || return 1
  sacct -X -n -P -j "$SCIENCE_IDS" \
    --format=JobIDRaw,State,ElapsedRaw,Start,NodeList \
    >"$AUDIT/science_accounting_before.txt" 2>"$AUDIT/sacct_before.err"
  SACCT_STATUS=$?
  if [ "$SACCT_STATUS" -ne 0 ] || [ ! -s "$AUDIT/science_accounting_before.txt" ]; then
    echo "Could not prove old scientific jobs never ran; nothing cancelled."
    return 1
  fi
  BAD_ROWS=$(awk -F'|' '
    {
      state=$2; sub(/[ +].*$/, "", state)
      elapsed=$3+0
      start=$4
      if (state != "PENDING" && state != "CANCELLED") print $0
      else if (elapsed != 0) print $0
      else if (state == "CANCELLED" && $5 != "" && $5 != "None assigned" &&
               $5 != "Unknown" && $5 != "N/A" && $5 != "None") print $0
    }
  ' "$AUDIT/science_accounting_before.txt")
  COVERED_IDS=$(awk -F'|' '
    {
      id=$1; sub(/[_.].*$/, "", id)
      if (id != "") seen[id]=1
    }
    END {for (id in seen) print id}
  ' "$AUDIT/science_accounting_before.txt" | sort | paste -sd, -)
  if [ "$COVERED_IDS" != "$EXPECTED_ARRAY_IDS" ] || [ -n "$BAD_ROWS" ]; then
    echo "Old scientific activity is not proven absent; nothing cancelled."
    echo "covered_ids=$COVERED_IDS"
    printf '%s\n' "$BAD_ROWS"
    return 1
  fi

  echo "=== Preserve plan, manifest, logs, accounting, and reservations ==="
  cp -a "$OLD_ROOT" "$AUDIT/campaign_root" || return 1
  git -C "$OLD_RUN_ROOT" rev-parse HEAD >"$AUDIT/git_head.txt" || return 1
  git -C "$OLD_RUN_ROOT" status --porcelain --untracked-files=all \
    >"$AUDIT/git_status.txt" || return 1
  if [ -f "$0" ]; then
    cp -p "$0" "$AUDIT/closeout_operator.sh" || return 1
    (
      cd "$AUDIT" || return 1
      sha256sum closeout_operator.sh >closeout_operator.sha256 || return 1
    ) || return 1
  else
    echo "Closeout operator script is not a regular file; nothing cancelled."
    return 1
  fi
  sacct -X -j "$ALL_IDS" \
    --format=JobIDRaw,JobName%18,State,Elapsed,ElapsedRaw,ExitCode,Reason%30,NodeList -P \
    >"$AUDIT/all_accounting_before.txt" 2>&1 || return 1
  for jid in 217102 217103 217104 217105 217106 217107 217108; do
    scontrol show job -dd -o "$jid" >>"$AUDIT/scontrol_before.txt" 2>&1 || return 1
  done
  RESERVATION_ROOT=$(jq -r '.reservation_root' "$OLD_PLAN")
  jq -r --arg root "$RESERVATION_ROOT" '
    [.jobs[] | ($root + "/" + .execution_digest + ".json")] | sort[]
  ' "$OLD_PLAN" >"$AUDIT/expected_reservation_paths.txt" || return 1
  jq -r '[.reservations[]] | sort[]' "$OLD_MANIFEST" \
    >"$AUDIT/manifest_reservation_paths.txt" || return 1
  if ! cmp -s "$AUDIT/expected_reservation_paths.txt" \
            "$AUDIT/manifest_reservation_paths.txt"; then
    echo "Manifest reservation set differs from the approved plan; nothing cancelled."
    return 1
  fi
  jq -r '.jobs[] | [.execution_digest, .job_key] | @tsv' "$OLD_PLAN" \
    >"$AUDIT/expected_reservations.tsv" || return 1
  if [ "$(cut -f1 "$AUDIT/expected_reservations.tsv" | sort -u | wc -l | tr -d ' ')" \
       -ne 138 ]; then
    echo "Approved plan does not contain 138 unique execution digests."
    return 1
  fi
  COPY_FAILED=0
  RESERVATION_COUNT=0
  while IFS=$'\t' read -r digest job_key; do
    RESERVATION_COUNT=$((RESERVATION_COUNT + 1))
    reservation="$RESERVATION_ROOT/$digest.json"
    if [ ! -f "$reservation" ] || [ -L "$reservation" ]; then
      echo "Reservation is missing, non-regular, or a symlink: $reservation"
      COPY_FAILED=1
      continue
    fi
    if ! jq -e --arg sha "$OLD_PLAN_SHA" --arg digest "$digest" \
      --arg job "$job_key" '
        .schema == "evsp-dr-scale-ladder-reservation-v1" and
        .plan_sha256 == $sha and .execution_digest == $digest and
        .job_key == $job
      ' "$reservation" >/dev/null; then
      echo "Reservation content mismatch: $reservation"
      COPY_FAILED=1
      continue
    fi
    target="$AUDIT/reservations/$digest.json"
    cp -p "$reservation" "$target" || COPY_FAILED=1
    if ! source_sha=$(sha256sum "$reservation" | awk 'NF {print $1; exit}') ||
       [ -z "$source_sha" ]; then
      COPY_FAILED=1
      continue
    fi
    if ! copied_sha=$(sha256sum "$target" | awk 'NF {print $1; exit}') ||
       [ -z "$copied_sha" ]; then
      COPY_FAILED=1
      continue
    fi
    if [ "$source_sha" != "$copied_sha" ]; then
      COPY_FAILED=1
    fi
    printf '%s  %s\n' "$source_sha" "$reservation" \
      >>"$AUDIT/reservation_hashes.txt"
  done <"$AUDIT/expected_reservations.tsv"
  if [ "$COPY_FAILED" -ne 0 ] || [ "$RESERVATION_COUNT" -ne 138 ]; then
    echo "Reservation preservation failed; nothing cancelled."
    return 1
  fi
  refresh_audit_hashes || return 1
  tar -C "$(dirname "$AUDIT")" -czf "$PRE_ARCHIVE" "$(basename "$AUDIT")" || return 1
  (
    cd "$(dirname "$PRE_ARCHIVE")" || return 1
    archive_name=$(basename "$PRE_ARCHIVE")
    sha256sum "$archive_name" >"$archive_name.sha256" || return 1
    sha256sum -c "$archive_name.sha256" || return 1
  ) || return 1

  echo "=== Cancel only the proven-held old gate and arrays ==="
  squeue -h -u "$USER" -o '%A|%T' \
    >"$AUDIT/squeue_before.txt" 2>"$AUDIT/squeue_before.err"
  if [ "$?" -ne 0 ]; then
    echo "Could not query old active jobs; nothing cancelled."
    return 1
  fi
  ACTIVE=$(awk -F'|' '
    $1=="217102" || $1=="217103" || $1=="217104" ||
    $1=="217105" || $1=="217106" || $1=="217107" || $1=="217108" {
      print $1
    }
  ' "$AUDIT/squeue_before.txt" | sort -u)
  BAD_ACTIVE_STATES=$(awk -F'|' '
    ($1=="217102" || $1=="217103" || $1=="217104" ||
     $1=="217105" || $1=="217106" || $1=="217107" || $1=="217108") &&
     $2!="PENDING" {print $0}
  ' "$AUDIT/squeue_before.txt")
  if [ -n "$BAD_ACTIVE_STATES" ]; then
    echo "An old target is active outside PENDING; nothing cancelled."
    printf '%s\n' "$BAD_ACTIVE_STATES"
    return 1
  fi
  if [ -n "$ACTIVE" ]; then
    mapfile -t ACTIVE_IDS <<<"$ACTIVE"
    scancel "${ACTIVE_IDS[@]}" || return 1
  fi
  attempt=0
  while [ "$attempt" -lt 30 ]; do
    squeue -h -u "$USER" -o '%A|%T' \
      >"$AUDIT/squeue_after_poll.txt" 2>"$AUDIT/squeue_after.err"
    SQUEUE_STATUS=$?
    if [ "$SQUEUE_STATUS" -ne 0 ]; then
      echo "Post-cancel squeue failed; do not launch a new campaign."
      return 1
    fi
    REMAINING=$(awk -F'|' '
      $1=="217102" || $1=="217103" || $1=="217104" ||
      $1=="217105" || $1=="217106" || $1=="217107" || $1=="217108" {
        print $0
      }
    ' "$AUDIT/squeue_after_poll.txt")
    [ -z "$REMAINING" ] && break
    sleep 2
    attempt=$((attempt + 1))
  done
  if [ -n "$REMAINING" ]; then
    echo "Old jobs remain active; no new campaign should be launched."
    return 1
  fi
  cp "$AUDIT/squeue_after_poll.txt" "$AUDIT/squeue_after.txt" || return 1
  sacct -X -j "$ALL_IDS" \
    --format=JobIDRaw,JobName%18,State,Elapsed,ElapsedRaw,ExitCode,Reason%30,NodeList -P \
    >"$AUDIT/all_accounting_after.txt" 2>&1 || return 1
  sacct -X -n -P -j "$SCIENCE_IDS" \
    --format=JobIDRaw,State,ElapsedRaw,Start,NodeList \
    >"$AUDIT/science_accounting_after.txt" 2>"$AUDIT/sacct_after.err"
  AFTER_SACCT_STATUS=$?
  AFTER_COVERED_IDS=$(awk -F'|' '
    {
      id=$1; sub(/[_.].*$/, "", id)
      if (id != "") seen[id]=1
    }
    END {for (id in seen) print id}
  ' "$AUDIT/science_accounting_after.txt" | sort | paste -sd, -)
  AFTER_BAD_ROWS=$(awk -F'|' '
    {
      state=$2; sub(/[ +].*$/, "", state)
      elapsed=$3+0
      if (state != "PENDING" && state != "CANCELLED") print $0
      else if (elapsed != 0) print $0
      else if (state == "CANCELLED" && $5 != "" && $5 != "None assigned" &&
               $5 != "Unknown" && $5 != "N/A" && $5 != "None") print $0
    }
  ' "$AUDIT/science_accounting_after.txt")
  CLOSEOUT_OK=true
  if [ "$AFTER_SACCT_STATUS" -ne 0 ] || \
     [ "$AFTER_COVERED_IDS" != "$EXPECTED_ARRAY_IDS" ] || \
     [ -n "$AFTER_BAD_ROWS" ]; then
    CLOSEOUT_OK=false
  fi
  {
    echo "schema=evsp-dr-scale-ladder-closeout-v1"
    echo "campaign=$OLD_CAMPAIGN"
    echo "approved_plan_sha256=$OLD_PLAN_SHA"
    echo "scientific_arrays_never_started=$CLOSEOUT_OK"
    echo "post_cancel_squeue_targets_empty=true"
    echo "post_cancel_covered_ids=$AFTER_COVERED_IDS"
  } >"$AUDIT/closeout_status.txt" || return 1
  refresh_audit_hashes || return 1
  tar -C "$(dirname "$AUDIT")" -czf "$FINAL_ARCHIVE" "$(basename "$AUDIT")" || return 1
  (
    cd "$(dirname "$FINAL_ARCHIVE")" || return 1
    archive_name=$(basename "$FINAL_ARCHIVE")
    sha256sum "$archive_name" >"$archive_name.sha256" || return 1
    sha256sum -c "$archive_name.sha256" || return 1
  ) || return 1

  if [ "$CLOSEOUT_OK" != "true" ]; then
    echo "Old jobs were cancelled, but post-cancel no-run proof failed."
    echo "Do not launch the new campaign; preserve and inspect $FINAL_ARCHIVE"
    printf '%s\n' "$AFTER_BAD_ROWS"
    return 1
  fi

  echo "OLD_CAMPAIGN_CLOSED=true"
  echo "PRE_CANCEL_ARCHIVE=$PRE_ARCHIVE"
  echo "FINAL_ARCHIVE=$FINAL_ARCHIVE"
}

main
