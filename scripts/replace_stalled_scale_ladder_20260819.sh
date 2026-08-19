#!/bin/bash

# One-time, fail-closed replacement of the dead ba09d46 scale-ladder campaign.
# This script intentionally avoids shell-wide errexit/nounset/pipefail so it
# cannot terminate the operator's surrounding Unicorn login shell.

cancelled_scalar_entry_from_rows() {
  local jid=$1 expected_name=$2 raw=$3
  local line row_id row_name row_state row_elapsed row_exit normalized
  local task_count=0 entry=''
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    IFS='|' read -r row_id row_name row_state row_elapsed row_exit <<<"$line"
    [[ "$row_id" == "$jid" ]] || return 1
    task_count=$((task_count + 1))
    normalized=${row_state%% *}
    normalized=${normalized%%+*}
    [[ "$row_name" == "$expected_name" && \
       "$normalized" == "CANCELLED" && \
       "$row_elapsed" == "00:00:00" ]] || return 1
    entry=$(jq -cn --arg id "$row_id" --arg name "$row_name" \
      --arg state "$normalized" --arg elapsed "$row_elapsed" \
      --arg exit_code "$row_exit" \
      '{job_id:$id,job_name:$name,kind:"scalar",state:$state,
        elapsed:$elapsed,exit_code:$exit_code}') || return 1
  done <<<"$raw"
  [[ "$task_count" -eq 1 && -n "$entry" ]] || return 1
  printf '%s\n' "$entry"
}

cancelled_array_entry_from_rows() {
  local jid=$1 expected_name=$2 expected_range=$3 raw=$4
  local range_start=${expected_range%-*} range_end=${expected_range#*-}
  local expected_task_count=$((range_end - range_start + 1))
  local line row_id row_name row_state row_elapsed row_exit normalized task
  local task_count=0 tasks_json='[]' entry
  declare -a TASK_NAME=() TASK_STATE=() TASK_ELAPSED=() TASK_EXIT=()
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    IFS='|' read -r row_id row_name row_state row_elapsed row_exit <<<"$line"
    [[ "$row_id" == "$jid"_* ]] || return 1
    task=${row_id#"$jid"_}
    [[ "$task" =~ ^[0-9]+$ && "$task" -ge "$range_start" && \
       "$task" -le "$range_end" ]] || return 1
    [[ -z "${TASK_STATE[$task]+present}" ]] || return 1
    normalized=${row_state%% *}
    normalized=${normalized%%+*}
    [[ "$row_name" == "$expected_name" && \
       "$normalized" == "CANCELLED" && \
       "$row_elapsed" == "00:00:00" ]] || return 1
    TASK_NAME[$task]=$row_name
    TASK_STATE[$task]=$normalized
    TASK_ELAPSED[$task]=$row_elapsed
    TASK_EXIT[$task]=$row_exit
    task_count=$((task_count + 1))
  done <<<"$raw"
  [[ "$task_count" -eq "$expected_task_count" ]] || return 1
  for ((task=range_start; task<=range_end; task++)); do
    [[ -n "${TASK_STATE[$task]+present}" ]] || return 1
    tasks_json=$(jq -c \
      --arg task_id "$task" --arg job_id "${jid}_${task}" \
      --arg name "${TASK_NAME[$task]}" --arg state "${TASK_STATE[$task]}" \
      --arg elapsed "${TASK_ELAPSED[$task]}" \
      --arg exit_code "${TASK_EXIT[$task]}" \
      '. + [{task_id:($task_id|tonumber),job_id:$job_id,job_name:$name,
              state:$state,elapsed:$elapsed,exit_code:$exit_code}]' \
      <<<"$tasks_json") || return 1
  done
  entry=$(jq -cn --arg id "$jid" --arg name "$expected_name" \
    --arg range "$expected_range" --argjson task_count "$expected_task_count" \
    --argjson tasks "$tasks_json" \
    '{job_id:$id,job_name:$name,kind:"array",task_range:$range,
      task_count:$task_count,tasks:$tasks}') || return 1
  printf '%s\n' "$entry"
}

canonical_dependency() {
  local value=$1 clause dependency_type dependency_ids dependency_token
  local dependency_id
  local atom existing insert_at index
  local -a clauses=() dependency_parts=() atoms=() sorted_atoms=()

  [[ -n "$value" && "$value" != *'?'* && \
     "$value" != ','* && "$value" != *',' && "$value" != *',,'* ]] || \
    return 1

  IFS=',' read -r -a clauses <<<"$value"
  [[ "${#clauses[@]}" -gt 0 ]] || return 1
  for clause in "${clauses[@]}"; do
    [[ "$clause" == *:* ]] || return 1
    dependency_type=${clause%%:*}
    dependency_ids=${clause#*:}
    case "$dependency_type" in
      after|afterany|afterburstbuffer|aftercorr|afternotok|afterok) ;;
      *) return 1 ;;
    esac
    [[ -n "$dependency_ids" && "$dependency_ids" != ':'* && \
       "$dependency_ids" != *':' && "$dependency_ids" != *'::'* ]] || \
      return 1
    IFS=':' read -r -a dependency_parts <<<"$dependency_ids"
    [[ "${#dependency_parts[@]}" -gt 0 ]] || return 1
    for dependency_token in "${dependency_parts[@]}"; do
      # Slurm renders a dependency on an entire array controller as JOBID_*.
      # Preserve that suffix as part of the semantic identity.  A numeric task
      # suffix is a different dependency and is deliberately rejected.
      [[ "$dependency_token" =~ ^([0-9]+(_\*)?)(\((unfulfilled|failed)\))?$ ]] || \
        return 1
      dependency_id=${BASH_REMATCH[1]}
      atoms+=("$dependency_type:$dependency_id")
    done
  done

  # Comma-separated dependency clauses are conjunctive, so their display
  # order is immaterial.  Sort without an unchecked pipeline.
  for atom in "${atoms[@]}"; do
    insert_at=${#sorted_atoms[@]}
    for ((index=0; index<${#sorted_atoms[@]}; index++)); do
      existing=${sorted_atoms[$index]}
      if [[ "$atom" < "$existing" ]]; then
        insert_at=$index
        break
      fi
    done
    sorted_atoms=(
      "${sorted_atoms[@]:0:$insert_at}"
      "$atom"
      "${sorted_atoms[@]:$insert_at}"
    )
  done
  printf '%s\n' "${sorted_atoms[@]}"
}

identity_mismatch() {
  local jid=$1 field=$2 expected=$3 observed=$4
  printf 'Old job %s identity mismatch: field=%s expected=%q observed=%q\n' \
    "$jid" "$field" "$expected" "$observed" >&2
  return 1
}

record_field_value() {
  local record=$1 key=$2 token value=''
  local found=0
  local -a tokens=()
  read -r -a tokens <<<"$record"
  for token in "${tokens[@]}"; do
    if [[ "$token" == "$key="* ]]; then
      found=$((found + 1))
      value=${token#*=}
    fi
  done
  [[ "$found" -eq 1 ]] || return 1
  printf '%s\n' "$value"
}

validate_pending_array_records() {
  local jid=$1 expected_name=$2 expected_partition=$3 expected_range=$4
  local expected_comment=$5 expected_dependency=$6 expected_user=$7 raw=$8
  local range_start range_end expected_task_count record_count=0
  local record job_field array_job_id task_expression task_start task_end task
  local name state partition runtime comment user_id user_number reason dependency
  local observed_canonical_dependency expected_canonical_dependency
  local seen_job_records='|'
  declare -a SEEN_TASK=()

  [[ "$expected_range" =~ ^([0-9]+)-([0-9]+)$ ]] || return 1
  range_start=${BASH_REMATCH[1]}
  range_end=${BASH_REMATCH[2]}
  [[ "$range_end" -ge "$range_start" ]] || return 1
  expected_task_count=$((range_end - range_start + 1))
  expected_canonical_dependency=$(canonical_dependency "$expected_dependency") ||
    return 1

  while IFS= read -r record; do
    [[ -n "$record" ]] || continue
    record_count=$((record_count + 1))

    job_field=$(record_field_value "$record" JobId) || {
      identity_mismatch "$jid" JobId 'one numeric value per record' missing
      return 1
    }
    [[ "$job_field" =~ ^[0-9]+$ ]] || {
      identity_mismatch "$jid" JobId numeric "$job_field"
      return 1
    }
    [[ "$seen_job_records" != *"|$job_field|"* ]] || {
      identity_mismatch "$jid" JobId unique "$job_field"
      return 1
    }
    seen_job_records+="$job_field|"

    array_job_id=$(record_field_value "$record" ArrayJobId) || {
      identity_mismatch "$jid" ArrayJobId "$jid" missing
      return 1
    }
    [[ "$array_job_id" == "$jid" ]] || {
      identity_mismatch "$jid" ArrayJobId "$jid" "$array_job_id"
      return 1
    }
    task_expression=$(record_field_value "$record" ArrayTaskId) || {
      identity_mismatch "$jid" ArrayTaskId "$expected_range" missing
      return 1
    }
    if [[ "$task_expression" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      task_start=${BASH_REMATCH[1]}
      task_end=${BASH_REMATCH[2]}
    elif [[ "$task_expression" =~ ^[0-9]+$ ]]; then
      task_start=$task_expression
      task_end=$task_expression
    else
      identity_mismatch "$jid" ArrayTaskId 'scalar or ascending range' \
        "$task_expression"
      return 1
    fi
    [[ "$task_end" -ge "$task_start" && \
       "$task_start" -ge "$range_start" && \
       "$task_end" -le "$range_end" ]] || {
      identity_mismatch "$jid" ArrayTaskId "$expected_range" "$task_expression"
      return 1
    }

    name=$(record_field_value "$record" JobName) || return 1
    state=$(record_field_value "$record" JobState) || return 1
    partition=$(record_field_value "$record" Partition) || return 1
    runtime=$(record_field_value "$record" RunTime) || return 1
    comment=$(record_field_value "$record" Comment) || return 1
    user_id=$(record_field_value "$record" UserId) || return 1
    reason=$(record_field_value "$record" Reason) || return 1
    dependency=$(record_field_value "$record" Dependency) || return 1

    [[ "$name" == "$expected_name" ]] || {
      identity_mismatch "$jid" JobName "$expected_name" "$name"
      return 1
    }
    [[ "$state" == "PENDING" ]] || {
      identity_mismatch "$jid" JobState PENDING "$state"
      return 1
    }
    [[ "$partition" == "$expected_partition" ]] || {
      identity_mismatch "$jid" Partition "$expected_partition" "$partition"
      return 1
    }
    [[ "$runtime" == "00:00:00" ]] || {
      identity_mismatch "$jid" RunTime 00:00:00 "$runtime"
      return 1
    }
    [[ "$comment" == "$expected_comment" ]] || {
      identity_mismatch "$jid" Comment "$expected_comment" "$comment"
      return 1
    }
    if [[ "$user_id" == "$expected_user("*")" ]]; then
      user_number=${user_id#"$expected_user("}
      user_number=${user_number%")"}
    else
      user_number=''
    fi
    [[ "$user_number" =~ ^[0-9]+$ ]] || {
      identity_mismatch "$jid" UserId "$expected_user(<numeric uid>)" "$user_id"
      return 1
    }
    [[ "$reason" == "Dependency" ]] || {
      identity_mismatch "$jid" Reason Dependency "$reason"
      return 1
    }
    observed_canonical_dependency=$(canonical_dependency "$dependency") || {
      identity_mismatch "$jid" DependencyFormat "$expected_dependency" \
        "$dependency"
      return 1
    }
    [[ "$observed_canonical_dependency" == \
       "$expected_canonical_dependency" ]] || {
      identity_mismatch "$jid" Dependency "$expected_canonical_dependency" \
        "$observed_canonical_dependency"
      return 1
    }

    for ((task=task_start; task<=task_end; task++)); do
      [[ -z "${SEEN_TASK[$task]+present}" ]] || {
        identity_mismatch "$jid" ArrayTaskId 'nonoverlapping coverage' \
          "$task_expression"
        return 1
      }
      SEEN_TASK[$task]=1
    done
  done <<<"$raw"

  [[ "$record_count" -gt 0 ]] || return 1
  for ((task=range_start; task<=range_end; task++)); do
    [[ -n "${SEEN_TASK[$task]+present}" ]] || {
      identity_mismatch "$jid" ArrayTaskCoverage "$expected_range" \
        "missing task $task"
      return 1
    }
  done
  [[ "${#SEEN_TASK[@]}" -eq "$expected_task_count" ]] || return 1
  return 0
}

wait_for_cancelled_accounting() {
  local attempt
  # SlurmDBD can lag the controller after an array is cancelled.  Preserve a
  # bounded five-minute recovery window before declaring the scientific
  # closeout incomplete; never relax the full-range/zero-runtime proof.
  for ((attempt=1; attempt<=60; attempt++)); do
    if verify_cancelled_accounting >/dev/null 2>&1; then
      return 0
    fi
    sleep 5 || {
      echo "Interrupted while waiting for cancellation accounting." >&2
      return 1
    }
    if ((attempt % 6 == 0)); then
      echo "Waiting for complete cancellation accounting ($((attempt * 5))s)."
    fi
  done
  verify_cancelled_accounting
}

main() {
  umask 077

  if [[ -z "$HOME" || -z "$USER" ]]; then
    echo "HOME and USER must both be set." >&2
    return 1
  fi

  OLD_RUN_ROOT="$HOME/EVSP-DR-scale-ladder-ba09d46"
  OLD_CAMPAIGN="slad_20260819_031808_ba09d46"
  OLD_ROOT="$OLD_RUN_ROOT/src/results/scale_ladder/$OLD_CAMPAIGN"
  OLD_PLAN="$OLD_ROOT/approved-plan.json"
  OLD_MANIFEST="$OLD_ROOT/campaign.json"
  OLD_PLAN_SHA="bcea6b9391cf49515de2dc8ae06ee6bdc70186fde2c80243a2eeaec62b2c083b"
  OLD_COMMIT="ba09d4602ded98f9c9157f52af169ef511b5abf7"

  NEW_COMMIT=${REVIEWED_COMMIT:-}
  if [[ ! "$NEW_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Set REVIEWED_COMMIT to the exact reviewed 40-character SHA." >&2
    return 1
  fi
  NEW_SHORT=${NEW_COMMIT:0:7}
  NEW_CAMPAIGN=${LADDER_CAMPAIGN:-"slad_flat_primary_v2_$NEW_SHORT"}
  if [[ ! "$NEW_CAMPAIGN" =~ ^[a-z0-9][a-z0-9_-]{2,79}$ ]]; then
    echo "LADDER_CAMPAIGN is invalid." >&2
    return 1
  fi
  SOURCE_ROOT="$HOME/EVSP-DR"
  LAUNCH_SCRIPT="$HOME/launch_scale_ladder_probe_first_$NEW_SHORT.sh"
  NEW_RUN_ROOT="$HOME/EVSP-DR-scale-ladder-${NEW_COMMIT:0:12}"
  NEW_PLAN_ROOT="$HOME/evsp_scale_ladder_plans"
  NEW_RESERVATIONS="$HOME/evsp_scale_ladder_reservations/$NEW_CAMPAIGN"
  NEW_PYTHON="$HOME/evsp_env/bin/python3.12"
  NEW_PLAN="$NEW_PLAN_ROOT/$NEW_CAMPAIGN.plan.json"
  NEW_MATRIX="$NEW_PLAN_ROOT/$NEW_CAMPAIGN.tasks.csv"

  for tool in git jq sha256sum tar scontrol squeue sacct scancel \
      awk date bash chmod cmp cp find ln rm mv sleep; do
    command -v "$tool" >/dev/null 2>&1 || {
      echo "Missing required command: $tool" >&2
      return 1
    }
  done

  echo "=== stage exact reviewed launcher before changing Slurm ==="
  if [[ ! -d "$SOURCE_ROOT/.git" ]]; then
    git clone https://github.com/ndandnd/EVSP-DR.git "$SOURCE_ROOT" || return 1
  fi
  git -C "$SOURCE_ROOT" fetch origin "$NEW_COMMIT" || return 1
  git -C "$SOURCE_ROOT" cat-file -e "$NEW_COMMIT^{commit}" || return 1
  launch_temporary="$LAUNCH_SCRIPT.tmp.$$"
  [[ ! -e "$launch_temporary" ]] || {
    echo "Temporary launcher path already exists: $launch_temporary" >&2
    return 1
  }
  git -C "$SOURCE_ROOT" show \
    "$NEW_COMMIT:scripts/launch_scale_ladder_probe_first.sh" \
    >"$launch_temporary" || return 1
  chmod 700 "$launch_temporary" || return 1
  bash -n "$launch_temporary" || return 1
  if [[ -e "$LAUNCH_SCRIPT" ]]; then
    if [[ -L "$LAUNCH_SCRIPT" ]] || \
       ! cmp -s "$launch_temporary" "$LAUNCH_SCRIPT"; then
      echo "Existing staged launcher differs; refusing overwrite." >&2
      return 1
    fi
    rm "$launch_temporary" || return 1
  else
    ln "$launch_temporary" "$LAUNCH_SCRIPT" || return 1
    rm "$launch_temporary" || return 1
  fi

  echo "=== preflight every fresh-launch prerequisite ==="
  [[ -x "$NEW_PYTHON" && \
     "$($NEW_PYTHON --version 2>&1)" == Python\ 3.12.* ]] || {
    echo "Approved Unicorn Python 3.12 is unavailable: $NEW_PYTHON" >&2
    return 1
  }
  mkdir -p "$NEW_PLAN_ROOT" "$(dirname "$NEW_RESERVATIONS")" || return 1
  if [[ -e "$NEW_RUN_ROOT" ]]; then
    [[ -d "$NEW_RUN_ROOT/.git" && \
       -z "$(git -C "$NEW_RUN_ROOT" status --porcelain --untracked-files=all)" ]] || {
      echo "Fresh run-root collision is not a clean Git checkout." >&2
      return 1
    }
  fi
  if [[ -e "$NEW_PLAN" || -e "$NEW_MATRIX" ]]; then
    [[ -s "$NEW_PLAN" && -s "$NEW_MATRIX" ]] || {
      echo "Fresh plan/matrix publication is incomplete." >&2
      return 1
    }
    jq -e --arg commit "$NEW_COMMIT" --arg campaign "$NEW_CAMPAIGN" \
      --arg reservations "$NEW_RESERVATIONS" '
        .checkout_identity.commit == $commit and
        .campaign == $campaign and
        .reservation_root == $reservations
      ' "$NEW_PLAN" >/dev/null || {
      echo "Existing fresh plan is bound to another launch." >&2
      return 1
    }
  elif [[ -d "$NEW_RESERVATIONS" && \
          -n "$(find "$NEW_RESERVATIONS" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Fresh reservation path is nonempty without its approved plan." >&2
    return 1
  fi

  echo "=== run the complete fresh campaign dry-run before cancellation ==="
  REVIEWED_COMMIT="$NEW_COMMIT" \
  LADDER_CAMPAIGN="$NEW_CAMPAIGN" \
  EVSP_LADDER_SUBMIT=NO \
  EVSP_LADDER_RETRY= \
  EVSP_LADDER_RUN_ROOT="$NEW_RUN_ROOT" \
  EVSP_LADDER_PLAN_ROOT="$NEW_PLAN_ROOT" \
  EVSP_LADDER_RESERVATIONS="$NEW_RESERVATIONS" \
  EVSP_LADDER_PYTHON="$NEW_PYTHON" \
    bash "$LAUNCH_SCRIPT" || {
    echo "Fresh campaign dry-run failed; old scheduler state is untouched." >&2
    return 1
  }

  echo "=== verify and archive the dead campaign ==="
  [[ -s "$OLD_PLAN" && -s "$OLD_MANIFEST" ]] || {
    echo "Old plan/manifest is missing; refusing scheduler changes." >&2
    return 1
  }
  old_plan_hash_line=$(sha256sum "$OLD_PLAN") || return 1
  old_plan_hash=${old_plan_hash_line%% *}
  [[ "$old_plan_hash" == "$OLD_PLAN_SHA" ]] || {
    echo "Old approved-plan hash mismatch; refusing scheduler changes." >&2
    return 1
  }
  jq -e --arg sha "$OLD_PLAN_SHA" --arg commit "$OLD_COMMIT" '
    .checkout_identity.commit == $commit and
    .campaign == "slad_20260819_031808_ba09d46" and
    .campaign_root ==
      "/home/nc437/EVSP-DR-scale-ladder-ba09d46/src/results/scale_ladder/slad_20260819_031808_ba09d46"
  ' "$OLD_PLAN" >/dev/null || {
    echo "Old plan identity mismatch; refusing scheduler changes." >&2
    return 1
  }
  jq -e --arg sha "$OLD_PLAN_SHA" '
    .approval_sha256 == $sha and
    (.gate_job_id | tostring) == "218102" and
    (.submitted_arrays.PREFLIGHT | tostring) == "218103" and
    (.submitted_arrays.SEED | tostring) == "218104" and
    (.submitted_arrays.CG | tostring) == "218105" and
    (.submitted_arrays.CG_SENSITIVITY | tostring) == "218106" and
    (.submitted_arrays.MIP_RAW | tostring) == "218107" and
    (.submitted_arrays.MIP_KNOWN | tostring) == "218108"
  ' "$OLD_MANIFEST" >/dev/null || {
    echo "Old manifest/Slurm binding mismatch; refusing scheduler changes." >&2
    return 1
  }

  declare -a EXPECTED_NAME EXPECTED_PARTITION EXPECTED_RANGE EXPECTED_COMMENT
  declare -a EXPECTED_DEPENDENCY
  EXPECTED_NAME[218102]="LDGbcea6"
  EXPECTED_NAME[218103]="LDPFbcea"
  EXPECTED_NAME[218104]="LDSDbcea"
  EXPECTED_NAME[218105]="LDCGbcea"
  EXPECTED_NAME[218106]="LDCSbcea"
  EXPECTED_NAME[218107]="LDMRbcea"
  EXPECTED_NAME[218108]="LDMKbcea"
  EXPECTED_PARTITION[218102]="default_partition"
  EXPECTED_PARTITION[218103]="default_partition"
  EXPECTED_PARTITION[218104]="default_partition"
  EXPECTED_PARTITION[218105]="default_partition"
  EXPECTED_PARTITION[218106]="default_partition"
  EXPECTED_PARTITION[218107]="scaglione"
  EXPECTED_PARTITION[218108]="scaglione"
  EXPECTED_RANGE[218102]=""
  EXPECTED_RANGE[218103]="0-21"
  EXPECTED_RANGE[218104]="0-20"
  EXPECTED_RANGE[218105]="0-22"
  EXPECTED_RANGE[218106]="0-29"
  EXPECTED_RANGE[218107]="0-20"
  EXPECTED_RANGE[218108]="0-20"
  EXPECTED_COMMENT[218102]="SLADG:${OLD_PLAN_SHA:0:20}"
  EXPECTED_COMMENT[218103]="SLAD:${OLD_PLAN_SHA:0:20}:PREFLIGHT"
  EXPECTED_COMMENT[218104]="SLAD:${OLD_PLAN_SHA:0:20}:SEED"
  EXPECTED_COMMENT[218105]="SLAD:${OLD_PLAN_SHA:0:20}:CG"
  EXPECTED_COMMENT[218106]="SLAD:${OLD_PLAN_SHA:0:20}:CG_SENSITIVITY"
  EXPECTED_COMMENT[218107]="SLAD:${OLD_PLAN_SHA:0:20}:MIP_RAW"
  EXPECTED_COMMENT[218108]="SLAD:${OLD_PLAN_SHA:0:20}:MIP_KNOWN"
  EXPECTED_DEPENDENCY[218102]="afterok:218196,afterok:218197"
  EXPECTED_DEPENDENCY[218103]="afterok:218102"
  EXPECTED_DEPENDENCY[218104]="afterok:218102"
  EXPECTED_DEPENDENCY[218105]="afterok:218102,afterok:218103_*"
  EXPECTED_DEPENDENCY[218106]="afterok:218102,afterok:218103_*"
  EXPECTED_DEPENDENCY[218107]="afterok:218102,aftercorr:218105_*"
  EXPECTED_DEPENDENCY[218108]="afterok:218102,aftercorr:218105_*:218104_*"
  OLD_IDS=(218102 218103 218104 218105 218106 218107 218108)

  validate_old_job() {
    local jid=$1 record rc lines job_field name state partition runtime
    local comment user_id user_number reason dependency
    local observed_canonical_dependency expected_canonical_dependency
    record=$(scontrol show job -o "$jid" 2>&1)
    rc=$?
    [[ "$rc" -eq 0 && -n "$record" ]] || {
      echo "Cannot inspect old job $jid: $record" >&2
      return 1
    }

    if [[ -n "${EXPECTED_RANGE[$jid]}" ]]; then
      validate_pending_array_records \
        "$jid" "${EXPECTED_NAME[$jid]}" "${EXPECTED_PARTITION[$jid]}" \
        "${EXPECTED_RANGE[$jid]}" "${EXPECTED_COMMENT[$jid]}" \
        "${EXPECTED_DEPENDENCY[$jid]}" "$USER" "$record"
      return $?
    fi

    lines=0
    while IFS= read -r line; do
      [[ -n "$line" ]] && lines=$((lines + 1))
    done <<<"$record"
    [[ "$lines" -eq 1 ]] || {
      echo "Old job $jid has no unique controller record." >&2
      return 1
    }
    job_field=$(record_field_value "$record" JobId) || {
      identity_mismatch "$jid" JobId present missing
      return 1
    }
    name=$(record_field_value "$record" JobName) || {
      identity_mismatch "$jid" JobName present missing
      return 1
    }
    state=$(record_field_value "$record" JobState) || {
      identity_mismatch "$jid" JobState present missing
      return 1
    }
    partition=$(record_field_value "$record" Partition) || {
      identity_mismatch "$jid" Partition present missing
      return 1
    }
    runtime=$(record_field_value "$record" RunTime) || {
      identity_mismatch "$jid" RunTime present missing
      return 1
    }
    comment=$(record_field_value "$record" Comment) || {
      identity_mismatch "$jid" Comment present missing
      return 1
    }
    user_id=$(record_field_value "$record" UserId) || {
      identity_mismatch "$jid" UserId present missing
      return 1
    }
    reason=$(record_field_value "$record" Reason) || {
      identity_mismatch "$jid" Reason present missing
      return 1
    }
    dependency=$(record_field_value "$record" Dependency) || {
      identity_mismatch "$jid" Dependency present missing
      return 1
    }

    [[ "$job_field" == "$jid" ]] || {
      identity_mismatch "$jid" JobId "$jid" "$job_field"
      return 1
    }
    [[ "$name" == "${EXPECTED_NAME[$jid]}" ]] || {
      identity_mismatch "$jid" JobName "${EXPECTED_NAME[$jid]}" "$name"
      return 1
    }
    [[ "$state" == "PENDING" ]] || {
      identity_mismatch "$jid" JobState PENDING "$state"
      return 1
    }
    [[ "$runtime" == "00:00:00" ]] || {
      identity_mismatch "$jid" RunTime 00:00:00 "$runtime"
      return 1
    }
    [[ "$partition" == "${EXPECTED_PARTITION[$jid]}" ]] || {
      identity_mismatch "$jid" Partition \
        "${EXPECTED_PARTITION[$jid]}" "$partition"
      return 1
    }
    [[ "$comment" == "${EXPECTED_COMMENT[$jid]}" ]] || {
      identity_mismatch "$jid" Comment "${EXPECTED_COMMENT[$jid]}" "$comment"
      return 1
    }
    if [[ "$user_id" == "$USER("*")" ]]; then
      user_number=${user_id#"$USER("}
      user_number=${user_number%")"}
    else
      user_number=''
    fi
    [[ "$user_number" =~ ^[0-9]+$ ]] || {
      identity_mismatch "$jid" UserId "$USER(<numeric uid>)" "$user_id"
      return 1
    }
    if [[ "$jid" == "218102" ]]; then
      [[ "$reason" == "DependencyNeverSatisfied" ]] || {
        identity_mismatch "$jid" Reason DependencyNeverSatisfied "$reason"
        return 1
      }
    else
      [[ "$reason" == "Dependency" || \
         "$reason" == "DependencyNeverSatisfied" ]] || {
        identity_mismatch "$jid" Reason \
          'Dependency or DependencyNeverSatisfied' "$reason"
        return 1
      }
    fi
    observed_canonical_dependency=$(canonical_dependency "$dependency") || {
      identity_mismatch "$jid" DependencyFormat \
        "${EXPECTED_DEPENDENCY[$jid]}" "$dependency"
      return 1
    }
    expected_canonical_dependency=$(
      canonical_dependency "${EXPECTED_DEPENDENCY[$jid]}"
    ) || return 1
    [[ "$observed_canonical_dependency" == \
       "$expected_canonical_dependency" ]] || {
      identity_mismatch "$jid" Dependency \
        "$expected_canonical_dependency" "$observed_canonical_dependency"
      return 1
    }

    return 0
  }

  expanded_accounting_rows() {
    local jid=$1 raw rc
    raw=$(sacct -X --array -n -P -j "$jid" \
      --format=JobID,JobName,State,Elapsed,ExitCode 2>&1)
    rc=$?
    [[ "$rc" -eq 0 ]] || {
      echo "Cannot query accounting for $jid: $raw" >&2
      return 1
    }
    printf '%s\n' "$raw"
  }

  verify_terminal_probe() {
    local jid=$1 expected_name=$2 expected_partition=$3 artifact=$4
    local row row_id row_name row_state row_elapsed row_exit normalized
    [[ -f "$artifact" && ! -L "$artifact" && \
       -f "$artifact.sha256" && ! -L "$artifact.sha256" ]] || {
      echo "Probe artifact/sidecar missing or unsafe: $artifact" >&2
      return 1
    }
    (
      cd "$(dirname "$artifact")" || exit 1
      sha256sum -c "$(basename "$artifact").sha256"
    ) || return 1
    jq -e --arg jid "$jid" --arg partition "$expected_partition" '
      .slurm_job_id == $jid and
      .slurm_partition == $partition and
      .compatible == false and
      (.differences | length) == 1 and
      .differences[0].field == "portable.numpy_build"
    ' "$artifact" >/dev/null || return 1
    row=$(expanded_accounting_rows "$jid") || return 1
    [[ "$row" != *$'\n'* ]] || {
      echo "Scalar probe $jid has ambiguous accounting rows." >&2
      return 1
    }
    IFS='|' read -r row_id row_name row_state row_elapsed row_exit <<<"$row"
    normalized=${row_state%% *}
    normalized=${normalized%%+*}
    [[ "$row_id" == "$jid" && "$row_name" == "$expected_name" && \
       "$normalized" == "FAILED" && "$row_exit" == "3:0" ]] || {
      echo "Probe $jid is not the exact terminal failed probe." >&2
      return 1
    }
    return 0
  }

  echo "=== prove old probe writers are terminal and immutable ==="
  verify_terminal_probe 218196 "LDPDE2bce" "default_partition" \
    "$OLD_ROOT/probes/default_partition.attempt2.json" || return 1
  verify_terminal_probe 218197 "LDPSC2bce" "scaglione" \
    "$OLD_ROOT/probes/scaglione.attempt2.json" || return 1

  cancelled_accounting_entry() {
    local jid=$1 raw
    raw=$(expanded_accounting_rows "$jid") || return 1
    if [[ -z "${EXPECTED_RANGE[$jid]}" ]]; then
      cancelled_scalar_entry_from_rows \
        "$jid" "${EXPECTED_NAME[$jid]}" "$raw" || {
        echo "Scalar job $jid lacks complete zero-runtime cancellation." >&2
        return 1
      }
    else
      cancelled_array_entry_from_rows \
        "$jid" "${EXPECTED_NAME[$jid]}" "${EXPECTED_RANGE[$jid]}" "$raw" || {
        echo "Array $jid lacks complete per-task zero-runtime cancellation." >&2
        return 1
      }
    fi
  }

  verify_cancelled_accounting() {
    local jid entry
    CANCELLED_JOBS_JSON='[]'
    for jid in "${OLD_IDS[@]}"; do
      entry=$(cancelled_accounting_entry "$jid") || return 1
      CANCELLED_JOBS_JSON=$(jq -c --argjson entry "$entry" \
        '. + [$entry]' <<<"$CANCELLED_JOBS_JSON") || return 1
    done
    return 0
  }

  RECEIPT_DIR="$OLD_ROOT/cancellation_receipt_v1.bundle"
  RECEIPT="$RECEIPT_DIR/receipt.json"
  verify_receipt() {
    [[ -d "$RECEIPT_DIR" && ! -L "$RECEIPT_DIR" && \
       -f "$RECEIPT" && ! -L "$RECEIPT" && \
       -f "$RECEIPT.sha256" && ! -L "$RECEIPT.sha256" ]] || return 1
    (
      cd "$(dirname "$RECEIPT")" || exit 1
      sha256sum -c "$(basename "$RECEIPT").sha256"
    ) || return 1
    jq -e --arg sha "$OLD_PLAN_SHA" \
      --argjson jobs "$CANCELLED_JOBS_JSON" '
      .schema == "evsp-dr-scale-ladder-cancellation-v1" and
      .plan_sha256 == $sha and
      .jobs == $jobs
    ' "$RECEIPT" >/dev/null
  }

  write_receipt() {
    local temporary_dir="$RECEIPT_DIR.tmp.$$"
    local temporary_receipt="$temporary_dir/receipt.json"
    [[ ! -e "$RECEIPT_DIR" && ! -e "$temporary_dir" ]] || return 1
    mkdir "$temporary_dir" || return 1
    jq -n --arg sha "$OLD_PLAN_SHA" --arg commit "$OLD_COMMIT" \
      --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      --argjson jobs "$CANCELLED_JOBS_JSON" '{
        schema:"evsp-dr-scale-ladder-cancellation-v1",
        plan_sha256:$sha,
        source_commit:$commit,
        cancelled_at_utc:$timestamp,
        jobs:$jobs
      }' >"$temporary_receipt" || return 1
    (
      cd "$temporary_dir" || exit 1
      sha256sum receipt.json >receipt.json.sha256 &&
        sha256sum -c receipt.json.sha256
    ) || return 1
    mv "$temporary_dir" "$RECEIPT_DIR" || return 1
    verify_receipt
  }

  LIVE_BASE_IDS=$(squeue -h -u "$USER" -o '%F') || return 1
  ACTIVE_OLD=()
  ABSENT_OLD=()
  for jid in "${OLD_IDS[@]}"; do
    found_live=0
    while IFS= read -r live_id; do
      if [[ "$live_id" == "$jid" ]]; then
        ACTIVE_OLD+=("$jid")
        found_live=1
        break
      fi
    done <<<"$LIVE_BASE_IDS"
    if [[ "$found_live" -eq 0 ]]; then
      ABSENT_OLD+=("$jid")
    fi
  done

  echo "=== classify every old job independently ==="
  for jid in "${ACTIVE_OLD[@]}"; do
    validate_old_job "$jid" || {
      echo "Live old job $jid failed its exact identity check." >&2
      return 1
    }
  done
  for jid in "${ABSENT_OLD[@]}"; do
    cancelled_accounting_entry "$jid" >/dev/null || {
      echo "Absent old job $jid lacks complete zero-runtime cancellation proof." >&2
      return 1
    }
  done

  ensure_pre_cancel_archive() {
    local closeout_dir="$OLD_ROOT/pre_cancel_closeout_v1.bundle"
    local closeout_tmp="$closeout_dir.tmp.$$"
    local archive_root="$HOME/evsp_archives"
    local archive_bundle="$archive_root/stalled_${OLD_CAMPAIGN}_pre_cancel_v1.bundle"
    local archive_tmp="$archive_bundle.tmp.$$"
    local archive_name="stalled_${OLD_CAMPAIGN}_pre_cancel_v1.tar.gz"
    local old_external_plan="$HOME/evsp_scale_ladder_plans/$OLD_CAMPAIGN.plan.json"
    local old_external_matrix="$HOME/evsp_scale_ladder_plans/$OLD_CAMPAIGN.tasks.csv"
    local reservation_root reservation_count reservation_paths reservation_path
    local basename jid file

    if [[ -e "$archive_bundle" ]]; then
      [[ -d "$archive_bundle" && ! -L "$archive_bundle" && \
         -f "$archive_bundle/$archive_name" && \
         -f "$archive_bundle/$archive_name.sha256" ]] || return 1
      (
        cd "$archive_bundle" || exit 1
        sha256sum -c "$archive_name.sha256"
      ) || return 1
      PRE_CANCEL_ARCHIVE="$archive_bundle/$archive_name"
      return 0
    fi

    [[ ! -e "$closeout_tmp" && ! -e "$archive_tmp" ]] || return 1
    if [[ ! -e "$closeout_dir" ]]; then
      mkdir "$closeout_tmp" || return 1
      squeue -h -u "$USER" \
        -o '%F|%i|%j|%T|%M|%R' >"$closeout_tmp/squeue.tsv" || return 1
      for jid in 218196 218197 "${OLD_IDS[@]}"; do
        if scontrol show job -dd -o "$jid" \
            >"$closeout_tmp/scontrol_${jid}.txt" 2>&1; then
          :
        else
          printf 'scontrol record unavailable for terminal job %s\n' "$jid" \
            >"$closeout_tmp/scontrol_${jid}.txt"
        fi
        expanded_accounting_rows "$jid" \
          >"$closeout_tmp/sacct_${jid}.tsv" || return 1
      done
      if [[ -s "$old_external_plan" ]]; then
        external_plan_hash_line=$(sha256sum "$old_external_plan") || return 1
        external_plan_hash=${external_plan_hash_line%% *}
        [[ "$external_plan_hash" == "$OLD_PLAN_SHA" ]] || return 1
        cp "$old_external_plan" "$closeout_tmp/external_plan.json" || return 1
      else
        printf 'absent; canonical copy is campaign/approved-plan.json\n' \
          >"$closeout_tmp/external_plan.absent.txt" || return 1
      fi
      if [[ -s "$old_external_matrix" ]]; then
        cp "$old_external_matrix" "$closeout_tmp/external_tasks.csv" || return 1
      else
        printf 'absent; canonical job records remain in approved-plan.json\n' \
          >"$closeout_tmp/external_tasks.absent.txt" || return 1
      fi
      reservation_root=$(jq -er '.reservation_root' "$OLD_PLAN") || return 1
      [[ "$reservation_root" == /* && -d "$reservation_root" && \
         ! -L "$reservation_root" ]] || return 1
      jq -e --arg root "$reservation_root" '
        ((.reservations // []) | length) == .task_count and
        ([.reservations[] | select(startswith($root + "/"))] | length)
          == .task_count and
        ([.reservations[] | split("/")[-1] | rtrimstr(".json")] | sort)
          == ([.jobs[].execution_digest] | sort)
      ' "$OLD_MANIFEST" >/dev/null || return 1
      reservation_count=$(jq -r '(.reservations // []) | length' \
        "$OLD_MANIFEST") || return 1
      : >"$closeout_tmp/reservation_basenames.txt" || return 1
      reservation_paths=$(jq -r '(.reservations // [])[]' \
        "$OLD_MANIFEST") || return 1
      while IFS= read -r reservation_path; do
        [[ -n "$reservation_path" ]] || continue
        [[ "$reservation_path" == "$reservation_root/"* && \
           -f "$reservation_path" && ! -L "$reservation_path" ]] || return 1
        basename=${reservation_path##*/}
        [[ "$basename" =~ ^[0-9a-f]{64}\.json$ ]] || return 1
        printf '%s\n' "$basename" \
          >>"$closeout_tmp/reservation_basenames.txt" || return 1
      done <<<"$reservation_paths"
      [[ "$(awk 'END {print NR+0}' "$closeout_tmp/reservation_basenames.txt")" \
          -eq "$reservation_count" ]] || return 1
      if [[ "$reservation_count" -gt 0 ]]; then
        tar -C "$reservation_root" -czf \
          "$closeout_tmp/reservations.tar.gz" \
          -T "$closeout_tmp/reservation_basenames.txt" || return 1
      else
        printf 'No reservations recorded in campaign manifest.\n' \
          >"$closeout_tmp/reservations.none.txt" || return 1
      fi
      jq -n --arg plan_sha "$OLD_PLAN_SHA" --arg commit "$OLD_COMMIT" \
        --arg captured "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg active "${ACTIVE_OLD[*]}" --arg absent "${ABSENT_OLD[*]}" \
        --argjson reservations "$reservation_count" '{
          schema:"evsp-dr-scale-ladder-pre-cancel-closeout-v1",
          plan_sha256:$plan_sha,source_commit:$commit,captured_at_utc:$captured,
          active_job_ids:($active|split(" ")|map(select(length>0))),
          already_absent_job_ids:($absent|split(" ")|map(select(length>0))),
          manifest_reservation_count:$reservations
        }' >"$closeout_tmp/metadata.json" || return 1
      (
        cd "$closeout_tmp" || exit 1
        : >SHA256SUMS.tmp || exit 1
        for file in *; do
          [[ -f "$file" && "$file" != "SHA256SUMS.tmp" ]] || continue
          sha256sum "$file" >>SHA256SUMS.tmp || exit 1
        done
        mv SHA256SUMS.tmp SHA256SUMS && sha256sum -c SHA256SUMS
      ) || return 1
      mv "$closeout_tmp" "$closeout_dir" || return 1
    else
      [[ -d "$closeout_dir" && ! -L "$closeout_dir" && \
         -f "$closeout_dir/SHA256SUMS" ]] || return 1
      (cd "$closeout_dir" && sha256sum -c SHA256SUMS) || return 1
    fi

    mkdir -p "$archive_root" "$archive_tmp" || return 1
    tar -C "$OLD_RUN_ROOT" -czf "$archive_tmp/$archive_name" \
      "src/results/scale_ladder/$OLD_CAMPAIGN" || return 1
    (
      cd "$archive_tmp" || exit 1
      sha256sum "$archive_name" >"$archive_name.sha256" &&
        sha256sum -c "$archive_name.sha256"
    ) || return 1
    mv "$archive_tmp" "$archive_bundle" || return 1
    PRE_CANCEL_ARCHIVE="$archive_bundle/$archive_name"
    return 0
  }

  echo "=== preserve a complete pre-cancel evidence archive ==="
  ensure_pre_cancel_archive || {
    echo "Pre-cancel evidence archive failed; no scheduler mutation performed." >&2
    return 1
  }

  if [[ "${#ACTIVE_OLD[@]}" -gt 0 ]]; then
    [[ ! -e "$RECEIPT_DIR" ]] || {
      echo "A full cancellation receipt exists while old jobs remain live." >&2
      return 1
    }
    echo "=== cancel only the remaining exact zero-runtime dead jobs ==="
    for jid in 218103 218104 218105 218106 218107 218108 218102; do
      should_cancel=0
      for active_id in "${ACTIVE_OLD[@]}"; do
        [[ "$active_id" == "$jid" ]] && should_cancel=1
      done
      [[ "$should_cancel" -eq 1 ]] || continue
      validate_old_job "$jid" || {
        echo "Old job $jid changed before cancellation; fresh launch refused." >&2
        return 1
      }
      if ! scancel "$jid"; then
        echo "scancel reported failure for $jid; checking final state." >&2
      fi
    done
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
      sleep 1
      LIVE_BASE_IDS=$(squeue -h -u "$USER" -o '%F') || return 1
      remaining=0
      for jid in "${OLD_IDS[@]}"; do
        while IFS= read -r live_id; do
          if [[ "$live_id" == "$jid" ]]; then
            remaining=$((remaining + 1))
            break
          fi
        done <<<"$LIVE_BASE_IDS"
      done
      [[ "$remaining" -eq 0 ]] && break
    done
    [[ "$remaining" -eq 0 ]] || {
      echo "One or more old jobs remain live; fresh launch refused." >&2
      return 1
    }
    wait_for_cancelled_accounting || return 1
  else
    echo "All seven old jobs are absent; proving zero-runtime cancellation."
    verify_cancelled_accounting || return 1
  fi

  if [[ -e "$RECEIPT_DIR" ]]; then
    verify_receipt || {
      echo "Existing cancellation receipt is invalid; launch refused." >&2
      return 1
    }
  else
    write_receipt || return 1
  fi
  verify_receipt || return 1

  echo "=== launch the corrected probe-first scale ladder ==="
  REVIEWED_COMMIT="$NEW_COMMIT" \
  LADDER_CAMPAIGN="$NEW_CAMPAIGN" \
  EVSP_LADDER_SUBMIT=YES \
  EVSP_LADDER_RETRY= \
  EVSP_LADDER_RUN_ROOT="$NEW_RUN_ROOT" \
  EVSP_LADDER_PLAN_ROOT="$NEW_PLAN_ROOT" \
  EVSP_LADDER_RESERVATIONS="$NEW_RESERVATIONS" \
  EVSP_LADDER_PYTHON="$NEW_PYTHON" \
    bash "$LAUNCH_SCRIPT"
  status=$?
  echo "OLD_ARCHIVE=$PRE_CANCEL_ARCHIVE"
  if [[ "$status" -ne 0 ]]; then
    echo "Fresh launch stopped fail-closed; paste this block's output." >&2
    return "$status"
  fi
  echo "FRESH_SCALE_LADDER_LAUNCH_ACCEPTED=true"
  return 0
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main
fi
