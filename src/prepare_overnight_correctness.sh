#!/bin/bash
# Prepare and submit the post-fix evidence campaign from an allocated compute
# node. Users should not invoke this heavy helper directly.
#
#   LA = raw-master audit of the latest terminal pool for each big run
#   MC = fleet-first MIP at 1/3/6/10/15 CG hours for six controls
#        plus a 15m/3h/6h MIP-budget ladder on each six-hour pool
#   CC = isolated no-stall continuation of those controls from 6h to 72h
#
# Idempotent: finished outputs and active semantic job names are skipped.
# It never writes to a live tariff_big/exact_big journal.

set -euo pipefail

ROOT="${EVSP_DR_ROOT:-$HOME/EVSP-DR}"
cd "$ROOT"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "[LAUNCH] refuse to scan/hash/archive on a login node." >&2
  echo "[LAUNCH] use: bash src/launch_overnight_correctness.sh" >&2
  exit 2
fi

if [ ! -d data/duty_unions_big ]; then
  echo "[LAUNCH] data/duty_unions_big is missing; copy the exact cluster corpus, do not regenerate it." >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[LAUNCH] tracked worktree changes exist; pull/resolve them before launching." >&2
  exit 2
fi
if [ "${EVSP_DRY_RUN:-0}" != "1" ] && ! command -v sbatch >/dev/null 2>&1; then
  echo "[LAUNCH] sbatch is unavailable; run this on Unicorn." >&2
  exit 2
fi

mkdir -p src/cluster_logs/overnight_correctness \
  src/results/master_audit \
  src/results/stopping_mip \
  src/results/stopping_controls \
  src/results/campaign_manifests

CAMPAIGN_ID="${EVSP_CAMPAIGN_ID:-job${SLURM_JOB_ID}}"
COMMIT="${EVSP_EXPECTED_COMMIT:-$(git rev-parse HEAD)}"
if [ "$(git rev-parse HEAD)" != "$COMMIT" ]; then
  echo "[LAUNCH] checkout drifted from campaign commit $COMMIT" >&2
  exit 2
fi
MANIFEST="src/results/campaign_manifests/overnight_correctness_${CAMPAIGN_ID}.tsv"
INPUT_ARCHIVE="src/results/campaign_manifests/duty_unions_big_${CAMPAIGN_ID}.tar.gz"
POOL_INPUT_LIST="src/results/campaign_manifests/pool_inputs_${CAMPAIGN_ID}.txt"
POOL_INPUT_ARCHIVE="src/results/campaign_manifests/pool_inputs_${CAMPAIGN_ID}.tar.gz"
: > "$POOL_INPUT_LIST"
{
  echo "# commit=${COMMIT}"
  echo "# campaign_id=${CAMPAIGN_ID}"
  echo "# prep_job_id=${SLURM_JOB_ID}"
  echo -e "family\tjob_name\tjob_id\tallocation_time\tsource\tsource_sha256\tsource_journal\tsource_journal_sha256\toutput\tstate"
} > "$MANIFEST"

submitted=0
skipped=0
declare -A SOURCE_INFO_CACHE
declare -A SOURCE_INFO_STATE
SOURCE_INFO_RESULT=""

source_info_for() {
  local source=$1
  local info
  SOURCE_INFO_RESULT=""
  if [ -n "${SOURCE_INFO_STATE[$source]+present}" ]; then
    if [ "${SOURCE_INFO_STATE[$source]}" = "ok" ]; then
      SOURCE_INFO_RESULT=${SOURCE_INFO_CACHE[$source]}
      return 0
    fi
    SOURCE_INFO_RESULT=${SOURCE_INFO_CACHE[$source]:-INVALID}
    return 2
  fi
  if info=$(python src/campaign_artifact_status.py \
      --source "$source" --print-source --require-terminal); then
    SOURCE_INFO_STATE[$source]=ok
    SOURCE_INFO_CACHE[$source]=$info
    SOURCE_INFO_RESULT=$info
    return 0
  fi
  SOURCE_INFO_STATE[$source]=invalid
  SOURCE_INFO_CACHE[$source]=$info
  SOURCE_INFO_RESULT=$info
  return 2
}

add_pool_input() {
  local path=$1
  local path_abs path_rel
  path_abs=$(realpath "$path")
  case "$path_abs" in
    "$ROOT"/*) path_rel=${path_abs#"$ROOT/"} ;;
    *) echo "[LAUNCH] archive input is outside repository: $path_abs" >&2; exit 2 ;;
  esac
  printf '%s\n' "$path_rel" >> "$POOL_INPUT_LIST"
}

submit_one() {
  family=$1
  semantic_job_name=$2
  job_name=$semantic_job_name
  worker=$3
  source=$4
  output=$5
  allocation_time=$6
  shift 6
  if ! source_info_for "$source"; then
    reason=${SOURCE_INFO_RESULT//$'\t'/ }
    reason=${reason//$'\n'/ }
    echo -e "${family}\t${job_name}\t-\t${allocation_time}\t${source}\t-\t-\t-\t${output}\tSKIP_INVALID ${reason}" >> "$MANIFEST"
    echo "[LAUNCH] ${job_name} skipped: invalid source ${source}" >&2
    skipped=$((skipped + 1))
    return
  fi
  source_info=$SOURCE_INFO_RESULT
  IFS=$'\t' read -r result_sha source_journal journal_sha source_stop \
    <<< "$source_info"
  # Bind queue identity to both immutable source bytes and campaign code.
  # A prep requeue may discover a newer source for the same semantic cell;
  # it must not confuse that with an older active job.
  job_name="${semantic_job_name}-r${result_sha:0:6}j${journal_sha:0:6}c${COMMIT:0:6}"
  add_pool_input "$source"
  add_pool_input "$source_journal"

  if [ -s "$output" ] && python src/campaign_artifact_status.py \
      --family "$family" --source "$source" --output "$output" \
      --total-wall-s 259200 --expected-commit "$COMMIT"; then
    echo -e "${family}\t${job_name}\t-\t${allocation_time}\t${source}\t${result_sha}\t${source_journal}\t${journal_sha}\t${output}\tSKIP_COMPLETE" >> "$MANIFEST"
    skipped=$((skipped + 1))
    return
  fi
  active_job=""
  if [ "${EVSP_DRY_RUN:-0}" != "1" ]; then
    active_job=$(squeue -h -u "$USER" -n "$job_name" -o '%A' 2>/dev/null || true)
  fi
  if [ -n "$active_job" ]; then
    active_job=${active_job%%$'\n'*}
    echo -e "${family}\t${job_name}\t${active_job}\t${allocation_time}\t${source}\t${result_sha}\t${source_journal}\t${journal_sha}\t${output}\tSKIP_ACTIVE" >> "$MANIFEST"
    skipped=$((skipped + 1))
    return
  fi
  if [ "${EVSP_DRY_RUN:-0}" = "1" ]; then
    job_id="DRYRUN"
  else
    job_id=$(sbatch --parsable --job-name "$job_name" \
      --time "$allocation_time" "$worker" "$@" "$COMMIT")
    job_id=${job_id%%;*}
  fi
  echo -e "${family}\t${job_name}\t${job_id}\t${allocation_time}\t${source}\t${result_sha}\t${source_journal}\t${journal_sha}\t${output}\tSUBMITTED" >> "$MANIFEST"
  echo "[LAUNCH] ${job_name} -> ${job_id}"
  submitted=$((submitted + 1))
}

archive_baseline_control() {
  local source=$1
  local label=$2
  local source_info result_sha source_journal journal_sha source_stop iters
  if [ ! -s "$source" ]; then
    echo "# baseline_${label}=UNAVAILABLE_NO_STATUS" >> "$MANIFEST"
    return
  fi
  if ! source_info_for "$source" 2>/dev/null; then
    echo "# baseline_${label}=UNAVAILABLE_NONTERMINAL_OR_INVALID" >> "$MANIFEST"
    return
  fi
  source_info=$SOURCE_INFO_RESULT
  IFS=$'\t' read -r result_sha source_journal journal_sha source_stop \
    <<< "$source_info"
  add_pool_input "$source"
  add_pool_input "$source_journal"
  iters="${source}.iters.csv"
  if [ ! -s "$iters" ]; then
    echo "# baseline_${label}=UNAVAILABLE_NO_ITERS status=${source} status_sha256=${result_sha} journal=${source_journal} journal_sha256=${journal_sha} stop=${source_stop}" >> "$MANIFEST"
    return
  fi
  add_pool_input "$iters"
  echo "# baseline_${label}=ARCHIVED status=${source} status_sha256=${result_sha} journal=${source_journal} journal_sha256=${journal_sha} iters=${iters} stop=${source_stop}" >> "$MANIFEST"
}

NAMES=(
  Practice_Custom_DutyUnion_k30_r1 Practice_Custom_DutyUnion_k30_r2
  Practice_Custom_DutyUnion_k30_r3 Practice_Custom_DutyUnion_k30_r4
  Practice_Custom_DutyUnion_k30_r5 Practice_Custom_DutyUnion_k30_r6
  Practice_Custom_DutyUnion_k40_r1 Practice_Custom_DutyUnion_k40_r2
  Practice_Custom_DutyUnion_k40_r3 Practice_Custom_DutyUnion_k40_r4
)
SHORTS=(30r1 30r2 30r3 30r4 30r5 30r6 40r1 40r2 40r3 40r4)
TAGS=(peak08 peak12 peak18 sek)
TAG_SHORTS=(p08 p12 p18 sek)

LATEST_TERMINAL_POOL_RESULT=""
latest_terminal_pool() {
  base=$1
  LATEST_TERMINAL_POOL_RESULT=""
  for candidate in \
    "${base}.json" \
    "${base}.m900.snapshot.json" \
    "${base}.m600.snapshot.json" \
    "${base}.m360.snapshot.json" \
    "${base}.m180.snapshot.json" \
    "${base}.m60.snapshot.json"; do
    if [ -s "$candidate" ] && source_info_for "$candidate" 2>/dev/null; then
      LATEST_TERMINAL_POOL_RESULT=$candidate
      return 0
    fi
  done
  return 1
}

FLAT_POOL_RESULT=""
flat_pool() {
  name=$1
  base="src/results/exact_big/${name}_soc15_b10"
  FLAT_POOL_RESULT=""
  for candidate in \
    "${base}_g300_res0.0.json" "${base}_g300.json" "${base}.json"; do
    if [ -s "$candidate" ] && source_info_for "$candidate" 2>/dev/null; then
      FLAT_POOL_RESULT=$candidate
      return 0
    fi
  done
  return 1
}

# Raw-master gate: ten flat pools plus up to forty non-flat immutable pools.
for i in "${!NAMES[@]}"; do
  name=${NAMES[$i]}
  short=${SHORTS[$i]}
  if flat_pool "$name"; then
    pool=$FLAT_POOL_RESULT
    out="src/results/master_audit/$(basename "$pool" .json).master_audit.json"
    submit_one LA "LA${short}flt" src/submit_master_audit.sub \
      "$pool" "$out" "02:00:00" "$pool" "$out"
  fi
  for t in "${!TAGS[@]}"; do
    tag=${TAGS[$t]}
    tag_short=${TAG_SHORTS[$t]}
    base="src/results/tariff_big/${name}_${tag}"
    if latest_terminal_pool "$base"; then
      pool=$LATEST_TERMINAL_POOL_RESULT
      out="src/results/master_audit/$(basename "$pool" .json).master_audit.json"
      submit_one LA "LA${short}${tag_short}" src/submit_master_audit.sub \
        "$pool" "$out" "02:00:00" "$pool" "$out"
    fi
  done
done

# Six stratified stopping-rule controls from the existing 40-task map.
CONTROL_NAMES=(
  Practice_Custom_DutyUnion_k30_r5 Practice_Custom_DutyUnion_k30_r3
  Practice_Custom_DutyUnion_k40_r1 Practice_Custom_DutyUnion_k40_r2
  Practice_Custom_DutyUnion_k40_r3 Practice_Custom_DutyUnion_k40_r4
)
CONTROL_SHORTS=(30r5 30r3 40r1 40r2 40r3 40r4)
CONTROL_TAGS=(peak12 sek peak12 sek peak08 peak18)
CONTROL_TAG_SHORTS=(p12 sek p12 sek p08 p18)
MARKS=(60 180 360 600 900)
HOURS=(01 03 06 10 15)

for c in "${!CONTROL_NAMES[@]}"; do
  name=${CONTROL_NAMES[$c]}
  short=${CONTROL_SHORTS[$c]}
  tag=${CONTROL_TAGS[$c]}
  tag_short=${CONTROL_TAG_SHORTS[$c]}
  base="src/results/tariff_big/${name}_${tag}"
  # Preserve the old canonical trajectory only when it is already terminal.
  # Otherwise this campaign's stopping comparison is explicitly forward-only
  # from the immutable six-hour snapshot.
  archive_baseline_control "${base}.json" "${short}_${tag_short}"
  for m in "${!MARKS[@]}"; do
    mark=${MARKS[$m]}
    snapshot="${base}.m${mark}.snapshot.json"
    if [ ! -s "$snapshot" ]; then continue; fi
    out="src/results/stopping_mip/${name}_${tag}_m${mark}_twostage.json"
    submit_one MC "MC${short}${tag_short}h${HOURS[$m]}" \
      src/submit_snapshot_pool_mip.sub "$snapshot" "$out" "01:20:00" \
      "$snapshot" "$out" 3600
  done

  # Fixed six-hour CG pool, varying only the fleet-MIP budget.  The one-hour
  # cell already exists in the CG-age curve above.
  snapshot="${base}.m360.snapshot.json"
  if [ -s "$snapshot" ]; then
    MIP_BUDGETS=(900 10800 21600)
    MIP_LABELS=(015m 03h 06h)
    MIP_ALLOCS=(00:35:00 03:20:00 06:20:00)
    for b in "${!MIP_BUDGETS[@]}"; do
      out="src/results/stopping_mip/${name}_${tag}_m360_budget_${MIP_LABELS[$b]}.json"
      submit_one MC "MB${short}${tag_short}${MIP_LABELS[$b]}" \
        src/submit_snapshot_pool_mip.sub "$snapshot" "$out" \
        "${MIP_ALLOCS[$b]}" "$snapshot" "$out" "${MIP_BUDGETS[$b]}"
    done
  fi

  snapshot="${base}.m360.snapshot.json"
  if [ -s "$snapshot" ]; then
    out="src/results/stopping_controls/${name}_${tag}_from_m360_nostall.json"
    submit_one CC "CC${short}${tag_short}" \
      src/submit_cg_snapshot_control.sub "$snapshot" "$out" "3-02:00:00" \
      "$snapshot" 360 "$out"
  fi
done

sort -u "$POOL_INPUT_LIST" -o "$POOL_INPUT_LIST"
if [ ! -s "$POOL_INPUT_LIST" ]; then
  echo "[LAUNCH] no eligible immutable/terminal source pools were found" >&2
  exit 2
fi
echo "[LAUNCH] worker submissions complete; archiving inputs on compute job ${SLURM_JOB_ID}"
INPUT_PARTIAL="${INPUT_ARCHIVE}.partial.${SLURM_JOB_ID}"
POOL_PARTIAL="${POOL_INPUT_ARCHIVE}.partial.${SLURM_JOB_ID}"
tar -czf "$INPUT_PARTIAL" data/duty_unions_big
mv -f "$INPUT_PARTIAL" "$INPUT_ARCHIVE"
INPUT_SHA=$(sha256sum "$INPUT_ARCHIVE" | awk '{print $1}')
tar -czf "$POOL_PARTIAL" -T "$POOL_INPUT_LIST"
mv -f "$POOL_PARTIAL" "$POOL_INPUT_ARCHIVE"
POOL_INPUT_SHA=$(sha256sum "$POOL_INPUT_ARCHIVE" | awk '{print $1}')
{
  echo "# input_archive=${INPUT_ARCHIVE}"
  echo "# input_archive_sha256=${INPUT_SHA}"
  echo "# pool_input_archive=${POOL_INPUT_ARCHIVE}"
  echo "# pool_input_archive_sha256=${POOL_INPUT_SHA}"
} >> "$MANIFEST"

echo "[LAUNCH] submitted=${submitted} skipped=${skipped}"
echo "[LAUNCH] manifest=${MANIFEST}"
echo "[LAUNCH] input archive sha256=${INPUT_SHA}"
echo "[LAUNCH] pool input archive sha256=${POOL_INPUT_SHA}"
if [ "${EVSP_DRY_RUN:-0}" != "1" ]; then
  squeue --me -o '%.14i %.42j %.2t %.10M %R' | sed -n '1,60p'
fi
