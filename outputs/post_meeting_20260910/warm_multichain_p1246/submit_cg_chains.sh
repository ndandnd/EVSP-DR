#!/bin/bash
set -euo pipefail
fatal() { echo "[submit inherited multichain CG] $*" >&2; exit 2; }
BASE=/home/nc437/ladder-lite
BATCH=$BASE/nested_warm_multichain_p1246_k2_10_20260910_ecb60c1
EXECUTION=$BASE/execution/ecb60c154a9a5db385e3a573949ec9fd0a737af3
COMMIT=ecb60c154a9a5db385e3a573949ec9fd0a737af3
WORKER_SHA=ed5f81bddd419000b0a0c426ef664a9e3a60a2f46a38921d30cfa3886a350610
RECORD="$BATCH/cg_jobs.tsv"
DRY_RUN="${EVSP_DRY_RUN:-1}"
SBATCH="${EVSP_SBATCH:-/usr/local/slurm/slurm-25.05.5/bin/sbatch}"
[[ -d "$BATCH" && -f "$BATCH/batch_manifest.json" ]] || fatal "batch not prepared"
[[ -x "$SBATCH" ]] || fatal "sbatch unavailable: $SBATCH"
[[ "$(git -C "$EXECUTION" rev-parse HEAD)" == "$COMMIT" ]] || fatal "execution commit mismatch"
[[ -z "$(git -C "$EXECUTION" status --porcelain --untracked-files=no)" ]] || fatal "execution checkout dirty"
[[ ! -e "$RECORD" && ! -e "${RECORD}.planned" ]] || fatal "CG record exists; refusing duplicate submission"
mkdir -p "$BATCH/locks"
exec 9>>"$BATCH/locks/submit-cg.lock"
flock -n 9 || fatal "another CG submission is active"
if [[ "$DRY_RUN" == 1 ]]; then
  command_file="$BATCH/dryrun_cg_commands.sh"
  : > "$command_file"; chmod 700 "$command_file"
fi
printf 'replicate\tscale\tjob_id\tdependency\tpartition\texclude\tcpus\tmem\ttime\trequeue\tinherit_workers\tworker_sha256\n' > "${RECORD}.planned"
for replicate in 1 2 4 6; do
  campaign="$BATCH/p$replicate"
  worker="$campaign/warm_chain_cg.sub"
  [[ -f "$worker" ]] || fatal "worker missing: $worker"
  [[ "$(sha256sum "$worker" | cut -d' ' -f1)" == "$WORKER_SHA" ]] || fatal "worker hash mismatch: $worker"
  dependency=""
  for scale in 2 3 4 5 6 7 8 9 10; do
    job_name=$(printf 'wi%dk%02d' "$replicate" "$scale")
    args=(
      "$SBATCH" --parsable --partition=default_partition --exclude=scaglione-compute-01
      --cpus-per-task=8 --mem=96G --time=08:15:00
      --requeue --signal=B:USR1@120 --job-name="$job_name"
      --output="$campaign/logs/cg/%j.out" --error="$campaign/logs/cg/%j.err"
      --export=ALL,EVSP_EXECUTION_REPO="$EXECUTION",EVSP_EXPECTED_COMMIT="$COMMIT",EVSP_CAMPAIGN_ROOT="$campaign",EVSP_SCALE="$scale",EVSP_REPLICATE="$replicate",EVSP_WORKER_SHA256="$WORKER_SHA",EVSP_MAX_ITERS=50000,EVSP_WALL_LIMIT_S=28800,EVSP_INHERIT_WORKERS=8
    )
    [[ -z "$dependency" ]] || args+=(--dependency="$dependency")
    args+=("$worker")
    if [[ "$DRY_RUN" == 1 ]]; then
      printf '%q ' "${args[@]}" >> "$command_file"; printf '\n' >> "$command_file"
      job_id="DRYRUN_P${replicate}_K${scale}"
    else
      job_id=$("${args[@]}")
      [[ "$job_id" =~ ^[0-9]+$ ]] || fatal "unexpected sbatch id: $job_id"
    fi
    printf '%s\t%s\t%s\t%s\tdefault_partition\tscaglione-compute-01\t8\t96G\t08:15:00\ttrue\t8\t%s\n' \
      "$replicate" "$scale" "$job_id" "${dependency:-none}" "$WORKER_SHA" >> "${RECORD}.planned"
    dependency="afterok:$job_id"
  done
done
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY_RUN=1; commands written to $command_file"
  cat "${RECORD}.planned"
  exit 0
fi
mv "${RECORD}.planned" "$RECORD"
sha256sum "$RECORD" > "$RECORD.sha256"
"$HOME/evsp_env/bin/python" "$BATCH/launch/collect_status.py" "$BATCH" --output "$BATCH/status_after_cg_submission.json"
cat "$RECORD"
