#!/bin/bash
set -euo pipefail
fatal() { echo "[submit warm downstream] $*" >&2; exit 2; }
BASE=/home/nc437/ladder-lite
BATCH=$BASE/nested_warm_multichain_p1246_k2_10_20260910_ecb60c1
MIP_REPO=$BASE/execution/871d057e1067411f09581e37d78f7c1ca43f68bb
MIP_COMMIT=871d057e1067411f09581e37d78f7c1ca43f68bb
SOURCE_COMMIT=ecb60c154a9a5db385e3a573949ec9fd0a737af3
FREEZE_ONE_SHA=472f18bf49f8a3e4dcd8ece66d87db1d41d5ccf4a3c1ead42fb24f6899449761
MIP_WORKER_SHA=d9aa1144742f65fac2b3cd533c6ef8bc75c24c2759aaa322314a172b67f9db71
MIP_RUNNER_SHA=bcb5a6b76040ff6ddfa932433d296a1f0f72207b28cbba738b1b4dd39f1eaac7
FREEZE_ONE=$BATCH/launch/freeze_one.sub
MIP_WORKER=$BATCH/launch/warm_mip.sub
CG_RECORD=$BATCH/cg_jobs.tsv
RECORD=$BATCH/freeze_mip_jobs.tsv
DRY_RUN="${EVSP_DRY_RUN:-1}"
SBATCH="${EVSP_SBATCH:-/usr/local/slurm/slurm-25.05.5/bin/sbatch}"
[[ -x "$SBATCH" && -f "$CG_RECORD" ]] || fatal "CG record missing"
[[ "$(sha256sum "$FREEZE_ONE" | cut -d' ' -f1)" == "$FREEZE_ONE_SHA" ]] || fatal "freeze worker hash mismatch"
[[ "$(sha256sum "$MIP_WORKER" | cut -d' ' -f1)" == "$MIP_WORKER_SHA" ]] || fatal "MIP worker hash mismatch"
[[ "$(sha256sum "$MIP_REPO/src/run_exact_pool_mip.py" | cut -d' ' -f1)" == "$MIP_RUNNER_SHA" ]] || fatal "MIP runner hash mismatch"
[[ "$(git -C "$MIP_REPO" rev-parse HEAD)" == "$MIP_COMMIT" ]] || fatal "MIP commit mismatch"
[[ -z "$(git -C "$MIP_REPO" status --porcelain --untracked-files=no)" ]] || fatal "MIP repo dirty"
! git -C "$MIP_REPO" symbolic-ref -q HEAD >/dev/null || fatal "MIP repo is not detached"
[[ ! -e "$RECORD" && ! -e "$RECORD.planned" ]] || fatal "downstream submission record exists"
exec 9>>"$BATCH/locks/submit-freeze-mip.lock"; flock -n 9 || fatal "another downstream submission is active"
if [[ "$DRY_RUN" == 1 ]]; then command_file="$BATCH/dryrun_freeze_mip_commands.sh"; : > "$command_file"; chmod 700 "$command_file"; record="$BATCH/freeze_mip_jobs.dryrun.tsv"; else record="$RECORD.planned"; fi
printf 'replicate\tscale\tcg_job\tfreeze_job\tmip_job\tfreeze_dependency\tmip_dependency\tfreeze_partition\tmip_partition\texclude\tcpus\tmem\ttime\tstage1_s\tstage2\tno_requeue\n' > "$record"
while IFS=$'\t' read -r replicate scale cg_job dependency rest; do
  [[ "$replicate" == "replicate" ]] && continue
  [[ "$replicate" =~ ^(1|2|4|6)$ && "$scale" =~ ^([2-9]|10)$ && "$cg_job" =~ ^[0-9]+$ ]] || fatal "bad CG row"
  campaign="$BATCH/p$replicate"; cell=$(printf 'k%02d_p%d' "$scale" "$replicate"); rep=event_2p5_event5
  instance=$(printf 'scale_ladder/instances/nested_probability_k2_15_20260908/Practice_Custom_DutyUnion_k%02d_p%02d_20260908.csv' "$scale" "$replicate")
  source="$campaign/cg/M__${cell}__warm_cover__${rep}.json"; snapshot="$campaign/snapshots/M__${cell}__warm_cover__${rep}.json"; freeze_record="$campaign/records/freeze__${cell}__warm_cover__${rep}.json"
  freeze_name=$(printf 'fi%dk%02d' "$replicate" "$scale")
  freeze_args=("$SBATCH" --parsable --partition=default_partition --exclude=scaglione-compute-01,scaglione-cpu-[01-05] --cpus-per-task=1 --mem=16G --time=02:00:00 --no-requeue --dependency="afterok:$cg_job" --job-name="$freeze_name" --output="$campaign/logs/freeze/%j.out" --error="$campaign/logs/freeze/%j.err" --export=ALL,EVSP_MIP_REPO="$MIP_REPO",EVSP_MIP_COMMIT="$MIP_COMMIT",EVSP_FREEZE_ONE_SHA256="$FREEZE_ONE_SHA" "$FREEZE_ONE" "$source" "$snapshot" "$freeze_record" "$cell" "$instance" "$SOURCE_COMMIT")
  if [[ "$DRY_RUN" == 1 ]]; then freeze_job="DRYRUN_FREEZE_P${replicate}_K${scale}"; printf '%q ' "${freeze_args[@]}" >> "$command_file"; printf '\n' >> "$command_file"; else freeze_job=$("${freeze_args[@]}"); [[ "$freeze_job" =~ ^[0-9]+$ ]] || fatal "bad freeze ID"; fi
  out="$campaign/mip/M__${cell}__warm_cover__${rep}__1h2stage.json"; case_key="warm_p${replicate}_k${scale}_cover_${rep}"; mip_name=$(printf 'MCw%dk%02d' "$replicate" "$scale")
  mip_args=("$SBATCH" --parsable --partition=scaglione --exclude=scaglione-compute-01 --cpus-per-task=8 --mem=16G --time=02:00:00 --no-requeue --dependency="afterok:$freeze_job" --job-name="$mip_name" --output="$campaign/logs/mip/%j.out" --error="$campaign/logs/mip/%j.err" --export=ALL,EVSP_DR_ROOT="$MIP_REPO",EVSP_EXPECTED_COMMIT="$MIP_COMMIT",EVSP_REQUIRE_DETACHED=1,EVSP_MIP_EXPECTED_WORKER_SHA256="$MIP_WORKER_SHA",EVSP_MIP_EXPECTED_RUNNER_SHA256="$MIP_RUNNER_SHA",EVSP_MIP_CASE_KEY="$case_key",EVSP_MIP_CASE_LOCK_PATH="$campaign/locks/${case_key}.lock",EXACT_MIP_TWO_STAGE=1,EXACT_MIP_COVER=1,EXACT_MIP_STAGE1_SECONDS=1800,EXACT_MIP_PROGRESS_DIR="$campaign/progress/${case_key}",EVSP_CONDA_ENV=/home/nc437/evsp_env "$MIP_WORKER" "$snapshot" 3600 "$out" 0.0001)
  if [[ "$DRY_RUN" == 1 ]]; then mip_job="DRYRUN_MIP_P${replicate}_K${scale}"; printf '%q ' "${mip_args[@]}" >> "$command_file"; printf '\n' >> "$command_file"; else mip_job=$("${mip_args[@]}"); [[ "$mip_job" =~ ^[0-9]+$ ]] || fatal "bad MIP ID"; fi
  printf '%s\t%s\t%s\t%s\t%s\tafterok:%s\tafterok:%s\tdefault_partition\tscaglione\tscaglione-compute-01\t8\t16G\t02:00:00\t1800\tat_most\ttrue\n' "$replicate" "$scale" "$cg_job" "$freeze_job" "$mip_job" "$cg_job" "$freeze_job" >> "$record"
done < "$CG_RECORD"
[[ "$(($(wc -l < "$record")-1))" -eq 36 ]] || fatal "expected 36 downstream rows"
if [[ "$DRY_RUN" == 1 ]]; then cat "$record"; exit 0; fi
mv "$RECORD.planned" "$RECORD"; sha256sum "$RECORD" > "$RECORD.sha256"; cat "$RECORD"
