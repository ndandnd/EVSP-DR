#!/bin/bash
set -euo pipefail
fatal() { echo "[submit fresh75 downstream] $*" >&2; exit 2; }
BASE=/home/nc437/ladder-lite
FRESH=$BASE/covering_complement75_20260911_21fbecb
CG_ARRAY=810454
MIP_REPO=$BASE/execution/871d057e1067411f09581e37d78f7c1ca43f68bb
MIP_COMMIT=871d057e1067411f09581e37d78f7c1ca43f68bb
SOURCE_EXECUTION=$BASE/execution/21fbecba826824c44f897feef038fcf51c532582
SOURCE_COMMIT=21fbecba826824c44f897feef038fcf51c532582
MATRIX_SHA=8c0bc819ab58ea8b770bebb91997126f6088681b886cf674d861ffed0eae5da6
FREEZE_ONE_SHA=472f18bf49f8a3e4dcd8ece66d87db1d41d5ccf4a3c1ead42fb24f6899449761
FREEZE_WRAPPER_SHA=f877b6eeba009d1b46667c6f713a881bbacd89990884ec25a4bc94b52e1ff43a
MIP_ARRAY_WRAPPER_SHA=7359b6dca60022a5ea629b208475bff45761b4397dd36acbb5391fa2e0ecb3be
MIP_WORKER_SHA=d9aa1144742f65fac2b3cd533c6ef8bc75c24c2759aaa322314a172b67f9db71
MIP_RUNNER_SHA=bcb5a6b76040ff6ddfa932433d296a1f0f72207b28cbba738b1b4dd39f1eaac7
DOWNSTREAM=$FRESH/downstream
RECORD=$FRESH/freeze_mip_jobs.tsv
DRY_RUN="${EVSP_DRY_RUN:-1}"
SBATCH="${EVSP_SBATCH:-/usr/local/slurm/slurm-25.05.5/bin/sbatch}"
[[ -x "$SBATCH" && -d "$FRESH" && -d "$DOWNSTREAM" ]] || fatal "campaign/downstream not staged"
[[ "$(sha256sum "$FRESH/matrix.tsv" | cut -d' ' -f1)" == "$MATRIX_SHA" ]] || fatal "matrix hash mismatch"
python3 - "$FRESH/matrix.tsv" <<'PY'
import csv, sys
rows=list(csv.reader(open(sys.argv[1]), delimiter='\t'))
assert len(rows)==75
assert all(int(row[0])==i for i,row in enumerate(rows))
assert all(int(row[2])<=10 for row in rows[:45])
assert all(int(row[2])>=11 for row in rows[45:])
assert len({row[1] for row in rows})==75
PY
[[ "$(sha256sum "$DOWNSTREAM/freeze_one.sub" | cut -d' ' -f1)" == "$FREEZE_ONE_SHA" ]] || fatal "freeze-one hash mismatch"
[[ "$(sha256sum "$DOWNSTREAM/fresh_freeze_array.sub" | cut -d' ' -f1)" == "$FREEZE_WRAPPER_SHA" ]] || fatal "freeze wrapper hash mismatch"
[[ "$(sha256sum "$DOWNSTREAM/fresh_mip_array.sub" | cut -d' ' -f1)" == "$MIP_ARRAY_WRAPPER_SHA" ]] || fatal "MIP array wrapper hash mismatch"
[[ "$(sha256sum "$DOWNSTREAM/warm_mip.sub" | cut -d' ' -f1)" == "$MIP_WORKER_SHA" ]] || fatal "MIP worker hash mismatch"
[[ "$(sha256sum "$MIP_REPO/src/run_exact_pool_mip.py" | cut -d' ' -f1)" == "$MIP_RUNNER_SHA" ]] || fatal "MIP runner hash mismatch"
[[ "$(git -C "$MIP_REPO" rev-parse HEAD)" == "$MIP_COMMIT" ]] || fatal "MIP commit mismatch"
[[ -z "$(git -C "$MIP_REPO" status --porcelain --untracked-files=no)" ]] || fatal "MIP repo dirty"
! git -C "$MIP_REPO" symbolic-ref -q HEAD >/dev/null || fatal "MIP repo is not detached"
[[ ! -e "$RECORD" && ! -e "$RECORD.planned" ]] || fatal "downstream submission record exists"
mkdir -p "$FRESH"/{snapshots,records,mip,progress,locks,logs/freeze,logs/mip}
exec 9>>"$FRESH/locks/submit-downstream.lock"; flock -n 9 || fatal "another downstream submission is active"
common_freeze_export=ALL,EVSP_MIP_REPO="$MIP_REPO",EVSP_MIP_COMMIT="$MIP_COMMIT",EVSP_FREEZE_ONE_SHA256="$FREEZE_ONE_SHA",EVSP_FRESH_ROOT="$FRESH",EVSP_MATRIX_SHA256="$MATRIX_SHA",EVSP_FREEZE_WRAPPER_SHA256="$FREEZE_WRAPPER_SHA",EVSP_FREEZE_ONE="$DOWNSTREAM/freeze_one.sub",EVSP_SOURCE_EXECUTION="$SOURCE_EXECUTION",EVSP_SOURCE_COMMIT="$SOURCE_COMMIT"
freeze_args=("$SBATCH" --parsable --partition=default_partition --exclude=scaglione-compute-01,scaglione-cpu-[01-05] --array=0-74%50 --cpus-per-task=1 --mem=16G --time=02:00:00 --no-requeue --dependency="aftercorr:$CG_ARRAY" --job-name=frzcv75 --output="$FRESH/logs/freeze/%A_%a.out" --error="$FRESH/logs/freeze/%A_%a.err" --export="$common_freeze_export" "$DOWNSTREAM/fresh_freeze_array.sub")
if [[ "$DRY_RUN" == 1 ]]; then freeze_job=DRYRUN_FREEZE; else freeze_job=$("${freeze_args[@]}"); [[ "$freeze_job" =~ ^[0-9]+$ ]] || fatal "bad freeze ID"; fi
common_mip_export=ALL,EVSP_DR_ROOT="$MIP_REPO",EVSP_EXPECTED_COMMIT="$MIP_COMMIT",EVSP_REQUIRE_DETACHED=1,EVSP_MIP_EXPECTED_WORKER_SHA256="$MIP_WORKER_SHA",EVSP_MIP_EXPECTED_RUNNER_SHA256="$MIP_RUNNER_SHA",EXACT_MIP_TWO_STAGE=1,EXACT_MIP_COVER=1,EXACT_MIP_STAGE1_SECONDS=1800,EVSP_CONDA_ENV=/home/nc437/evsp_env,EVSP_FRESH_ROOT="$FRESH",EVSP_MATRIX_SHA256="$MATRIX_SHA",EVSP_MIP_ARRAY_WRAPPER_SHA256="$MIP_ARRAY_WRAPPER_SHA",EVSP_MIP_WORKER="$DOWNSTREAM/warm_mip.sub"
small_args=("$SBATCH" --parsable --partition=scaglione --exclude=scaglione-compute-01 --array=0-44%8 --cpus-per-task=8 --mem=16G --time=02:00:00 --no-requeue --dependency="aftercorr:$freeze_job" --job-name=MCcv75s --output="$FRESH/logs/mip/%A_%a.out" --error="$FRESH/logs/mip/%A_%a.err" --export="$common_mip_export" "$DOWNSTREAM/fresh_mip_array.sub")
large_args=("$SBATCH" --parsable --partition=scaglione --exclude=scaglione-compute-01 --array=45-74%8 --cpus-per-task=8 --mem=32G --time=02:00:00 --no-requeue --dependency="aftercorr:$freeze_job" --job-name=MCcv75l --output="$FRESH/logs/mip/%A_%a.out" --error="$FRESH/logs/mip/%A_%a.err" --export="$common_mip_export" "$DOWNSTREAM/fresh_mip_array.sub")
if [[ "$DRY_RUN" == 1 ]]; then
  small_job=DRYRUN_MIP_SMALL; large_job=DRYRUN_MIP_LARGE
  { printf '%q ' "${freeze_args[@]}"; printf '\n'; printf '%q ' "${small_args[@]}"; printf '\n'; printf '%q ' "${large_args[@]}"; printf '\n'; } > "$FRESH/dryrun_freeze_mip_commands.sh"
  record="$FRESH/freeze_mip_jobs.dryrun.tsv"
else
  small_job=$("${small_args[@]}"); [[ "$small_job" =~ ^[0-9]+$ ]] || fatal "bad small MIP ID"
  large_job=$("${large_args[@]}"); [[ "$large_job" =~ ^[0-9]+$ ]] || fatal "bad large MIP ID"
  record="$RECORD.planned"
fi
printf 'stage\tjob_id\tarray\tdependency\tpartition\texclude\tcpus\tmem\ttime\tconcurrency\trequeue\n' > "$record"
printf 'freeze\t%s\t0-74\taftercorr:%s\tdefault_partition\tscaglione-compute-01,scaglione-cpu-[01-05]\t1\t16G\t02:00:00\t50\tfalse\n' "$freeze_job" "$CG_ARRAY" >> "$record"
printf 'mip_small_k2_10\t%s\t0-44\taftercorr:%s\tscaglione\tscaglione-compute-01\t8\t16G\t02:00:00\t8\tfalse\n' "$small_job" "$freeze_job" >> "$record"
printf 'mip_large_k11_15\t%s\t45-74\taftercorr:%s\tscaglione\tscaglione-compute-01\t8\t32G\t02:00:00\t8\tfalse\n' "$large_job" "$freeze_job" >> "$record"
if [[ "$DRY_RUN" == 1 ]]; then cat "$record"; exit 0; fi
mv "$RECORD.planned" "$RECORD"; sha256sum "$RECORD" > "$RECORD.sha256"; cat "$RECORD"
