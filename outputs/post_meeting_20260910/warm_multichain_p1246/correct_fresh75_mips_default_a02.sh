#!/bin/bash
set -euo pipefail
fatal() { echo "[correct fresh75 MIPs a02] $*" >&2; exit 2; }

BASE=/home/nc437/ladder-lite
FRESH=$BASE/covering_complement75_20260911_21fbecb
DOWNSTREAM=$FRESH/downstream
FREEZE_ARRAY=810587
OLD_SMALL=812608
OLD_LARGE=812619
MIP_REPO=$BASE/execution/871d057e1067411f09581e37d78f7c1ca43f68bb
MIP_COMMIT=871d057e1067411f09581e37d78f7c1ca43f68bb
MATRIX_SHA=8c0bc819ab58ea8b770bebb91997126f6088681b886cf674d861ffed0eae5da6
ATTEMPT_WRAPPER_SHA=13d9d5a6d2cd674e78ca046146254aa6b817cdda215609ba575467b5eadeea52
export ATTEMPT_WRAPPER_SHA
MIP_WORKER_SHA=d9aa1144742f65fac2b3cd533c6ef8bc75c24c2759aaa322314a172b67f9db71
MIP_RUNNER_SHA=bcb5a6b76040ff6ddfa932433d296a1f0f72207b28cbba738b1b4dd39f1eaac7
SBATCH=/usr/local/slurm/slurm-25.05.5/bin/sbatch
SCANCEL=/usr/local/slurm/slurm-25.05.5/bin/scancel
SQUEUE=/usr/local/slurm/slurm-25.05.5/bin/squeue
SACCT=/usr/local/slurm/slurm-25.05.5/bin/sacct
ATTEMPT_TAG=default1h_a02
RECORD=$DOWNSTREAM/default_mip_migration_a02.json

[[ ! -e "$RECORD" ]] || fatal "migration record exists"
[[ "$(sha256sum "$FRESH/matrix.tsv" | awk '{print $1}')" == "$MATRIX_SHA" ]] || fatal "matrix hash mismatch"
[[ "$(sha256sum "$DOWNSTREAM/fresh_mip_default_attempt.sub" | awk '{print $1}')" == "$ATTEMPT_WRAPPER_SHA" ]] || fatal "attempt wrapper mismatch"
[[ "$(sha256sum "$DOWNSTREAM/warm_mip.sub" | awk '{print $1}')" == "$MIP_WORKER_SHA" ]] || fatal "MIP worker mismatch"
[[ "$(sha256sum "$MIP_REPO/src/run_exact_pool_mip.py" | awk '{print $1}')" == "$MIP_RUNNER_SHA" ]] || fatal "MIP runner mismatch"
[[ "$(git -C "$MIP_REPO" rev-parse HEAD)" == "$MIP_COMMIT" ]] || fatal "MIP commit mismatch"
[[ -z "$(git -C "$MIP_REPO" status --porcelain --untracked-files=no)" ]] || fatal "MIP checkout dirty"
! git -C "$MIP_REPO" symbolic-ref -q HEAD >/dev/null || fatal "MIP checkout is not detached"

for old in "$OLD_SMALL" "$OLD_LARGE"; do
  states=$($SQUEUE -h -j "$old" -r -o '%T' | sort -u)
  [[ "$states" == "PENDING" ]] || fatal "old array $old is not wholly pending: ${states:-absent}"
  starts=$($SACCT -n -X -j "$old" -o Start -P | sed '/^Unknown$/d;/^$/d')
  [[ -z "$starts" ]] || fatal "old array $old has an observed start"
done

mkdir -p "$DOWNSTREAM/migration_records" "$FRESH/logs/mip_default"
$SQUEUE -h -j "$OLD_SMALL,$OLD_LARGE" -r -o '%A|%a|%j|%T|%P|%R' > "$DOWNSTREAM/migration_records/old_pending_before_cancel.tsv"
$SCANCEL "$OLD_SMALL" "$OLD_LARGE"

common_export=ALL,EVSP_DR_ROOT="$MIP_REPO",EVSP_EXPECTED_COMMIT="$MIP_COMMIT",EVSP_REQUIRE_DETACHED=1,EVSP_MIP_EXPECTED_WORKER_SHA256="$MIP_WORKER_SHA",EVSP_MIP_EXPECTED_RUNNER_SHA256="$MIP_RUNNER_SHA",EVSP_ALLOW_PREEMPTIBLE_MIP=1,EVSP_MIP_REQUIRED_PARTITION=default_partition,EXACT_MIP_TWO_STAGE=1,EXACT_MIP_COVER=1,EXACT_MIP_STAGE1_SECONDS=1800,EVSP_MIP_SECONDS=3600,EVSP_CONDA_ENV=/home/nc437/evsp_env,EVSP_FRESH_ROOT="$FRESH",EVSP_MATRIX_SHA256="$MATRIX_SHA",EVSP_MIP_ATTEMPT_WRAPPER_SHA256="$ATTEMPT_WRAPPER_SHA",EVSP_MIP_WORKER="$DOWNSTREAM/warm_mip.sub",EVSP_ATTEMPT_TAG="$ATTEMPT_TAG"
small_job=$($SBATCH --parsable --partition=default_partition --exclude=scaglione-compute-01,scaglione-cpu-[01-05] --array=0-44%30 --cpus-per-task=8 --mem=16G --time=02:00:00 --no-requeue --dependency="aftercorr:$FREEZE_ARRAY" --job-name=MPdv75s --output="$FRESH/logs/mip_default/%A_%a.out" --error="$FRESH/logs/mip_default/%A_%a.err" --export="$common_export" "$DOWNSTREAM/fresh_mip_default_attempt.sub")
[[ "$small_job" =~ ^[0-9]+$ ]] || fatal "bad small job ID"
large_job=$($SBATCH --parsable --partition=default_partition --exclude=scaglione-compute-01,scaglione-cpu-[01-05] --array=45-74%20 --cpus-per-task=8 --mem=32G --time=02:00:00 --no-requeue --dependency="aftercorr:$FREEZE_ARRAY" --job-name=MPdv75l --output="$FRESH/logs/mip_default/%A_%a.out" --error="$FRESH/logs/mip_default/%A_%a.err" --export="$common_export" "$DOWNSTREAM/fresh_mip_default_attempt.sub")
[[ "$large_job" =~ ^[0-9]+$ ]] || fatal "bad large job ID"

python3 - "$RECORD" "$small_job" "$large_job" <<'PY'
import datetime, hashlib, json, os, subprocess, sys
out, small, large = sys.argv[1:]
slurm = "/usr/local/slurm/slurm-25.05.5/bin"
def show(job):
    text=subprocess.check_output([slurm+"/scontrol","show","job",job],text=True)
    required={"Partition=default_partition","Requeue=0","TimeLimit=02:00:00","ExcNodeList=scaglione-compute-01,scaglione-cpu-[01-05]"}
    missing=sorted(x for x in required if x not in text)
    return {"job_id":job,"scontrol":text,"effective_checks_passed":not missing,"missing":missing}
payload={
 "schema":"evsp-dr-fresh75-default-mip-correction-v1",
 "recorded_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "old_arrays":[{"job_id":"812608","disposition":"cancelled_before_start_scientific_budget_correction"},{"job_id":"812619","disposition":"cancelled_before_start_scientific_budget_correction"}],
 "freeze_dependency":"aftercorr:810587",
 "attempt_tag":"default1h_a02",
 "solver_budget_seconds":3600,
 "stage1_max_seconds":1800,
 "slurm_time":"02:00:00",
 "global_concurrency_cap":50,
 "new_arrays":[
   {"job_id":small,"indices":"0-44","throttle":30,"memory":"16G","cpus":8},
   {"job_id":large,"indices":"45-74","throttle":20,"memory":"32G","cpus":8}],
 "execution_commit":"871d057e1067411f09581e37d78f7c1ca43f68bb",
 "worker_sha256":"d9aa1144742f65fac2b3cd533c6ef8bc75c24c2759aaa322314a172b67f9db71",
 "runner_sha256":"bcb5a6b76040ff6ddfa932433d296a1f0f72207b28cbba738b1b4dd39f1eaac7",
 "wrapper_sha256":os.environ["ATTEMPT_WRAPPER_SHA"],
 "effective":[show(small),show(large)]}
payload["all_effective_checks_passed"]=all(x["effective_checks_passed"] for x in payload["effective"])
tmp=out+".tmp"; open(tmp,"w").write(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,out)
print(json.dumps({"small_job":small,"large_job":large,"checks":payload["all_effective_checks_passed"]}))
PY
sha256sum "$RECORD" > "$RECORD.sha256"
