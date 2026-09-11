#!/bin/bash
set -euo pipefail
fatal() { echo "[prepare multichain] $*" >&2; exit 2; }
BASE=/home/nc437/ladder-lite
BATCH=$BASE/nested_warm_multichain_p1246_k2_10_20260910_ecb60c1
STAGING=$BASE/launch_packages/warm_multichain_p1246_20260910
EXECUTION=$BASE/execution/ecb60c154a9a5db385e3a573949ec9fd0a737af3
SOURCE_CACHE=$BASE/nested_probability_k2_15_fresh84_20260908_21fbecb
COMMIT=ecb60c154a9a5db385e3a573949ec9fd0a737af3
[[ -d "$STAGING" && -f "$STAGING/prepare_chain.py" && -f "$STAGING/warm_chain_cg.sub" ]] || fatal "staging package incomplete"
[[ "$(git -C "$EXECUTION" rev-parse HEAD)" == "$COMMIT" ]] || fatal "execution commit mismatch"
[[ -z "$(git -C "$EXECUTION" status --porcelain --untracked-files=no)" ]] || fatal "execution checkout dirty"
if [[ ! -e "$BATCH" ]]; then
  mkdir -p "$BATCH/launch"
elif [[ -f "$BATCH/batch_manifest.json" ]]; then
  fatal "completed batch manifest already exists: $BATCH/batch_manifest.json"
else
  mkdir -p "$BATCH/launch"
fi
cp "$STAGING"/* "$BATCH/launch/"
chmod 755 "$BATCH/launch/"*.sh "$BATCH/launch/"*.py "$BATCH/launch/"*.sub
for replicate in 1 2 4 6; do
  if [[ -f "$BATCH/p$replicate/execution_plan.json" ]]; then
    echo "reuse verified prepared chain p$replicate"
    continue
  fi
  [[ ! -e "$BATCH/p$replicate" ]] || fatal "partial chain without execution plan: p$replicate"
  "$HOME/evsp_env/bin/python" "$BATCH/launch/prepare_chain.py" \
    "$BATCH/p$replicate" "$EXECUTION" "$SOURCE_CACHE" "$COMMIT" \
    --replicate "$replicate" --worker-source "$BATCH/launch/warm_chain_cg.sub"
done
"$HOME/evsp_env/bin/python" - "$BATCH" <<'PY'
import datetime as dt, hashlib, json, sys
from pathlib import Path
root=Path(sys.argv[1])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
plans=[]
for replicate in (1,2,4,6):
    path=root/f"p{replicate}"/"execution_plan.json"
    plan=json.loads(path.read_text())
    plans.append({
        "replicate":replicate,"campaign_root":str(root/f"p{replicate}"),
        "execution_plan":str(path),"execution_plan_sha256":sha(path),
        "cache_count":len(plan["cache_rows"]),
        "input_sha256_by_scale":{str(r["scale"]):r["instance_file_sha256"] for r in plan["selection_rows"]},
        "cache_sha256_by_scale":{str(r["scale"]):r["pickle_sha256"] for r in plan["cache_rows"]},
    })
payload={
 "schema":"evsp-dr-inherited-multichain-batch-v1",
 "created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
 "batch_root":str(root),"replicates":[1,2,4,6],"scales":list(range(2,11)),
 "source_cache_root":"/home/nc437/ladder-lite/nested_probability_k2_15_fresh84_20260908_21fbecb",
 "source_selection_manifest_sha256":"3d3fdf1d13f3849dd4af9d7d24d5cdef3108f03c1a6521af9d98b596402d4899",
 "execution_commit":"ecb60c154a9a5db385e3a573949ec9fd0a737af3",
 "physics":{"master_sense":"cover","battery_kwh":240.0,"charge_kw":240.0,"soc_step_kwh":2.5,"block_minutes":5,"time_model":"event","event_arc_mode":"lazy","initial_pool":"real singletons"},
 "dependency_scope":"within each replicate only: k(n) afterok k(n-1); no cross-replicate dependency",
 "slurm_cg":{"partition":"default_partition","exclude":"scaglione-compute-01","cpus_per_task":8,"memory":"96G","time":"08:15:00","requeue":True,"signal":"B:USR1@120","maximum_runnable_chains":4},
 "chains":plans,
 "status":"prepared_not_submitted"
}
out=root/"batch_manifest.json"; out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({"batch_root":str(root),"batch_manifest_sha256":sha(out),"chains":len(plans),"cache_records":sum(p["cache_count"] for p in plans)},sort_keys=True))
PY
sha256sum "$BATCH/launch/"* "$BATCH/batch_manifest.json" > "$BATCH/launch_manifest.sha256"
