#!/bin/bash

main() {
  [ "$#" -eq 1 ] || { echo "usage: record_results.sh RUN_ID" >&2; return 2; }
  RUN_ID=$1; [[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "unsafe RUN_ID" >&2; return 2; }
  LL_ROOT=${LL_ROOT:-"$HOME/ladder-lite"}; REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd) || return 1
  PYTHON=${LL_PYTHON:-/home/nc437/evsp_env/bin/python3.12}; NORM="$LL_ROOT/normalized"
  if [ ! -s "$NORM/cg_run_summary.csv" ] || [ ! -s "$NORM/mip_run_summary.csv" ]; then
    echo "normalized summaries missing: $NORM" >&2; return 1
  fi
  RECORDS=${LL_RECORDS_ROOT:-"$REPO/records"}
  DEST="$RECORDS/runs/$RUN_ID"; mkdir -p "$DEST" || return 1
  cp -n "$NORM"/*.csv "$DEST/" || return 1
  "$PYTHON" -B - "$REPO" "$LL_ROOT" "$RUN_ID" "$RECORDS" <<'PY'
import csv,datetime,hashlib,json,pathlib,sys
repo=pathlib.Path(sys.argv[1]); root=pathlib.Path(sys.argv[2]); run=sys.argv[3]
norm=root/"normalized"; plan=json.load(open(root/"campaign/approved-plan.json"))
manifest=json.load(open(root/"campaign/campaign.json")); target=pathlib.Path(sys.argv[4])/"RESULTS_LOG.csv"
def rows(name):
 with open(norm/name,newline="") as h:return list(csv.DictReader(h))
cg=rows("cg_run_summary.csv"); mi=rows("mip_run_summary.csv"); ci=rows("cg_iteration_long.csv"); mc=rows("mip_checkpoint_long.csv")
role=lambda j:"primary" if j["phase"]=="CG" else "small_grid_sensitivity"
cgmap={(r["cell_id"],r["campaign_role"],r["soc_step"],r["block_min"],r["cg_replicate"]):r for r in cg}
mipmap={(r["cell_id"],r["arm"],r["cg_replicate"]):r for r in mi}
lastcg={}
for r in ci:lastcg[(r["cell_id"],r["campaign_role"],r["soc_step"],r["block_min"],r["cg_replicate"])]=r
lastm={}
for r in mc:lastm[(r["cell_id"],r["arm"],r["cg_replicate"])]=r
with open(target,newline="") as h: reader=csv.DictReader(h); fields=reader.fieldnames; existing={(r["run_id"],r["cell_id"]) for r in reader}
new=[]; skipped=0
for j in plan["jobs"]:
 if j["phase"] not in {"CG","CG_SENSITIVITY","MIP"}:continue
 key=(j["cell_id"],j["arm"],str(j["cg_replicate"])) if j["phase"]=="MIP" else (j["cell_id"],role(j),str(j["soc_step"]),str(j["block_min"]),str(j["cg_replicate"]))
 summary=(mipmap if j["phase"]=="MIP" else cgmap).get(key); detail=(lastm if j["phase"]=="MIP" else lastcg).get(key,{})
 if (run,j["job_key"]) in existing:skipped+=1;continue
 out=pathlib.Path(j["output"]); censored=(summary or {}).get("censored")=="True"; missing=summary is None
 row={f:"" for f in fields}; row.update({
  "date_utc":datetime.datetime.now(datetime.timezone.utc).date().isoformat(),"run_id":run,
  "execution_mode":manifest["execution_mode"],"commit":manifest["commit"],
  "group":("MIP_"+("RAW" if j["arm"]=="RAW" else "KNOWN") if j["phase"]=="MIP" else j["phase"]),
  "cell_id":j["job_key"],"phase":j["phase"],"arm":j["arm"] or "","scale":j["scale"],
  "sel_rep":j["selection_replicate"],"cg_rep":j["cg_replicate"],"soc_step":j["soc_step"],
  "block_min":j["block_min"],"budget_s":j["budget_s"],"status":"missing" if missing else "censored" if censored else "completed",
  "label":"budget_overridden" if pathlib.Path(str(out)+".override.json").exists() else "ladder_lite_direct_array",
  "route_weight":(summary or {}).get("final_route_weight",""),"route_weight_meaning":"combined-cost-master route weight",
  "min_reduced_cost":(summary or {}).get("final_min_reduced_cost",""),"certified":(summary or {}).get("pricing_certified",(summary or {}).get("fleet_proven","")),
  "artificial_mass":(summary or {}).get("final_artificial_mass",""),"n_columns":(summary or {}).get("pool_columns",""),
  "iters":(summary or {}).get("iterations",""),"master_s":detail.get("master_time_s",""),"pricing_s":detail.get("pricing_time_s",""),
  "wall_s":(summary or {}).get("elapsed_s",(summary or {}).get("runtime_s","")),"stop_reason":(summary or {}).get("stopping_reason",(summary or {}).get("status_name","")),
  "censor_reason":"normalized row missing" if missing else (summary or {}).get("missing_reason",""),"mip_incumbent_fleet":(summary or {}).get("buses",""),
  "mip_bound":(summary or {}).get("fleet_bound",""),"mip_gap":(summary or {}).get("mip_gap",""),"mip_nodes":detail.get("node_count",""),
  "target_fleet":j["target_fleet"],"artifact_path":str(out),"artifact_sha256":hashlib.sha256(out.read_bytes()).hexdigest() if out.is_file() else "",
  "notes":j.get("scientific_role") or ""})
 new.append(row);existing.add((run,j["job_key"]))
with open(target,"a",newline="") as h: csv.DictWriter(h,fieldnames=fields,lineterminator="\n").writerows(new)
print(f"appended={len(new)} skipped={skipped}")
PY
}
main "$@"
