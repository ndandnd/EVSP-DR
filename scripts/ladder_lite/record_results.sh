#!/bin/bash
main() {
  [ "$#" -eq 1 ] || { echo "usage: record_results.sh RUN_ID" >&2; return 2; }
  RUN_ID=$1; [[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "unsafe RUN_ID" >&2; return 2; }
  LL_ROOT=${LL_ROOT:-"$HOME/ladder-lite"}; REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd) || return 1
  PYTHON=${LL_PYTHON:-/home/nc437/evsp_env/bin/python3.12}; NORM="$LL_ROOT/normalized"; CAMPAIGN=${LL_CAMPAIGN:-"ll_$(date -u +%Y%m%d)"}
  if [ ! -s "$NORM/cg_run_summary.csv" ] || [ ! -s "$NORM/mip_run_summary.csv" ]; then echo "normalized summaries missing: $NORM" >&2; return 1; fi
  RECORDS=${LL_RECORDS_ROOT:-"$REPO/records"}
  DEST="$RECORDS/runs/$RUN_ID"; mkdir -p "$DEST" || return 1
  cp -n "$NORM"/*.csv "$DEST/" || return 1
  "$PYTHON" -B - "$REPO" "$LL_ROOT" "$RUN_ID" "$RECORDS" "$CAMPAIGN" <<'PY'
import csv,datetime,hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[2]); run=sys.argv[3]; campaign=root/"campaign"/sys.argv[5]
norm=root/"normalized"; plan=json.load(open(campaign/"approved-plan.json"))
manifest=json.load(open(campaign/"campaign.json")); target=pathlib.Path(sys.argv[4])/"RESULTS_LOG.csv"
def rows(name):return list(csv.DictReader(open(norm/name,newline="")))
cg=rows("cg_run_summary.csv"); mi=rows("mip_run_summary.csv"); ci=rows("cg_iteration_long.csv"); mc=rows("mip_checkpoint_long.csv")
role=lambda j:"primary" if j["phase"]=="CG" else "small_grid_sensitivity"
cgmap={(r["cell_id"],r["campaign_role"],r["soc_step"],r["block_min"],r["cg_replicate"]):r for r in cg}
mipmap={(r["cell_id"],r["arm"],r["cg_replicate"]):r for r in mi}
lastcg={(r["cell_id"],r["campaign_role"],r["soc_step"],r["block_min"],r["cg_replicate"]):r for r in ci}
lastm={(r["cell_id"],r["arm"],r["cg_replicate"]):r for r in mc}
with open(target,newline="") as h: reader=csv.DictReader(h); fields=reader.fieldnames; existing={(r["run_id"],r["cell_id"]) for r in reader}
expected="date_utc,run_id,execution_mode,commit,group,cell_id,phase,arm,scale,sel_rep,cg_rep,soc_step,block_min,budget_s,status,label,route_weight,route_weight_meaning,min_reduced_cost,certified,artificial_mass,n_columns,iters,master_s,pricing_s,wall_s,stop_reason,censor_reason,max_rss_mb,mip_incumbent_fleet,mip_bound,mip_gap,first_incumbent_s,mip_nodes,target_fleet,artifact_path,artifact_sha256,notes,source_cg_certified,source_cg_stop_reason,source_cg_iterations,pool_fleet_proven,pool_mip_bound,model_fleet_proven,model_optimality_method,optimality_scope,physical_witness_valid".split(",")
if fields!=expected:raise SystemExit("RESULTS_LOG.csv header differs")
new=[]; skipped=0; seen_mip=set()
for j in plan["jobs"]:
 if j["phase"] not in {"CG","CG_SENSITIVITY","MIP"}:continue
 key=(j["cell_id"],j["arm"],str(j["cg_replicate"])) if j["phase"]=="MIP" else (j["cell_id"],role(j),str(j["soc_step"]),str(j["block_min"]),str(j["cg_replicate"]))
 summary=(mipmap if j["phase"]=="MIP" else cgmap).get(key); detail=(lastm if j["phase"]=="MIP" else lastcg).get(key,{})
 science=summary; dep=next((x for x in plan["jobs"] if j["phase"]=="MIP" and x["job_key"]==j["dependency_cg"]),None)
 if dep:seen_mip.add(key);science=cgmap.get((dep["cell_id"],role(dep),str(dep["soc_step"]),str(dep["block_min"]),str(dep["cg_replicate"])),{})
 if (run,j["job_key"]) in existing:skipped+=1;continue
 out=pathlib.Path(j["output"]); censored=(summary or {}).get("censored")=="True"; missing=summary is None; explicit_missing=missing or (summary or {}).get("stopping_reason")=="missing" or str((summary or {}).get("missing_reason","")).startswith("missing:")
 row={f:"" for f in fields}; row.update({
  "date_utc":datetime.datetime.now(datetime.timezone.utc).date().isoformat(),"run_id":run,"execution_mode":manifest["execution_mode"],"commit":manifest["commit"],
  "group":("MIP_"+("RAW" if j["arm"]=="RAW" else "KNOWN") if j["phase"]=="MIP" else j["phase"]),
  "cell_id":j["job_key"],"phase":j["phase"],"arm":j["arm"] or "","scale":j["scale"],
  "sel_rep":j["selection_replicate"],"cg_rep":j["cg_replicate"],"soc_step":j["soc_step"],
  "block_min":j["block_min"],"budget_s":j["budget_s"],"status":"missing" if explicit_missing else "censored" if censored else "completed",
  "label":"budget_overridden" if pathlib.Path(str(out)+".override.json").exists() else "ladder_lite_direct_array",
  "route_weight":(science or {}).get("final_route_weight",""),"route_weight_meaning":("fleet LP lower bound (certified discretized model; grid stated; D0019)" if (science or {}).get("pricing_certified") in {True,"True"} else "upper bound on LP optimum only; no fleet LP lower bound") if (science or {}).get("final_route_weight","")!="" else "",
  "min_reduced_cost":(science or {}).get("final_min_reduced_cost",""),"certified":("" if j["phase"]=="MIP" else (summary or {}).get("pricing_certified","")),
  "source_cg_certified":((science or {}).get("pricing_certified","") if j["phase"]=="MIP" else ""),
  "source_cg_stop_reason":((science or {}).get("stopping_reason","") if j["phase"]=="MIP" else ""),
  "source_cg_iterations":((science or {}).get("iterations","") if j["phase"]=="MIP" else ""),
  "pool_fleet_proven":((summary or {}).get("fleet_proven","") if j["phase"]=="MIP" else ""),
  "pool_mip_bound":((summary or {}).get("fleet_bound","") if j["phase"]=="MIP" else ""),
  "optimality_scope":("finite_pool" if j["phase"]=="MIP" and (summary or {}).get("fleet_proven") in {True,"True"} else ""),
  "physical_witness_valid":((summary or {}).get("physically_validated_schedule","") if j["phase"]=="MIP" else ""),
  "artificial_mass":(science or {}).get("final_artificial_mass",""),"n_columns":(science or {}).get("pool_columns",""),
  "iters":(science or {}).get("iterations",""),"master_s":detail.get("master_time_s",""),"pricing_s":detail.get("pricing_time_s",""),
  "wall_s":(summary or {}).get("elapsed_s",(summary or {}).get("runtime_s","")),"stop_reason":(summary or {}).get("stopping_reason",(summary or {}).get("status_name","")),
  "censor_reason":"normalized row missing" if missing else ((summary or {}).get("missing_reason") or (summary or {}).get("stopping_reason") or (summary or {}).get("status_name") or "censored") if censored else "","mip_incumbent_fleet":(summary or {}).get("buses",""),
  "mip_bound":(summary or {}).get("fleet_bound",""),"mip_gap":(summary or {}).get("mip_gap",""),"mip_nodes":detail.get("node_count",""),"target_fleet":j["target_fleet"],"artifact_path":str(out),"artifact_sha256":hashlib.sha256(out.read_bytes()).hexdigest() if out.is_file() else "",
  "notes":j.get("scientific_role") or ""})
 new.append(row);existing.add((run,j["job_key"]))
for key,s in mipmap.items():
 if key in seen_mip:continue
 rid=f"reuse_{key[0]}_{key[1].lower().replace('-','_')}"
 if (run,rid) in existing:skipped+=1;continue
 row={f:"" for f in fields};row.update({"date_utc":datetime.datetime.now(datetime.timezone.utc).date().isoformat(),"run_id":run,"execution_mode":manifest["execution_mode"],"commit":manifest["commit"],"group":"MIP_REUSE","cell_id":rid,"phase":"MIP","arm":key[1],"scale":s.get("scale",40),"cg_rep":key[2],"budget_s":s.get("budget_s",""),"status":"missing" if s.get("output_available")=="False" else "censored" if s.get("censored")=="True" else "completed","label":"ladder_lite_direct_array","censor_reason":s.get("missing_reason",""),"mip_incumbent_fleet":s.get("buses",""),"mip_bound":s.get("fleet_bound",""),"mip_gap":s.get("mip_gap",""),"pool_fleet_proven":s.get("fleet_proven",""),"pool_mip_bound":s.get("fleet_bound",""),"optimality_scope":("finite_pool" if s.get("fleet_proven") in {True,"True"} else ""),"physical_witness_valid":s.get("physically_validated_schedule",""),"notes":"validated k40 reuse slot"});new.append(row);existing.add((run,rid))
with open(target,"a",newline="") as h: csv.DictWriter(h,fieldnames=fields,lineterminator="\n").writerows(new)
print(f"appended={len(new)} skipped={skipped}")
PY
}
main "$@"
