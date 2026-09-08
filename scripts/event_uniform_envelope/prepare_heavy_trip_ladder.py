#!/usr/bin/env python3
"""Prepare immutable controls for six heavy-trip nested-prefix runs."""

from __future__ import annotations
import argparse,csv,hashlib,json,subprocess
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

SCALES=(2,3,5,8,10,15); CELLS=6; REP="event_2p5_event5"
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def git(repo,*a): return subprocess.run(["git","-C",str(repo),*a],check=True,text=True,capture_output=True).stdout.strip()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execution-repo",type=Path,required=True); ap.add_argument("--execution-commit",required=True)
    ap.add_argument("--output-root",type=Path,required=True); a=ap.parse_args()
    repo=a.execution_repo.resolve(); out=a.output_root.expanduser().resolve()
    if out.exists(): raise SystemExit(f"output exists: {out}")
    if git(repo,"rev-parse","HEAD")!=a.execution_commit or git(repo,"status","--porcelain","--untracked-files=no"):
        raise SystemExit("execution checkout identity mismatch")
    sym=subprocess.run(["git","-C",str(repo),"symbolic-ref","-q","HEAD"],capture_output=True)
    if sym.returncode==0: raise SystemExit("execution checkout must be detached")
    if sym.returncode!=1: raise SystemExit("cannot verify detached checkout")
    source=repo/"data/scale_ladder/instances/heavy_trip_nested_20260908"
    input_plan_path=source/"input_plan.json"; manifest_path=source/"selection_manifest.csv"; order_path=source/"duty_order.csv"
    plan=json.loads(input_plan_path.read_text()); rows=list(csv.DictReader(manifest_path.open(newline="")))
    if (plan.get("schema")!="evsp-dr-heavy-trip-nested-inputs-v1" or plan.get("selected_rows")!=CELLS
        or sha(manifest_path)!=plan["files"]["selection_manifest.csv"] or sha(order_path)!=plan["files"]["duty_order.csv"]):
        raise SystemExit("input plan identity mismatch")
    rows.sort(key=lambda r:int(r["scale"]))
    if Counter(int(r["scale"]) for r in rows)!=Counter({k:1 for k in SCALES}): raise SystemExit("scale counts mismatch")
    if len({r["cell_id"] for r in rows})!=CELLS: raise SystemExit("duplicate cell")
    for r in rows:
        p=repo/r["relative_path"]
        if sha(p)!=r["instance_file_sha256"]: raise SystemExit(f"instance hash mismatch: {r['cell_id']}")
    for d in ("network_cache","cg","snapshots","records","mip","progress","logs/cache","logs/cg","logs/freeze","logs/mip"):
        (out/d).mkdir(parents=True,exist_ok=True)
    (out/"input_selection_manifest.csv").write_bytes(manifest_path.read_bytes())
    matrix=out/"matrix.tsv"; pool=out/"pool_matrix.tsv"
    with matrix.open("x",newline="") as f, pool.open("x",newline="") as g:
        mw=csv.writer(f,delimiter="\t",lineterminator="\n")
        fields=("index","cell","scale","replicate","trips","representation","instance_relative_to_data","instance_sha256","base_status","resume_status","snapshot","record","mip_output")
        pw=csv.DictWriter(g,fieldnames=fields,delimiter="\t",lineterminator="\n"); pw.writeheader()
        for i,r in enumerate(rows):
            instance=repo/r["relative_path"]; stem=f"M__{r['cell_id']}__{REP}.json"; status=out/"cg"/stem
            mw.writerow([i,r["cell_id"],r["scale"],r["family_replicate"],r["trip_count"],str(instance),r["instance_file_sha256"],REP,"2.5","5","28800"])
            pw.writerow(dict(zip(fields,[i,r["cell_id"],r["scale"],r["family_replicate"],r["trip_count"],REP,r["relative_path"].removeprefix("data/"),r["instance_file_sha256"],status,status,out/"snapshots"/stem,out/"records"/(stem+".freeze.json"),out/"mip"/(stem+".raw_pool_mip_budgeted.json")])) )
    code=("scripts/event_uniform_envelope/nested_threshold_cache.sub","scripts/event_uniform_envelope/nested_threshold_cg.sub","scripts/event_uniform_envelope/nested_threshold_freeze.sub","scripts/event_uniform_envelope/heavy_threshold_mip.sub","src/exact_pricer_expanded.py","src/event_pricer_network.py","src/freeze_terminal_exact_cg_pool.py","src/run_exact_pool_mip.py")
    execution={"schema":"evsp-dr-heavy-trip-ladder-raw6-v1","created_utc":datetime.now(timezone.utc).isoformat(),"execution_commit":a.execution_commit,"input_plan_sha256":sha(input_plan_path),"selection_sha256":sha(manifest_path),"duty_order_sha256":sha(order_path),"cells":CELLS,"scales":list(SCALES),"chains":1,"matrix_sha256":sha(matrix),"pool_matrix_sha256":sha(pool),"fresh_runs":6,"reuses_historical_solver_outcomes":False,"representation":REP,"physics":{"battery_kwh":240.0,"charge_kw":240.0,"reserve_kwh":0.0,"soc_step":2.5,"block_min":5,"tariff":"hourly_prices_flat.csv"},"cg":{"master_backend":"gurobi","wall_limit_s":28800,"network_build_excluded":True,"columns_per_iter":30,"column_selection":"reduced_cost","column_pool_treatment":"RAW","checkpoint_every":25,"requeue":True},"mip":{"backend":"gurobi","scientific_time_limit_s_total_by_scale":{"2":28800,"3":28800,"5":28800,"8":3600,"10":3600,"15":3600},"large_stage_limits_s":{"fleet":1800,"cost":1800},"large_conditional_cost":True,"two_stage":True,"threads":8,"memory":"48G","requeue":False,"preemption_semantics":"censored; retry starts a fresh tree; progress is observational only"},"code_sha256":{p:sha(repo/p) for p in code}}
    ep=out/"execution_plan.json"; ep.write_text(json.dumps(execution,indent=2,sort_keys=True)+"\n")
    (out/"PREPARATION_COMPLETE").write_text(f"execution_plan.json {sha(ep)}\nmatrix.tsv {sha(matrix)}\npool_matrix.tsv {sha(pool)}\n")
    print(json.dumps({"root":str(out),"cells":CELLS,"plan_sha256":sha(ep)},sort_keys=True))
if __name__=="__main__": main()
