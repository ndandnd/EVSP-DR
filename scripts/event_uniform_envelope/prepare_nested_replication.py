#!/usr/bin/env python3
"""Prepare immutable controls for fresh p7--p20 k3/k5/k6 replication runs."""

from __future__ import annotations
import argparse,csv,hashlib,json,subprocess
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

SCALES=(3,5,6); CELLS=42; REP="event_2p5_event5"
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
    source=repo/"data/scale_ladder/instances/nested_replication_p7_20_20260909"
    input_plan_path=source/"input_plan.json"; manifest_path=source/"selection_manifest.csv"; order_path=source/"chain_order.csv"
    plan=json.loads(input_plan_path.read_text()); rows=list(csv.DictReader(manifest_path.read_text().splitlines()))
    if (plan.get("schema")!="evsp-dr-nested-replication-p7-20-inputs-v1" or plan.get("selected_rows")!=CELLS
        or sha(manifest_path)!=plan["files"]["selection_manifest.csv"] or sha(order_path)!=plan["files"]["chain_order.csv"]):
        raise SystemExit("input plan identity mismatch")
    rows.sort(key=lambda r:(int(r["scale"]),int(r["family_replicate"])))
    if Counter(int(r["scale"]) for r in rows)!=Counter({k:14 for k in SCALES}): raise SystemExit("scale counts mismatch")
    if len({r["cell_id"] for r in rows})!=CELLS: raise SystemExit("duplicate cell")
    if {(int(r['scale']),int(r['family_replicate'])) for r in rows}!={(k,p) for k in SCALES for p in range(7,21)}:
        raise SystemExit("replicate grid mismatch")
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
            pw.writerow(dict(zip(fields,[i,r["cell_id"],r["scale"],r["family_replicate"],r["trip_count"],REP,r["relative_path"].removeprefix("data/"),r["instance_file_sha256"],status,status,out/"snapshots"/stem,out/"records"/(stem+".freeze.json"),out/"mip"/(stem+".raw_pool_mip8h.json")])) )
    code=("scripts/event_uniform_envelope/nested_threshold_cache.sub","scripts/event_uniform_envelope/nested_threshold_cg.sub","scripts/event_uniform_envelope/nested_threshold_freeze.sub","scripts/event_uniform_envelope/nested_replication_mip8h.sub","src/exact_pricer_expanded.py","src/event_pricer_network.py","src/freeze_terminal_exact_cg_pool.py","src/run_exact_pool_mip.py")
    execution={"schema":"evsp-dr-nested-replication-p7-20-fresh42-v1","created_utc":datetime.now(timezone.utc).isoformat(),"execution_commit":a.execution_commit,"input_plan_sha256":sha(input_plan_path),"selection_sha256":sha(manifest_path),"chain_order_sha256":sha(order_path),"cells":CELLS,"scales":list(SCALES),"chains":14,"matrix_sha256":sha(matrix),"pool_matrix_sha256":sha(pool),"fresh_runs":42,"replicates":list(range(7,21)),"scheduler_nice":10000,"cache_cg_concurrency":12,"mip_concurrency":2,"mip_partition":"scaglione","mip_excluded_nodes":["scaglione-compute-01","scaglione-cpu-04"],"reuses_historical_solver_outcomes":False,"representation":REP,"physics":{"battery_kwh":240.0,"charge_kw":240.0,"reserve_kwh":0.0,"soc_step":2.5,"block_min":5,"tariff":"hourly_prices_flat.csv"},"cg":{"master_backend":"gurobi","wall_limit_s":28800,"network_build_excluded":True,"columns_per_iter":30,"column_selection":"reduced_cost","column_pool_treatment":"RAW","checkpoint_every":25,"requeue":True},"mip":{"backend":"gurobi","scientific_time_limit_s_total":28800,"two_stage":True,"threads":8,"memory":"48G","requeue":False,"preemption_semantics":"censored; retry starts a fresh tree; progress is observational only"},"code_sha256":{p:sha(repo/p) for p in code}}
    ep=out/"execution_plan.json"; ep.write_text(json.dumps(execution,indent=2,sort_keys=True)+"\n")
    (out/"PREPARATION_COMPLETE").write_text(f"execution_plan.json {sha(ep)}\nmatrix.tsv {sha(matrix)}\npool_matrix.tsv {sha(pool)}\n")
    print(json.dumps({"root":str(out),"cells":CELLS,"plan_sha256":sha(ep)},sort_keys=True))
if __name__=="__main__": main()
