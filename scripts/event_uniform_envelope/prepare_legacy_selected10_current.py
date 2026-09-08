#!/usr/bin/env python3
"""Prepare hash-bound 240/240 and 300/300 current-algorithm replays."""

from __future__ import annotations
import argparse,csv,hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path

PHYSICS=((240,240),(300,300)); CELLS=2
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
    source=repo/"data/scale_ladder/instances/legacy_selected10_20260426"
    input_plan_path=source/"input_plan.json"; manifest_path=source/"selection_manifest.csv"
    plan=json.loads(input_plan_path.read_text()); rows=list(csv.DictReader(manifest_path.open(newline="")))
    if (plan.get("schema")!="evsp-dr-legacy-selected10-input-v1" or plan.get("selected_rows")!=1
        or sha(manifest_path)!=plan["files"]["selection_manifest.csv"] or len(rows)!=1):
        raise SystemExit("input plan identity mismatch")
    source_row=rows[0]; instance=repo/source_row["relative_path"]
    if sha(instance)!=source_row["instance_file_sha256"]: raise SystemExit("instance hash mismatch")
    for d in ("network_cache","cg","snapshots","records","mip","progress","logs/cache","logs/cg","logs/freeze","logs/mip"):
        (out/d).mkdir(parents=True,exist_ok=True)
    (out/"input_selection_manifest.csv").write_bytes(manifest_path.read_bytes())
    matrix=out/"matrix.tsv"; pool=out/"pool_matrix.tsv"
    fields=("index","cell","scale","replicate","trips","representation","instance_relative_to_data","instance_sha256","base_status","resume_status","snapshot","record","mip_output")
    with matrix.open("x",newline="") as f, pool.open("x",newline="") as g:
        mw=csv.writer(f,delimiter="\t",lineterminator="\n")
        pw=csv.DictWriter(g,fieldnames=fields,delimiter="\t",lineterminator="\n"); pw.writeheader()
        for i,(battery,charge) in enumerate(PHYSICS):
            cell=f"legacy_selected10_g{battery}_raw"; rep=f"event_2p5_event5_g{battery}"
            stem=f"M__{cell}__{rep}.json"; status=out/"cg"/stem
            mw.writerow([i,cell,10,1,source_row["trip_count"],str(instance),source_row["instance_file_sha256"],rep,"2.5","5","28800",battery,charge])
            pw.writerow(dict(zip(fields,[i,cell,10,1,source_row["trip_count"],rep,source_row["relative_path"].removeprefix("data/"),source_row["instance_file_sha256"],status,status,out/"snapshots"/stem,out/"records"/(stem+".freeze.json"),out/"mip"/(stem+".raw_pool_mip8h.json")])) )
    code=("scripts/event_uniform_envelope/legacy_selected10_cache.sub","scripts/event_uniform_envelope/legacy_selected10_cg.sub","scripts/event_uniform_envelope/nested_threshold_freeze.sub","scripts/event_uniform_envelope/nested_threshold_mip8h.sub","src/exact_pricer_expanded.py","src/event_pricer_network.py","src/freeze_terminal_exact_cg_pool.py","src/run_exact_pool_mip.py")
    execution={"schema":"evsp-dr-legacy-selected10-current-sensitivity-v1","created_utc":datetime.now(timezone.utc).isoformat(),"execution_commit":a.execution_commit,"input_plan_sha256":sha(input_plan_path),"selection_sha256":sha(manifest_path),"cells":CELLS,"matrix_sha256":sha(matrix),"pool_matrix_sha256":sha(pool),"reuses_historical_solver_outcomes":False,"comparison_scope":"same 175-trip input and current algorithm/event grid; 240/240 versus 300/300 physics sensitivity, not a May-code replication","physics":[{"battery_kwh":x,"charge_kw":y,"reserve_kwh":0.0,"soc_step":2.5,"block_min":5,"tariff":"hourly_prices_flat.csv"} for x,y in PHYSICS],"cg":{"master_backend":"gurobi","wall_limit_s":28800,"network_build_excluded":True,"columns_per_iter":30,"column_selection":"reduced_cost","column_pool_treatment":"RAW","checkpoint_every":25,"requeue":True},"mip":{"backend":"gurobi","scientific_time_limit_s_total":28800,"two_stage":True,"threads":8,"memory":"48G","requeue":False},"code_sha256":{p:sha(repo/p) for p in code}}
    ep=out/"execution_plan.json"; ep.write_text(json.dumps(execution,indent=2,sort_keys=True)+"\n")
    (out/"PREPARATION_COMPLETE").write_text(f"execution_plan.json {sha(ep)}\nmatrix.tsv {sha(matrix)}\npool_matrix.tsv {sha(pool)}\n")
    print(json.dumps({"root":str(out),"cells":CELLS,"plan_sha256":sha(ep)},sort_keys=True))
if __name__=="__main__": main()
