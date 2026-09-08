#!/usr/bin/env python3
"""Write immutable controls for replacing the legacy sensitivity freeze stage."""

from __future__ import annotations
import argparse,hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path

SOURCE_COMMIT="bead34452aa422ec6b2c7799d0c9c9698208aa32"
SOURCE_PLAN_SHA="9d086701208a679f15f350c522c4e76b62445fea459355544cbd0e624eba1e3a"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(repo,*a): return subprocess.run(["git","-C",str(repo),*a],check=True,text=True,capture_output=True).stdout.strip()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execution-repo",type=Path,required=True)
    ap.add_argument("--execution-commit",required=True)
    ap.add_argument("--campaign-root",type=Path,required=True)
    a=ap.parse_args(); repo=a.execution_repo.resolve(); root=a.campaign_root.expanduser().resolve()
    out=root/"replacement_freeze_plan.json"
    if out.exists(): raise SystemExit(f"replacement plan exists: {out}")
    if git(repo,"rev-parse","HEAD")!=a.execution_commit or git(repo,"status","--porcelain","--untracked-files=no"):
        raise SystemExit("replacement execution checkout identity mismatch")
    sym=subprocess.run(["git","-C",str(repo),"symbolic-ref","-q","HEAD"],capture_output=True)
    if sym.returncode!=1: raise SystemExit("replacement execution checkout must be detached")
    source_plan=root/"execution_plan.json"; pool=root/"pool_matrix.tsv"
    if sha(source_plan)!=SOURCE_PLAN_SHA: raise SystemExit("source execution plan hash mismatch")
    source=json.loads(source_plan.read_text())
    if source.get("execution_commit")!=SOURCE_COMMIT or source.get("cells")!=2 or source.get("pool_matrix_sha256")!=sha(pool):
        raise SystemExit("source campaign identity mismatch")
    worker="scripts/event_uniform_envelope/legacy_selected10_freeze_physics.sub"
    freezer="src/freeze_terminal_exact_cg_pool.py"
    plan={"schema":"evsp-dr-legacy-selected10-freeze-replacement-v1","created_utc":datetime.now(timezone.utc).isoformat(),"source_execution_commit":SOURCE_COMMIT,"source_execution_plan_sha256":SOURCE_PLAN_SHA,"source_pool_matrix_sha256":sha(pool),"replacement_execution_commit":a.execution_commit,"reason":"source freezer accepted only 240/240; replacement validates expected physics per row","cells":[{"index":0,"cell":"legacy_selected10_g240_raw","g_kwh":240.0,"charge_kw":240.0},{"index":1,"cell":"legacy_selected10_g300_raw","g_kwh":300.0,"charge_kw":300.0}],"code_sha256":{worker:sha(repo/worker),freezer:sha(repo/freezer)}}
    out.write_text(json.dumps(plan,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"path":str(out),"sha256":sha(out)},sort_keys=True))
if __name__=="__main__": main()
