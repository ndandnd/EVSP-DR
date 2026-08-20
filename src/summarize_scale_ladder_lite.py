#!/usr/bin/env python3
"""Normalize any completed subset of a ladder-lite campaign."""
from __future__ import annotations
import argparse,csv,hashlib,json,shutil,tempfile
from pathlib import Path
import summarize_scale_ladder as base
ALLOW_CENSORED=set()

def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _write(path,fields,rows):
    with Path(path).open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields,extrasaction="ignore",lineterminator="\n")
        w.writeheader();w.writerows(rows)
def _validate(job,_plan_sha):
    out=Path(job["output"]); phase=job["phase"]
    if not out.exists() or (not Path(str(out)+".done").is_file()
                            and job["job_key"] not in ALLOW_CENSORED):
        raise ValueError(f"ladder-lite completion missing: {job['job_key']}")
    if phase=="PREFLIGHT":
        payload=json.loads(out.read_text()); observed=hashlib.sha256(
            json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
        if not out.with_suffix(".csv").is_file() or observed!=job.get("prelaunch_membership_sha256"):
            raise ValueError("PREFLIGHT output/hash invalid")
    if phase in {"CG","CG_SENSITIVITY"}:
        status=json.loads(out.read_text()); provenance=status.get("provenance") or {}
        if (provenance.get("git_commit")!=PLAN["checkout_identity"]["commit"]
                or status.get("stop_reason") in {None,"resume_starting"}
                or not Path(str(out)+".columns.jsonl").is_file()
                or not Path(str(out)+".iters.csv").is_file()):
            raise ValueError(f"CG identity/artifacts invalid: {job['job_key']}")
        if job.get("telemetry"):
            with Path(job["telemetry"]).open() as h:
                for line in h:
                    if line.strip():json.loads(line)
        for mark in job["snapshot_minutes"]:
            snap=out.parent/f"{out.stem}.m{int(mark)}.snapshot.json"; journal=Path(str(snap)+".columns.jsonl")
            reported=(status.get("snapshot_availability") or {}).get(str(int(mark)))
            if (reported=="available")!=(snap.is_file() and journal.is_file()):
                raise ValueError(f"CG snapshot mismatch: {job['job_key']} m{mark}")
            if reported not in {"available","censored_solver_terminated_before_mark","missed_in_prior_allocation"}:
                raise ValueError("CG snapshot classification missing")
            if reported!="available" and (snap.exists() or journal.exists()):
                raise ValueError(f"CG censored snapshot orphan: {job['job_key']} m{mark}")
    if phase=="MIP":
        result=json.loads(out.read_text()); provenance=result.get("mip_provenance") or {}
        arguments=provenance.get("arguments") or {}; progress=Path(job["progress_dir"])
        if (provenance.get("expected_git_commit")!=PLAN["checkout_identity"]["commit"]
                or provenance.get("observed_git_commit")!=PLAN["checkout_identity"]["commit"]
                or provenance.get("final_observed_git_commit")!=PLAN["checkout_identity"]["commit"]
                or provenance.get("git_dirty") is not False
                or provenance.get("tracked_clean_at_end") is not True
                or arguments.get("two_stage") is not True or arguments.get("cover") is not False
                or int(arguments.get("threads",-1))!=int(job["threads"])
                or int(arguments.get("timelimit",-1))!=int(job["budget_s"])
                or float(arguments.get("mipgap",-1))!=0.0001
                or not (progress/"final.json").is_file()):
            raise ValueError(f"MIP identity/progress invalid: {job['job_key']}")
        schedule=(result.get("progress") or {}).get("checkpoint_schedule_s")
        if not isinstance(schedule,list):raise ValueError("MIP checkpoint schedule missing")
        for mark in schedule:
            path=progress/f"checkpoint_{int(round(float(mark)/60)):04d}m.json"
            if not path.is_file():raise ValueError(f"MIP checkpoint missing: {path}")
def _append_missing(output,omitted):
    paths=[output/"cg_run_summary.csv",output/"mip_run_summary.csv",output/"scale_progress_summary.csv"]
    tables=[];fields=[]
    for path in paths:
        with path.open(newline="") as h:
            r=csv.DictReader(h);tables.append(list(r));fields.append(r.fieldnames)
    cg,mip,progress=tables
    for job,state,reason in omitted:
        common={"cell_id":job["cell_id"],"scale":job["scale"],
          "selection_replicate":job["selection_replicate"],"cg_replicate":job["cg_replicate"],
          "budget_s":job["budget_s"],"soc_step":job["soc_step"],"block_min":job["block_min"],
          "target_fleet":job["target_fleet"],"instance_file_sha256":job["instance"]["instance_file_sha256"],
          "trip_identity_schema":job["instance"]["trip_identity_schema"],"censored":True}
        if job["phase"] in {"CG","CG_SENSITIVITY"}:
            cg.append({**common,"campaign_role":"primary" if job["phase"]=="CG" else "small_grid_sensitivity",
                       "stopping_reason":state,"grid_interpretation":reason})
            if job["phase"]=="CG":progress.append({**common,"missing_reason":reason,
                                                  "cg_stopping_reason":state,"cg_censored":True})
        elif job["phase"]=="MIP":
            mip.append({**common,"arm":job["arm"],"scientific_role":job.get("scientific_role"),
                        "output_available":False,"missing_reason":reason})
    for path,field,rows in zip(paths,fields,(cg,mip,progress)):_write(path,field,rows)
def _mark_censored(output,jobs):
    for name in ("cg_run_summary.csv","mip_run_summary.csv","scale_progress_summary.csv"):
        path=output/name
        with path.open(newline="") as h:r=csv.DictReader(h);rows=list(r);fields=r.fieldnames
        for row in rows:
            matches=[j for j in jobs if (name.startswith("cg_") and j["phase"] in {"CG","CG_SENSITIVITY"} and j["cell_id"]==row.get("cell_id") and ("primary" if j["phase"]=="CG" else "small_grid_sensitivity")==row.get("campaign_role") and str(j["soc_step"])==row.get("soc_step") and str(j["block_min"])==row.get("block_min") and str(j["cg_replicate"])==row.get("cg_replicate")) or (name.startswith("mip_") and j["phase"]=="MIP" and j["cell_id"]==row.get("cell_id") and j["arm"]==row.get("arm") and str(j["cg_replicate"])==row.get("cg_replicate")) or (name.startswith("scale_") and j["phase"]=="CG" and str(j["scale"])==row.get("scale") and str(j["selection_replicate"])==row.get("selection_replicate") and str(j["cg_replicate"])==row.get("cg_replicate"))]
            if matches:
                row["cg_censored" if name.startswith("scale_") else "censored"]="True"
                if "stopping_reason" in row:row["stopping_reason"]="censored: output present without .done"
                if "cg_stopping_reason" in row:row["cg_stopping_reason"]="censored"
                if "missing_reason" in row:row["missing_reason"]="censored: output present without .done"
        _write(path,fields,rows)
def summarize(campaign_root,output_dir):
    root=Path(campaign_root).resolve();output=Path(output_dir).resolve()
    raw=(root/"approved-plan.json").read_bytes();original=json.loads(raw)
    manifest=json.loads((root/"campaign.json").read_text())
    if (hashlib.sha256(raw).hexdigest()!=manifest.get("approval_sha256")
            or manifest.get("execution_mode")!="ladder_lite_direct_array"
            or manifest.get("commit")!=original["checkout_identity"]["commit"]):
        raise ValueError("ladder-lite approval/commit mismatch")
    completed=[];omitted=[];censored_jobs=[]
    global PLAN;PLAN=original
    for job in original["jobs"]:
        out=Path(job["output"])
        if Path(str(out)+".override.json").exists():omitted.append((job,"excluded","excluded: budget_overridden"));continue
        state=("completed" if Path(str(out)+".done").is_file() else "censored" if out.exists()
               else "blocked" if Path(str(out)+".blocked").exists()
               else "failed" if Path(str(out)+".failed").exists() else "missing")
        if state=="censored":
            ALLOW_CENSORED.add(job["job_key"])
            try:_validate(job,"")
            except (OSError,ValueError,KeyError,json.JSONDecodeError):omitted.append((job,state,f"{state}: unusable partial output"))
            else:completed.append(job);censored_jobs.append(job)
        elif state=="completed":completed.append(job)
        else:omitted.append((job,state,f"{state}: ladder-lite marker/output state"))
    filtered=dict(original);filtered["jobs"]=completed
    filtered["task_groups"]={g:[k for k in keys if any(j["job_key"]==k for j in completed)]
                             for g,keys in original["task_groups"].items()}
    filtered["execution_mode"]="local_diagnostic";temp=Path(tempfile.mkdtemp(prefix="ladder-lite-normalize-"))
    try:
        plan_raw=json.dumps(filtered,sort_keys=True,separators=(",",":")).encode()
        (temp/"approved-plan.json").write_bytes(plan_raw)
        (temp/"campaign.json").write_text(json.dumps({"approval_sha256":hashlib.sha256(plan_raw).hexdigest(),
          "execution_mode":"local_diagnostic","diagnostic_only":False,"submitted":False}))
        saved=base._validate_completion;base._validate_completion=_validate
        try:base.summarize(temp,output)
        finally:base._validate_completion=saved
        _append_missing(output,omitted);_mark_censored(output,censored_jobs)
        for path in (p for p in output.iterdir() if p.suffix in {".csv",".json"}):
            path.write_text(path.read_text().replace('"local_diagnostic"','"ladder_lite_direct_array"').replace("known_duties_contained_fallback_grid","declared_resolution_scale_grid"))
        if any(b'"local_diagnostic"' in p.read_bytes() for p in output.iterdir() if p.is_file()):raise ValueError("local_diagnostic provenance survived lite normalization")
        provenance_path=output/"provenance.json"
        provenance=json.loads(provenance_path.read_text())
        provenance.update({"plan_sha256":hashlib.sha256(raw).hexdigest(),
          "git_commit":original["checkout_identity"]["commit"],
          "execution_mode":"ladder_lite_direct_array","provenance":"ladder_lite_direct_array"})
        provenance["output_sha256"]={p.name:_sha(p) for p in output.iterdir()
                                     if p.is_file() and p!=provenance_path}
        provenance_path.write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n")
    finally:shutil.rmtree(temp)
    return {"completed":len(completed),"omitted":len(omitted),"output":str(output)}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--campaign-root",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True);a=p.parse_args()
    print(json.dumps(summarize(a.campaign_root,a.out_dir),indent=2))
if __name__=="__main__":main()
