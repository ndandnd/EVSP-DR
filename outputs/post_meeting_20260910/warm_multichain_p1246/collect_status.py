#!/usr/bin/env python3
"""Collect immutable scheduler/result status for the inherited multichain batch."""
from __future__ import annotations
import argparse, csv, datetime as dt, hashlib, json, subprocess
from pathlib import Path

SCHEMA="evsp-dr-inherited-multichain-status-v1"
def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def run(args):
    return subprocess.run(args,text=True,capture_output=True,check=False)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('batch',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); root=args.batch.resolve(); output=args.output.resolve()
    if output.exists(): raise SystemExit(f'refusing existing output: {output}')
    manifest_path=root/'batch_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    record_path=root/'cg_jobs.tsv'
    records=[]
    if record_path.exists():
        with record_path.open(newline='') as f: records=list(csv.DictReader(f,delimiter='\t'))
    ids=[r['job_id'] for r in records if r.get('job_id','').isdigit()]
    squeue={}
    if ids:
        proc=run(['/usr/local/slurm/slurm-25.05.5/bin/squeue','-h','-j',','.join(ids),'-o','%A|%T|%M|%R|%N'])
        for line in proc.stdout.splitlines():
            fields=line.split('|',4)
            if len(fields)==5:squeue[fields[0]]=dict(zip(('state','elapsed','reason','node'),fields[1:]))
    sacct={}
    if ids:
        proc=run(['/usr/local/slurm/slurm-25.05.5/bin/sacct','-X','-j',','.join(ids),'--format=JobIDRaw,State,Elapsed,ExitCode,MaxRSS','-n','-P'])
        for line in proc.stdout.splitlines():
            fields=line.strip().split('|')
            if len(fields)>=5 and fields[0] in ids:
                sacct[fields[0]]=dict(zip(('state','elapsed','exit_code','max_rss'),fields[1:5]))
    rows=[]
    exclusions_ok=0
    for record in records:
        rep=int(record['replicate']); scale=int(record['scale']); jid=record['job_id']
        out=root/f'p{rep}'/'cg'/f'M__k{scale:02d}_p{rep}__warm_cover__event_2p5_event5.json'
        effective={}
        if jid.isdigit():
            proc=run(['/usr/local/slurm/slurm-25.05.5/bin/scontrol','show','job','-o',jid])
            for token in proc.stdout.strip().split():
                if '=' in token:
                    k,v=token.split('=',1)
                    if k in {'Partition','ExcNodeList','NumCPUs','MinMemoryNode','TimeLimit','Requeue','Dependency'}: effective[k]=v
        exclude_ok=effective.get('ExcNodeList')=='scaglione-compute-01'
        exclusions_ok += int(exclude_ok)
        rows.append({
            **record,'squeue':squeue.get(jid),'sacct':sacct.get(jid),
            'effective_slurm':effective,'reserved_node_excluded':exclude_ok,
            'result_path':str(out),'result_exists':out.is_file(),
            'result_sha256':sha(out) if out.is_file() else None,
            'journal_path':str(out)+'.columns.jsonl',
            'journal_exists':Path(str(out)+'.columns.jsonl').is_file(),
        })
    payload={
      'schema':SCHEMA,'collected_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
      'batch_root':str(root),'batch_manifest_path':str(manifest_path),
      'batch_manifest_sha256':sha(manifest_path),'cg_record_path':str(record_path),
      'cg_record_sha256':sha(record_path) if record_path.is_file() else None,
      'jobs_expected':36,'jobs_recorded':len(records),
      'effective_reserved_node_exclusion_count':exclusions_ok,
      'all_recorded_jobs_exclude_reserved_node':bool(records) and exclusions_ok==len(records),
      'records':rows,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'output':str(output),'sha256':sha(output),'jobs':len(records),'exclusions_ok':exclusions_ok},sort_keys=True))
if __name__=='__main__': main()
