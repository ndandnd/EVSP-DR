from pathlib import Path
import sys,json,hashlib,subprocess,datetime
q=json.load(sys.stdin);base=Path('/home/nc437/ladder-lite/review_full40_20260916');dest=base/'scaglione_12h_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not dest.exists(),'Immutable deployment directory already exists; inspect instead of overwriting'
dest.mkdir()
for name,content in q['files'].items():
 (dest/name).write_text(content)
 assert sha(dest/name)==q['sha256'][name]
m=json.loads((dest/'manifest.json').read_text())
assert subprocess.check_output(['git','-C',str(base/'code'),'rev-parse','HEAD'],text=True).strip()==m['execution_commit']
assert not subprocess.check_output(['git','-C',str(base/'code'),'status','--porcelain','--untracked-files=no'],text=True).strip()
for name,h in m['tooling_sha256'].items():assert sha(base/name)==h,(name,sha(base/name),h)
for name,h in m['data_sha256'].items():assert sha(base/'code/data'/name)==h
case=m['cases']['c1_full40'];assert sha(base/'code/data'/case['csv'])==case['input_sha256']
slurm='/usr/local/slurm/slurm-25.05.5/bin/'
snapshot={j:subprocess.check_output([slurm+'scontrol','show','job',j,'-o'],text=True).strip() for j in ['341404_0','341405','341406']}
assert 'Reason=JobHeldUser' in snapshot['341405'] and 'Reason=JobHeldUser' in snapshot['341406']
policy=Path('/home/nc437/ladder-lite/SCAGLIONE_RESOURCE_POLICY.md')
out={'deployed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote_version':str(dest),'files_sha256':q['sha256'],'code_commit':m['execution_commit'],'code_clean':True,'original_files_untouched':True,'policy_sha256':sha(policy),'policy_text':policy.read_text(),'scheduler_before':snapshot,'nodes':subprocess.check_output([slurm+'sinfo','-N','-p','scaglione','-o','%N|%t|%C|%m|%G'],text=True),'sstat_graph':subprocess.check_output([slurm+'sstat','-j','341404.batch','--format=JobID,MaxRSS,AveRSS,MaxVMSize','-P'],text=True)}
(dest/'deployment_receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
