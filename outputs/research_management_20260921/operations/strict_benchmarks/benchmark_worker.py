import os,sys,time,json,math,subprocess,hashlib
from pathlib import Path
R=Path(sys.argv[1]);case=sys.argv[2];m=json.load(open(R/'manifest.json'));c=next(c for c in m['cases'] if c['id']==case)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p,h in m['input_hashes'].items():assert sha(p)==h,p
for code,commit in m['code_commits'].items():
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()==commit
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=code,text=True).strip()
attempt=os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT','0');o=R/case/attempt;o.mkdir(parents=True,exist_ok=False)
results={};start=time.time()
for label,code,mode in [('original_explicit',m['old_code'],'explicit'),('new_explicit',m['new_code'],'explicit'),('new_packed',m['new_code'],'lazy')]:
 cmd=[sys.executable,str(R/'benchmark_variant.py'),code,mode,str(o/(label+'.json')),c['input']]
 (o/(label+'_command.json')).write_text(json.dumps(cmd))
 with open(o/(label+'.log'),'w') as f:subprocess.run(cmd,cwd=code,stdout=f,stderr=subprocess.STDOUT,check=True)
 results[label]=json.load(open(o/(label+'.json')))
for label in ('new_explicit','new_packed'):
 a=results['original_explicit'];b=results[label]
 assert a['input_sha256']==b['input_sha256']==c['input_sha256']
 assert a['metrics']['event_lattice_sha256']==b['metrics']['event_lattice_sha256']
 assert len(a['pricing'])==len(b['pricing'])==5
 for x,y in zip(a['pricing'],b['pricing']):assert math.isclose(x['rc'],y['rc'],rel_tol=0,abs_tol=1e-7) and x['physical_replay_pass'] and y['physical_replay_pass']
 if label=='new_explicit':assert a['metrics']==b['metrics']
result={'case':c,'host':os.uname().nodename,'attempt':attempt,'all_five_fixed_duals_match':True,'physical_route_replay_all_15':True,'wall_s':time.time()-start,'results':results,'manifest_sha256':sha(R/'manifest.json')}
(o/'result.json').write_text(json.dumps(result,indent=2));(o/'COMPLETE.json').write_text(json.dumps({'result_sha256':sha(o/'result.json')}))
