import json,os,sys,hashlib,subprocess,datetime
from pathlib import Path
root=Path('/home/nc437/ladder-lite/zero_fee_full_cg_20260916')
m=json.loads((root/'manifest.json').read_text());case=m['cases'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
code=Path(m['code'])
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==m['execution_commit']
assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain','--untracked-files=no'],text=True).strip()
for p,h in case['input_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
attempt=root/'results'/case['id']/('job'+os.environ['SLURM_JOB_ID']+'_restart'+os.environ.get('SLURM_RESTART_COUNT','0'))
attempt.mkdir(parents=True,exist_ok=False)
meta=dict(case=case['id'],execution_commit=m['execution_commit'],manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='running',output=str(attempt/'optimization'))
(attempt/'attempt.json').write_text(json.dumps(meta,indent=2))
argv=['/home/nc437/evsp_env/bin/python','-u',str(code/'src/run_terminal_energy_cg.py'),'--data-dir',case['data_dir'],'--csv',case['instance'],'--prices',case['tariff'],'--out',str(attempt/'optimization')]
result=subprocess.run(argv)
meta.update(state='finished' if result.returncode==0 else 'failed',returncode=result.returncode,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(attempt/'attempt.json').write_text(json.dumps(meta,indent=2))
sys.exit(result.returncode)
