from pathlib import Path
import os,json,subprocess
root=Path('/home/nc437/ladder-lite/terminal_duplicate_cleanup_20260916');m=json.loads((root/'manifest.json').read_text());c=m['cases'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
code=Path('/home/nc437/ladder-lite/code_pins/terminal_cleanup_2c7445ac')
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==m['execution_commit']
assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain','--untracked-files=no'],text=True).strip()
import hashlib
assert hashlib.sha256((Path(c['source']).parent/'selected_routes.json').read_bytes()).hexdigest()==c['selected_source_sha256']
python='/home/nc437/evsp_env/bin/python'
subprocess.run([python,'-m','unittest','discover','-s','tests','-p','test_terminal_cleanup.py'],cwd=code,check=True)
out=root/'results'/c['id']/('job'+os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT','0'))
subprocess.run([python,'-u',str(code/'src/terminal_duplicate_cleanup.py'),'--source',c['source'],'--source-sha256',c['sha256'],'--out',str(out)],check=True)
