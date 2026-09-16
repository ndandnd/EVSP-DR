from pathlib import Path
import os,json,subprocess
root=Path('/home/nc437/ladder-lite/terminal_exact_once_20260916');m=json.loads((root/'manifest.json').read_text());c=m['cases'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
code=Path('/home/nc437/ladder-lite/code_pins/terminal_partition_c92a34ea')
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==m['execution_commit']
assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain','--untracked-files=no'],text=True).strip()
python='/home/nc437/evsp_env/bin/python'
subprocess.run([python,'-m','unittest','discover','-s','tests','-p','test_terminal_partition.py'],cwd=code,check=True)
out=root/'results'/c['id']/('job'+os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT','0'))
subprocess.run([python,'-u',str(code/'src/validate_terminal_pool_partition.py'),'--source',c['source'],'--source-sha256',c['sha256'],'--out',str(out)],check=True)
