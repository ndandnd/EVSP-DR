"""Mock Slurm test: repeated manager invocation submits each of six cells once."""
import os,subprocess,tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as tmp:
 w=Path(tmp); b=w/'bin';b.mkdir()
 (w/'check_native_gate.py').write_text('print("passed mocked gate")\n')
 (b/'sbatch').write_text('#!/bin/bash\nprintf "submit\\n" >> "$TEST_WORK/calls"\nwc -l < "$TEST_WORK/calls"\n')
 (b/'scontrol').write_text('#!/bin/bash\necho "ExcNodeList=scaglione-compute-01"\n')
 for p in b.iterdir():p.chmod(0o755)
 src=Path('/home/nc437/ladder-lite/integer_columns_k15_20260921/submit_wave.sh').read_text()
 src=src.replace('source /etc/profile >/dev/null 2>&1',': # inherited test PATH')
 src=src.replace('/home/nc437/ladder-lite/integer_columns_k15_20260921',str(w))
 (w/'submit.sh').write_text(src)
 env=dict(os.environ,PATH=str(b)+':'+os.environ['PATH'],TEST_WORK=str(w))
 for _ in range(2):subprocess.run(['bash',str(w/'submit.sh')],env=env,check=True,capture_output=True)
 assert len((w/'calls').read_text().splitlines())==6
 assert len((w/'jobs.tsv').read_text().splitlines())==7
 print('PASS: two invocations, exactly six mocked submissions, six unique ledger cells')
