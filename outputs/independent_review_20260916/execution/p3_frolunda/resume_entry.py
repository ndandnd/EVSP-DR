"""Copy hashed native checkpoints into a fresh attempt, then run the pinned CG adapter."""
from pathlib import Path
import json,hashlib,shutil,sys,os
p=json.loads(sys.argv[1]);sha=lambda s:hashlib.sha256(Path(s).read_bytes()).hexdigest()
assert sha(p['from'])==p['source_sha256'];assert sha(p['from']+'.columns.jsonl')==p['journal_sha256']
for suffix in ['', '.columns.jsonl','.iters.csv']:
 old=Path(p['from']+suffix)
 if old.exists():shutil.copy2(old,p['to']+suffix)
os.execv(sys.executable,[sys.executable,*sys.argv[2:]])
