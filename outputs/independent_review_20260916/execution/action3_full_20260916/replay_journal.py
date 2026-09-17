"""Strict append-journal recovery and atomic completion publication."""
import hashlib,json,os
from pathlib import Path
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def atomic_json(path,value):
 path=Path(path);tmp=path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
 with open(tmp,'w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def recover(path,source,arm,audit):
 path=Path(path);done=[]
 if not path.exists():return done
 with open(path,'rb+') as f:
  good=0
  while line:=f.readline():
   try:row=json.loads(line)
   except (ValueError,UnicodeDecodeError):
    tail=f.read()
    if tail or line.endswith(b'\n'):raise ValueError('interior/corrupt complete journal record; refusing truncation')
    (Path(audit)/'interrupted_suffix.bin').write_bytes(line);f.truncate(good);f.flush();os.fsync(f.fileno());break
   if len(done)>=len(source):raise ValueError('journal longer than source shard')
   if row.get('record_sha256')!=canonical({'outcome':row['outcome'],'route':row['route']}):raise ValueError('journal record checksum mismatch')
   outcome=row['outcome'];src=source[len(done)]
   if outcome['sequence_sha256']!=src['sequence_sha256'] or outcome['arm']!=arm or outcome['source_sequence']!=src:raise ValueError('journal source/arm/order mismatch')
   if (outcome['status']=='feasible')!=(row['route'] is not None):raise ValueError('journal survivor/outcome mismatch')
   done.append(row);good=f.tell()
   if not line.endswith(b'\n'):
    f.seek(0,os.SEEK_END);f.write(b'\n');f.flush();os.fsync(f.fileno());good=f.tell()
 return done
def completed(case,expected):
 path=Path(case)/'COMPLETE.json'
 if not path.exists():return False
 try:receipt=json.loads(path.read_text())
 except (ValueError,UnicodeDecodeError) as e:raise ValueError('corrupt completion receipt; refusing completion shortcut') from e
 for key,value in expected.items():
  if receipt.get(key)!=value:raise ValueError('completion identity/count mismatch: '+key)
 for field,name in [('outcomes_sha256','outcomes.jsonl'),('survivors_sha256','survivors.jsonl'),('records_sha256','records.jsonl')]:
  if sha(Path(case)/name)!=receipt[field]:raise ValueError('completion output hash mismatch: '+name)
 return True
