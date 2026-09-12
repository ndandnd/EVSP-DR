from pathlib import Path
import json,os,runpy,sys,hashlib
p=Path(sys.argv[1]);m=json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
if 'source_status' in m:
 d=json.loads(Path(m['source_status']).read_text());assert d['final']['artificials']==0 and d['final']['iter']>0
 os.environ['EVSP_EXPECTED_COMMIT']=m['execution_commit'];os.environ['EVSP_REQUIRE_DETACHED']='1'
 os.environ['EVSP_MIP_EXPECTED_RESULT_SHA256']=sha(m['source_status']);os.environ['EVSP_MIP_EXPECTED_JOURNAL_SHA256']=sha(d['columns_journal'])
sys.argv=['retry_inherited_import.py',str(p)]
runpy.run_path('/home/nc437/ladder-lite/w2_k14_import_fix_20260912/code/scripts/retry_inherited_import.py',run_name='__main__')
