from pathlib import Path
import sys,json,tempfile
sys.path.insert(0,'/home/nc437/ladder-lite/review_full40_20260916/code/src')
from exact_pricer_expanded import load_iteration_log,ITERATION_LOG_HEADER
from durable_io import read_jsonl_records,DurableFileError
p=Path(tempfile.mkdtemp(prefix='evsp-checkpoint-tail-'));good=(ITERATION_LOG_HEADER+'\n4.2,3,100000,1,0,-0.1,5\n').encode();f=p/'tail.csv';f.write_bytes(good+b'99,');r=load_iteration_log(f,repair_trailing=True);assert len(r)==1 and f.read_bytes()==good
bad=good+b'invalid,row\n5.2,4,99999,1,0,-0.01,6\n';g=p/'interior.csv';g.write_bytes(bad)
try:load_iteration_log(g,repair_trailing=True)
except DurableFileError:pass
else:raise AssertionError('interior CSV corruption accepted')
assert g.read_bytes()==bad
j=p/'no_newline.jsonl';j.write_bytes(b'{"v":1}');assert read_jsonl_records(j,repair_trailing=True)==[{'v':1}] and j.read_bytes()==b'{"v":1}\n'
print(json.dumps({'iteration_partial_tail_repaired':True,'iteration_interior_corruption_rejected_unchanged':True,'journal_complete_last_record_newline_restored':True,'temp_root':str(p),'production_modified':False},indent=2))
