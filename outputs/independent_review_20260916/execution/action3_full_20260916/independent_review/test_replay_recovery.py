from pathlib import Path
import sys,tempfile,json,hashlib
B=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(B));import replay_journal as j
root=Path(tempfile.mkdtemp(prefix='action3-journal-review-'));source=[{'sequence_sha256':'first','trip_sequence':[0]},{'sequence_sha256':'second','trip_sequence':[1]}];arm='baseline'
def record(i):
 outcome={'sequence_sha256':source[i]['sequence_sha256'],'source_sequence':source[i],'arm':arm,'status':'unknown_timeout'};r={'outcome':outcome,'route':None};r['record_sha256']=j.canonical(r);return r
a=json.dumps(record(0)).encode();b=json.dumps(record(1)).encode();tests=[]
def run(name,raw,success,n=None):
 d=root/name;d.mkdir();p=d/'records.jsonl';p.write_bytes(raw)
 try:
  rows=j.recover(p,source,arm,d)
 except ValueError:
  if success:raise
  assert p.read_bytes()==raw,'rejected journal changed'
 else:
  assert success and len(rows)==n,(name,rows)
 tests.append({'test':name,'passed':True})
 return p
run('complete_two_records',a+b'\n'+b+b'\n',True,2)
p=run('valid_final_without_newline',a,True,1);assert p.read_bytes().endswith(b'\n');p.write_bytes(p.read_bytes()+b+b'\n');assert len(j.recover(p,source,arm,p.parent))==2
run('torn_final_repaired',a+b'\n'+b'{"partial":',True,1)
run('malformed_interior_rejected',a+b'\n'+b'{bad\n'+b+b'\n',False)
run('malformed_complete_final_rejected',a+b'\n'+b'{bad\n',False)
r=record(1);r['outcome']['status']='feasible';run('checksum_corruption_rejected',a+b'\n'+json.dumps(r).encode()+b'\n',False)
run('out_of_order_rejected',b+b'\n'+a+b'\n',False)
case=root/'complete';case.mkdir();expected={'arm':arm,'sequence_count':2,'full_campaign_manifest_sha256':'m','source_shard_sha256':'s'}
for name in ['records.jsonl','outcomes.jsonl','survivors.jsonl']:(case/name).write_text('test\n')
receipt={**expected,**{field:j.sha(case/name) for field,name in [('records_sha256','records.jsonl'),('outcomes_sha256','outcomes.jsonl'),('survivors_sha256','survivors.jsonl')]}}
j.atomic_json(case/'COMPLETE.json',receipt);assert j.completed(case,expected);tests.append({'test':'valid_atomic_completion','passed':True})
for name in ['torn_receipt','wrong_count','changed_output']:
 j.atomic_json(case/'COMPLETE.json',receipt)
 if name=='torn_receipt':(case/'COMPLETE.json').write_text('{')
 if name=='wrong_count':j.atomic_json(case/'COMPLETE.json',{**receipt,'sequence_count':1})
 if name=='changed_output':(case/'survivors.jsonl').write_text('changed')
 try:j.completed(case,expected);raise AssertionError('bad completion accepted')
 except ValueError:pass
 tests.append({'test':name+'_rejected','passed':True})
print(json.dumps({'tests':tests,'files_sha256':{n:hashlib.sha256((B/n).read_bytes()).hexdigest() for n in ['replay_worker.py','replay_journal.py','prepare_shards.py']},'no_solver_or_scheduler_calls':True},indent=2))
