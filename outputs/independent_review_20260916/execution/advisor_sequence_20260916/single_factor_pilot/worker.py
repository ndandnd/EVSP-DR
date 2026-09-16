"""Bounded one-arm pilot. Never submits jobs, performs CG, or solves a MIP."""
import argparse,collections,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True);a=p.parse_args()
root=a.root.resolve();m=json.loads((root/'pilot_manifest.json').read_text());prepared=root/'prepared';code=Path(m['code'])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(prepared/'manifest.json')==m['prepared_manifest_sha256']
for name,h in json.loads((prepared/'manifest.json').read_text())['tooling_sha256'].items():assert sha(prepared/name)==h
attempt=f"{os.environ.get('SLURM_JOB_ID','local')}_r{os.environ.get('SLURM_RESTART_COUNT','0')}"
work=root/'runs'/a.arm/attempt;work.mkdir(parents=True,exist_ok=False);start=time.monotonic()
record={'arm':a.arm,'attempt':attempt,'job_id':os.environ.get('SLURM_JOB_ID'),'started_epoch':time.time(),'pilot_manifest_sha256':sha(root/'pilot_manifest.json'),'stages':[],'status':'running'}
def save(): (work/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
save()
def stage(name,args,limit):
 t=time.monotonic();timefile=work/f'{name}.time';cmd=['/usr/bin/time','-v','-o',str(timefile),sys.executable,*args]
 with open(work/f'{name}.log','w') as f:
  result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=limit)
 txt=timefile.read_text() if timefile.exists() else '';match=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',txt)
 rec={'stage':name,'command':cmd,'returncode':result.returncode,'elapsed_s':time.monotonic()-t,'max_rss_kib':int(match[1]) if match else None,'log_sha256':sha(work/f'{name}.log')}
 record['stages'].append(rec);save()
 if result.returncode:raise RuntimeError(f'{name} returned {result.returncode}')
try:
 if a.arm=='baseline':
  extracted=work/'extracted'
  stage('extract',[str(prepared/'extract_sequences.py'),'--root',str(prepared),'--code',str(code),'--out',str(extracted)],1200)
 else:
  assert a.arm in m['conditional_arms']
  approved=json.loads((root/'control_pass.json').read_text());assert approved['pass'] and approved['pilot_manifest_sha256']==sha(root/'pilot_manifest.json')
  extracted=Path(approved['extracted']);assert sha(extracted/'pilot/sequences.jsonl')==approved['pilot_sequences_sha256']
 pilot=extracted/'pilot/sequences.jsonl'
 extraction=json.loads((extracted/'extraction.json').read_text());pilotmeta=json.loads((pilot.parent/'extraction.json').read_text())
 assert extraction['source_columns']==254068 and extraction['scope']=='full_source_pool'
 assert extraction['source_ordered_pool_sha256']==json.loads((prepared/'manifest.json').read_text())['source_ordered_pool_sha256']
 assert sha(extracted/'sequences.jsonl')==extraction['sequences_sha256']
 rows=[json.loads(line) for line in open(extracted/'sequences.jsonl')]
 ranked=sorted(rows,key=lambda r:(len(r['trip_sequence']),max(r['source_charge_start_counts']),r['sequence_sha256']))
 ix=sorted({round(i*(len(ranked)-1)/19) for i in range(20)})
 expected=[ranked[i]['sequence_sha256'] for i in ix];actual=[json.loads(line)['sequence_sha256'] for line in open(pilot)]
 assert len(actual)==20 and actual==expected and sha(pilot)==pilotmeta['sequences_sha256']
 del rows,ranked
 stage('replay',[str(prepared/'replay_chunk.py'),'--root',str(prepared),'--code',str(code),'--sequences',str(pilot),'--arm',a.arm,'--offset','0','--limit','20','--sequence-seconds','120','--out',str(work/'replay')],2700)
 outcomes=[json.loads(line) for line in open(work/'replay/outcomes.jsonl')]
 assert len(outcomes)==20 and [r['sequence_sha256'] for r in outcomes]==expected
 unknown=[r for r in outcomes if r['status'].startswith('unknown_')]
 counts=dict(collections.Counter(r['status'] for r in outcomes))
 elapsed=time.monotonic()-start;resource_ok=all(s['max_rss_kib'] is not None and s['max_rss_kib']<=m['control_pass_gate']['max_stage_rss_kib'] for s in record['stages']) and elapsed<=4500
 all_feasible=all(r['status']=='feasible' and r['physical_replay_validated'] for r in outcomes)
 gate={'arm':a.arm,'pass':a.arm=='baseline' and all_feasible and resource_ok and not unknown,'pilot_manifest_sha256':sha(root/'pilot_manifest.json'),
 'source_pool_hash_verified':True,'source_pool_columns':254068,'deterministic_pilot_verified':True,'pilot_sequences':20,'pilot_sequences_sha256':sha(pilot),
 'all_control_sequences_feasible':all_feasible,'outcome_counts':counts,'unknown_count':len(unknown),'resource_ok':resource_ok,'elapsed_s':elapsed,
 'stage_resources':record['stages'],'extracted':str(extracted),'replay_path':str(work/'replay'),'outcomes_sha256':sha(work/'replay/outcomes.jsonl'),
 'pilot_trip_coverage':len({t for r in outcomes if r['status']=='feasible' for t in r['trip_sequence']}),'full_instance_trip_count':716,
 'coverage_scope':'20 sampled sequences only; not full source pool or full-instance coverage','full_replay_authorized':False,'cg_mip_authorized':False}
 (work/'pilot_result.json').write_text(json.dumps(gate,indent=2)+'\n')
 if gate['pass']:
  assert not (root/'control_pass.json').exists(),'existing control pass must not be replaced'
  (root/'control_pass.json').write_text(json.dumps(gate,indent=2)+'\n')
 record.update(status='completed',ended_epoch=time.time(),elapsed_s=elapsed,pilot_result_sha256=sha(work/'pilot_result.json'));save()
 print(json.dumps({'arm':a.arm,'pass':gate['pass'],'outcomes':counts,'elapsed_s':elapsed}))
except BaseException as exc:
 record.update(status='failed',ended_epoch=time.time(),elapsed_s=time.monotonic()-start,error=f'{type(exc).__name__}: {exc}');save();raise
