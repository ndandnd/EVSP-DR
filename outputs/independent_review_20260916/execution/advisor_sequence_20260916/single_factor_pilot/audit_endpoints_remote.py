from pathlib import Path
import json,hashlib,collections
b=Path('/home/nc437/ladder-lite/advisor_sequence_20260916/single_factor_pilot');jobs=json.loads((b/'jobs.json').read_text())['jobs'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rows=[];expected=None
for arm,j in jobs.items():
 a=b/'runs'/arm/(j['job_id']+'_r0');p=a/'pilot_result.json';r=json.loads(p.read_text());e=json.loads((a/'execution.json').read_text());assert e['status']=='completed' and e['pilot_result_sha256']==sha(p);assert r['pilot_manifest_sha256']==sha(b/'pilot_manifest.json')
 out=Path(r['replay_path'])/'outcomes.jsonl';assert sha(out)==r['outcomes_sha256'];records=[json.loads(s) for s in out.read_text().splitlines()];assert len(records)==20;assert dict(collections.Counter(s['status'] for s in records))==r['outcome_counts'];ids=[s['sequence_sha256'] for s in records]
 if expected is None:expected=ids
 else:assert ids==expected
 for s in records:
  if s['status']=='feasible':assert s['physical_replay_validated'] and s['charging_optimal_in_fixed_sequence_event_model']
  assert not s['full_model_pricing_certificate']
 rows.append({'arm':arm,'job_id':j['job_id'],'result_path':str(p),'result_sha256':sha(p),'outcomes_sha256':sha(out),'outcome_counts':r['outcome_counts'],'unknown_count':r['unknown_count'],'elapsed_s':r['elapsed_s'],'stage_resources':[{k:s[k] for k in ['stage','elapsed_s','max_rss_kib']} for s in r['stage_resources']],'sequences':[{'sha256':s['sequence_sha256'],'trips':len(s['trip_sequence']),'status':s['status'],'physical_valid':s['physical_replay_validated']} for s in records]})
print(json.dumps({'finding':'F4','status':'VERIFIED_NATIVE_ENDPOINT_BINDINGS_AND_VALIDATION_FLAGS','scope':'20 deterministic original-pool sequences, not full-pool/fleet success; no-path onlywithin event model','rows':rows},indent=2))
