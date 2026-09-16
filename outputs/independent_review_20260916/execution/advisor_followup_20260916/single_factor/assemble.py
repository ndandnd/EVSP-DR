"""Join completed replay chunks, audit coverage, and add ONLY missing singletons.

Does not submit jobs or solve a master. A prepared pool is not an integer solution.
"""
import argparse,collections,json,subprocess,sys
from pathlib import Path
from sequence_replay import ARMS,replay,sha,coverage_gate,canonical

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code',type=Path,required=True)
 p.add_argument('--sequences',type=Path,required=True);p.add_argument('--chunks',type=Path,required=True)
 p.add_argument('--arm',choices=list(ARMS),required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 m=json.load(open(a.root/'manifest.json'));assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.code,text=True).strip()==m['execution_commit']
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=a.code,text=True).strip()
 for name,h in m['static_hashes'].items():assert sha(a.code/'data'/name)==h
 assert sha(a.root/m['group_map'])==m['group_map_sha256']
 extraction=json.load(open(a.sequences.parent/'extraction.json'));assert sha(a.sequences)==extraction['sequences_sha256']
 assert extraction.get('scope','full_source_pool')=='full_source_pool','pilot sequences cannot be assembled as a production seed'
 expected={json.loads(line)['sequence_sha256'] for line in open(a.sequences)};observed={};routes=[];sources=[]
 for path in sorted(a.chunks.glob('*/COMPLETE.json')):
  receipt=json.load(open(path))
  if receipt['arm']!=a.arm:continue
  assert receipt['manifest_sha256']==sha(a.root/'manifest.json')
  assert receipt['sequences_sha256']==extraction['sequences_sha256']
  for name in ['outcomes','survivors']:assert sha(path.parent/f'{name}.jsonl')==receipt[f'{name}_sha256']
  for line in open(path.parent/'outcomes.jsonl'):
   rec=json.loads(line);key=rec['sequence_sha256'];assert key in expected and key not in observed,'missing/duplicate/retried chunks require explicit attempt selection'
   observed[key]=rec
  routes.extend(json.loads(line) for line in open(path.parent/'survivors.jsonl'))
  sources.append({'receipt':str(path),'sha256':sha(path)})
 assert set(observed)==expected,'not all source sequences have completed replay attempts; do not silently truncate'
 feasible_keys={k for k,r in observed.items() if r['status']=='feasible'}
 route_keys=[canonical(r['single_factor_source']['trip_sequence']) for r in routes]
 assert len(set(route_keys))==len(route_keys) and set(route_keys)==feasible_keys
 groups={int(k):v['group'] for k,v in json.load(open(a.root/m['group_map'])).items()}
 before=coverage_gate(observed.values(),range(m['trip_count']));fallback=[]
 if before['missing_trip_ids']:
  sys.path.insert(0,str(a.code/'src'))
  from audit_giro_known_columns import build_problem,STATIONS
  from event_pricer_network import _event_times,normalize_event_station_prices
  from utils_v2 import load_station_hourly_prices,base_station_name
  instance=a.root/m['instance'];assert sha(instance)==m['instance_sha256']
  problem=build_problem(instance.parent,instance.name,reference_data_dir=a.code/'data',max_station_to_trip_wait_min=1560)
  prices=load_station_hourly_prices(a.code/'data/hourly_prices_flat.csv',sorted({base_station_name(s) for s in STATIONS}))
  events=_event_times(problem,normalize_event_station_prices(prices,horizon_min=1560,strict_tariff_coverage=False),5)
  for trip in before['missing_trip_ids']:
   rec,route=replay(problem,prices,events,(trip,),ARMS[a.arm],groups,120)
   rec['origin']='new_singleton_coverage_fallback';fallback.append(rec)
   if route:
    route.update(origin='new_singleton_coverage_fallback',found_iter=0,single_factor_arm=a.arm);routes.append(route)
 after=coverage_gate([*observed.values(),*fallback],range(m['trip_count']))
 counts=collections.Counter(r['status'] for r in observed.values())
 unknown=sum(v for k,v in counts.items() if k.startswith('unknown_'))
 a.out.mkdir(parents=True,exist_ok=False)
 with open(a.out/'seed_pool.jsonl','w') as f:
  for route in routes:f.write(json.dumps(route)+'\n')
 report={'arm':a.arm,'manifest_sha256':sha(a.root/'manifest.json'),'source_ordered_pool_sha256':m['source_ordered_pool_sha256'],
 'sequences_sha256':sha(a.sequences),'all_source_sequences_attempted':True,'outcome_counts':dict(counts),
 'all_source_sequence_feasibility_resolved':unknown==0,'unknown_sequences':unknown,'coverage_before_singletons':before,
 'singleton_fallbacks':fallback,'coverage_after_singletons':after,'source_survivor_routes':len(feasible_keys),
 'total_seed_routes':len(routes),'seed_pool_sha256':sha(a.out/'seed_pool.jsonl'),'source_receipts':sources,
 'cg_seed_ready':after['all_trips_covered_by_physical_routes'],'full_model_pricing_certificate':False,
 'finite_pool_mip_proof':False,'seed_pool_is_integer_solution':False,
 'interpretation':'Survivors are validated individual routes. Their RMP objective is not a full-model lower bound. Unresolved sequences are excluded from this seed, not declared physically infeasible. A feasible integer selection would give an upper bound only.'}
 (a.out/'assembly.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'arm':a.arm,'counts':dict(counts),'cg_seed_ready':report['cg_seed_ready']}))
if __name__=='__main__':main()
