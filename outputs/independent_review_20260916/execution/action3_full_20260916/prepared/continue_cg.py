"""Import an assembled seed into the pinned strict driver; default is prepare-only.

--run-cg is a future authorized worker action, never a scheduler submission.
"""
import argparse,json,subprocess,sys
from pathlib import Path
from sequence_replay import ARMS,sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code',type=Path,required=True)
 p.add_argument('--seed',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--group',choices=['18E1','18E2']);p.add_argument('--allow-unresolved-sequences',action='store_true')
 p.add_argument('--network-cache',type=Path,required=True)
 p.add_argument('--run-cg',action='store_true');p.add_argument('--cg-seconds',type=float,default=14400);p.add_argument('--threads',type=int,default=8);a=p.parse_args()
 a.root=a.root.resolve();a.code=a.code.resolve();a.seed=a.seed.resolve();a.out=a.out.resolve()
 m=json.load(open(a.root/'manifest.json'));receipt=json.load(open(a.seed/'assembly.json'));arm=receipt['arm'];physics=ARMS[arm]
 assert receipt['manifest_sha256']==sha(a.root/'manifest.json') and receipt['cg_seed_ready']
 assert sha(a.seed/'seed_pool.jsonl')==receipt['seed_pool_sha256']
 if receipt['unknown_sequences'] and not a.allow_unresolved_sequences:raise ValueError('Unresolved sequence feasibility: review exclusions, then explicitly pass --allow-unresolved-sequences if justified')
 if physics['segregate']!=bool(a.group):raise ValueError('Only segregation arm must be run as separate 18E1/18E2 components')
 sys.path.insert(0,str(a.code/'src'))
 from run_capacity_speed_event_cg import parser,provenance,checkpoint_id,atomic_pool,run_cg,route_key
 from inherit_capacity_pool import remap_route
 from audit_giro_known_columns import build_problem,STATIONS,HORIZON_MIN
 from run_exact_pool_mip import validate_injected_route
 from utils_v2 import load_station_hourly_prices,base_station_name
 if a.group:
  spec=m['group_inputs'][a.group];instance=a.root/spec['path'];assert sha(instance)==spec['sha256'];mapping={int(k):v for k,v in spec['global_to_group_local'].items()}
 else:instance=a.root/m['instance'];assert sha(instance)==m['instance_sha256'];mapping={i:i for i in range(m['trip_count'])}
 a.out.mkdir(parents=True,exist_ok=False)
 args=parser().parse_args(['--mode','cg','--arm','parx60' if physics['parx_kw']==60 else 'baseline',
 '--instance',str(instance),'--prices',str(a.code/'data/hourly_prices_flat.csv'),'--reference-data-dir',str(a.code/'data'),
 '--out',str(a.out/'cg.json'),'--pool-out',str(a.out/'pool.jsonl'),'--battery-kwh',str(physics['battery_kwh']),
 '--reserve-kwh',str(physics['reserve_kwh']),'--non-parx-kw','240','--max-station-wait-min','1560',
 '--cg-wall-s',str(a.cg_seconds),'--max-iters','100000','--threads',str(a.threads),'--resume','--network-arc-mode','lazy',
 '--network-cache',str(a.network_cache.resolve()),'--checkpoint-interval-s','300','--expected-commit',m['execution_commit'],'--require-clean'])
 # Pinned provenance() reads cwd's Git commit, so use the selected code checkout.
 import os
 os.chdir(a.code)
 prov=provenance(args,instance,args.prices,args.reference_data_dir)
 for name,h in m['static_hashes'].items():assert sha(a.code/'data'/name)==h
 problem=build_problem(instance.parent,instance.name,reference_data_dir=a.code/'data',max_station_to_trip_wait_min=1560)
 identity=checkpoint_id(args,problem,prov);routes=[];keys=set()
 for line in open(a.seed/'seed_pool.jsonl'):
  old=json.loads(line)
  if not set(old['trips']).issubset(mapping):
   if a.group and not set(old['trips']).intersection(mapping):continue
   raise ValueError('mixed/unknown trip IDs in assembled seed')
  route=remap_route(old,mapping,identity)
  reason=validate_injected_route(problem,route,physics['battery_kwh'],240,physics['reserve_kwh'],HORIZON_MIN,arrival_grace_min=0,station_charge_kw={'PARX':physics['parx_kw']})
  if reason is not None:raise ValueError(f'Seed physical replay failed: {reason}')
  key=route_key(route)
  if key not in keys:keys.add(key);routes.append(route)
 assert {t for r in routes for t in r['trips']}==set(problem.trips),'component seed lacks physical coverage'
 atomic_pool(a.out/'pool.jsonl',routes)
 (a.out/'seed_import.json').write_text(json.dumps({'arm':arm,'group':a.group,'assembly_sha256':sha(a.seed/'assembly.json'),'manifest_sha256':sha(a.root/'manifest.json'),
 'seed_routes':len(routes),'seed_pool_sha256':sha(a.out/'pool.jsonl'),'checkpoint_id':identity,'every_route_replayed':True,
 'full_model_pricing_certificate':False,'prior_unknown_sequence_count':receipt['unknown_sequences'],'cg_executed':a.run_cg},indent=2)+'\n')
 if a.run_cg:
  prices=load_station_hourly_prices(args.prices,sorted({base_station_name(s) for s in STATIONS}))
  run_cg(args,problem,prices,prov,a.out/'cg.json',a.out/'pool.jsonl')
if __name__=='__main__':main()
