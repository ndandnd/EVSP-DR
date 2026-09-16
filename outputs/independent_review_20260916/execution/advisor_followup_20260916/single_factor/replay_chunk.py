"""Run one bounded sequence chunk, no scheduler actions; every outcome is recorded."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from sequence_replay import ARMS,replay,sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code',type=Path,required=True)
 p.add_argument('--sequences',type=Path,required=True);p.add_argument('--arm',choices=list(ARMS),required=True)
 p.add_argument('--offset',type=int,default=0);p.add_argument('--limit',type=int,default=128);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--sequence-seconds',type=float,default=120);a=p.parse_args();m=json.load(open(a.root/'manifest.json'))
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.code,text=True).strip()==m['execution_commit']
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=a.code,text=True).strip()
 for name,h in m['static_hashes'].items():assert sha(a.code/'data'/name)==h
 assert sha(a.root/m['instance'])==m['instance_sha256'];assert sha(a.root/m['group_map'])==m['group_map_sha256']
 extraction=json.load(open(a.sequences.parent/'extraction.json'))
 assert sha(a.sequences)==extraction['sequences_sha256']
 assert extraction['source_ordered_pool_sha256']==m['source_ordered_pool_sha256']
 sys.path.insert(0,str(a.code/'src'))
 from audit_giro_known_columns import build_problem,STATIONS
 from event_pricer_network import _event_times,normalize_event_station_prices
 from utils_v2 import load_station_hourly_prices,base_station_name
 instance=a.root/m['instance'];problem=build_problem(instance.parent,instance.name,reference_data_dir=a.code/'data',max_station_to_trip_wait_min=1560)
 prices=load_station_hourly_prices(a.code/'data/hourly_prices_flat.csv',sorted({base_station_name(s) for s in STATIONS}))
 events=_event_times(problem,normalize_event_station_prices(prices,horizon_min=1560,strict_tariff_coverage=False),5)
 groups={int(k):v['group'] for k,v in json.load(open(a.root/m['group_map'])).items()}
 a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();count=0
 receipt={'arm':a.arm,'manifest_sha256':sha(a.root/'manifest.json'),'sequences_sha256':sha(a.sequences),'offset':a.offset,'limit':a.limit,'started_epoch':time.time(),'completed':False}
 (a.out/'attempt.json').write_text(json.dumps(receipt,indent=2))
 with open(a.sequences) as source,open(a.out/'outcomes.jsonl','w') as outcomes,open(a.out/'survivors.jsonl','w') as survivors:
  for index,line in enumerate(source):
   if index<a.offset:continue
   if index>=a.offset+a.limit:break
   seq=json.loads(line);result,route=replay(problem,prices,events,seq['trip_sequence'],ARMS[a.arm],groups,a.sequence_seconds)
   result.update(arm=a.arm,source_sequence=seq)
   if route:
    route.update(origin='original_c5_k31_sequence_charging_reoptimized',found_iter=0,single_factor_source=seq,single_factor_arm=a.arm)
    survivors.write(json.dumps(route)+'\n');survivors.flush()
   outcomes.write(json.dumps(result)+'\n');outcomes.flush();count+=1
 receipt.update(completed=True,sequence_count=count,elapsed_s=time.monotonic()-started,ended_epoch=time.time(),outcomes_sha256=sha(a.out/'outcomes.jsonl'),survivors_sha256=sha(a.out/'survivors.jsonl'))
 (a.out/'COMPLETE.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
