#!/usr/bin/env python3
"""One deterministic retiming order, using only paths from blind_link.py, no saved duty labels."""
import argparse,collections,dataclasses,hashlib,json,pathlib,signal,sys,time
T0=time.monotonic()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out-dir',type=pathlib.Path,required=True)
parser.add_argument('--data',type=pathlib.Path,required=True)
parser.add_argument('--instance',type=pathlib.Path,required=True)
parser.add_argument('--start-fee',type=float,choices=[5],required=True)
args=parser.parse_args()
OUT=args.out_dir;LIMIT=180
receipt={'schema':'blind-link-single-retime-v1','wall_time_limit_seconds':LIMIT,'ordering_attempts_limit':1,'ordering':'path list order (by first-trip start time/id)','witness_data_read':False,'validated':False,'attempts':[],'physics':{'battery_kwh':240,'initial_kwh':240,'charge_kw':240,'station_kw':{'PARX':60},'reserve_kwh':0,'soc_step':2.5,'time_step':5,'start_fee':5,'terminal_target_kwh':0}}
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save():
 receipt['elapsed_seconds']=time.monotonic()-T0
 receipt['code_sha256']=sha(__file__)
 (OUT/'retime_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
def timeout(*_):raise TimeoutError('single pilot wall time limit reached')
signal.signal(signal.SIGALRM,timeout);signal.alarm(LIMIT)
try:
 # --start-fee 5 must be on argv before these imports; modules capture it on import.
 assert sys.argv[sys.argv.index('--start-fee')+1]=='5'
 from run_capacity_terminal_cg import CapacityTerminalNetwork,DEFAULT_PLUGS,audit_energy,problem_kwargs
 from audit_giro_known_columns import build_problem,HORIZON_MIN
 from config import CHARGING_STATIONS
 from capacity_event_pricer import record_occupancy
 from utils_v2 import load_station_hourly_prices
 from run_exact_pool_mip import validate_injected_route
 from station_power import StationPower
 import config,capacity_event_pricer,event_pricer_network,expanded_path_realization
 receipt['captured_fees']={m.__name__:getattr(m,'CHARGE_START_COST',None) for m in [config,capacity_event_pricer,event_pricer_network,expanded_path_realization]}
 assert all(x==5 for x in receipt['captured_fees'].values())
 class FixedSequence(CapacityTerminalNetwork):
  def __init__(self,problem,prices,trips):
   self.following=dict(zip(trips,[*trips[1:],None]));self.first=trips[0]
   super().__init__(dataclasses.replace(problem,trips=tuple(trips)),prices,plugs=DEFAULT_PLUGS,soc_step=2.5,block_min=5,g_kwh=240,charge_kw=240,station_kw={'PARX':60},reserve_kwh=0,strict_tariff_coverage=False)
  def _source_candidates(self):
   for edge in super()._source_candidates():
    if edge[2]==self.first:yield edge
  def _direct_candidates(self,trip,soc_exit,depart):
   for edge in super()._direct_candidates(trip,soc_exit,depart):
    if edge[2]==self.following[trip]:yield edge
  def _charge_candidates(self,trip,soc_exit,depart):
   successor=self.following[trip];outgoing,depot=self.station_trip,self.station_depot
   self.station_trip={station:{successor:trips[successor]} for station,trips in outgoing.items() if successor in trips}
   self.station_depot=depot if successor is None else {}
   try:yield from super()._charge_candidates(trip,soc_exit,depart)
   finally:self.station_trip,self.station_depot=outgoing,depot
 data=args.data;instance=args.instance;price=data/'hourly_prices_flat.csv'
 inputs=[instance,price,data/'Ref_dict.csv',data/'par_ref_dhd.csv']
 receipt['input_sha256']={str(p):sha(p) for p in inputs}
 assert list(receipt['input_sha256'].values())==['c1dbe1dd15d20979799c68d01430a5fd0eddc2ca3ced3f46bd8522123e6c480d','1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200','7bda0e1f439dc8bf5081499566eb2c6a0314190ef27294707f1403fd2c13e3a0','5993e922c671f053611635578b32a1be13bab87b3b5fd8c02b699b81fe0eb66c']
 paths=json.loads((OUT/'paths.json').read_text());receipt['paths_sha256']=sha(OUT/'paths.json')
 problem=build_problem(data,str(instance),**problem_kwargs());prices=load_station_hourly_prices(price,CHARGING_STATIONS)
 receipt['plugs']=DEFAULT_PLUGS;receipt['graph_parameters']=problem_kwargs();receipt['source_code_sha256']={str(pathlib.Path(m.__file__)):sha(m.__file__) for m in [sys.modules['run_capacity_terminal_cg'],sys.modules['audit_giro_known_columns'],capacity_event_pricer,event_pricer_network,expanded_path_realization,sys.modules['run_exact_pool_mip']]}
 occupancy=collections.Counter();selection=[]
 for i,trips in enumerate(paths):
  mark=time.monotonic();attempt={'path_index':i,'trip_count':len(trips),'started_seconds':mark-T0};receipt['attempts'].append(attempt);save()
  net=FixedSequence(problem,prices,trips);attempt['graph_seconds']=time.monotonic()-mark;attempt['arcs']=net.n_arcs
  penalty={key:10000.0 for key,count in occupancy.items() if DEFAULT_PLUGS[key[0]] is not None and count>=DEFAULT_PLUGS[key[0]]}
  candidates=net.capacity_terminal_batch({},penalty,limit=1);attempt['search_seconds']=time.monotonic()-mark-attempt['graph_seconds']
  if not candidates:
   attempt['outcome']='no_reachable_terminal_on_fixed_sequence_event_graph';receipt['outcome']=attempt['outcome'];break
  record=audit_energy(net,candidates[0]['record']);assert record['trips']==trips
  reason=validate_injected_route(problem,record,240,StationPower(240,{'PARX':60}),0,HORIZON_MIN,arrival_grace_min=0)
  assert reason is None,reason
  occ=record_occupancy(record);occupancy.update(occ)
  overload=sum(DEFAULT_PLUGS[s] is not None and count>DEFAULT_PLUGS[s] for (s,_),count in occupancy.items())
  attempt.update(outcome='individually_replayed',terminal_energy_kwh=record['terminal_energy_kwh'],shared_overload_minutes=overload)
  selection.append(record);(OUT/'partial_routes.json').write_text(json.dumps(selection,indent=2)+'\n');save()
  print(json.dumps(attempt),flush=True)
  if overload:receipt['outcome']='greedy_order_shared_capacity_conflict';break
  del net
 else:
  assert sorted(t for r in selection for t in r['trips'])==list(range(173))
  receipt.update(validated=True,outcome='eight_replayed_shared_capacity_feasible',terminal_energy_kwh=sum(r['terminal_energy_kwh'] for r in selection),shared_overload_minutes=0,exact_once=True)
  assert receipt['terminal_energy_kwh']>=-1e-6
 receipt['completed_routes']=len(selection)
except Exception as exc:
 receipt.update(outcome=type(exc).__name__,reason=str(exc))
finally:
 signal.alarm(0);save();print(json.dumps(receipt,indent=2));
