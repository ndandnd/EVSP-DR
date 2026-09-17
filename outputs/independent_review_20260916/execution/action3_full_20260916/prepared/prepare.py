"""Prepare local manifests and trip/group inputs only. Never submits jobs."""
import csv,json,shutil
from pathlib import Path
from sequence_replay import ARMS,sha
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'outputs/chain_extension_31_32_20260915').exists())
source=ROOT/'outputs/independent_review_20260916/execution/p1/manifest.json'
case=json.load(open(source))['cases']['warm_w5_k31_seed0_12h']
instance=ROOT/'outputs/chain_extension_31_32_20260915/inputs/w5_k31.csv'
assert sha(instance)==case['input_sha256']
master=ROOT/'outputs/chain_extension_20260913/inputs/sources/Par_VehicleDetails_Updated.csv'
assert sha(master)=='6b46acce8b0870aff967c73aac372b90873ed32a6e424e55b851e4b8676ab57f'
input_rows=list(csv.DictReader(open(instance))); master_rows=list(csv.DictReader(open(master)))
lookup={int(float(r['Ordered_Trip_ID'])):r['VehicleTask'] for r in master_rows if r['Identifier']=='Regular'}
groups={i:('18E1' if lookup[int(float(row['Ordered_Trip_ID']))].startswith('134') else '18E2') for i,row in enumerate(input_rows)}
assert all(lookup[int(float(r['Ordered_Trip_ID']))].startswith(('133','134')) for r in input_rows)
(HERE/'inputs').mkdir(exist_ok=True);shutil.copyfile(instance,HERE/'inputs/w5_k31.csv')
(HERE/'inputs/groups.json').write_text(json.dumps({i:{'group':groups[i],'original_duty':lookup[int(float(r['Ordered_Trip_ID']))],'ordered_trip_id':int(float(r['Ordered_Trip_ID']))} for i,r in enumerate(input_rows)},indent=2)+'\n')
group_inputs={}
for group in ['18E1','18E2']:
 indices=[i for i in range(len(input_rows)) if groups[i]==group]; path=HERE/'inputs'/f'{group}.csv'
 with open(path,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=input_rows[0]);w.writeheader();w.writerows(input_rows[i] for i in indices)
 group_inputs[group]={'path':str(path.relative_to(HERE)),'sha256':sha(path),'global_to_group_local':{i:j for j,i in enumerate(indices)}}
manifest={'schema':'advisor-single-factor-fixed-sequence-v1','finding':'F4','review_item':11,
 'status':'PREPARED_ONLY_DO_NOT_SUBMIT','launch_requires':'user cluster-load confirmation',
 'execution_commit':'50ceb6c095a580f79f87b53bef536cac31f81963',
 'local_code':str(ROOT/'.codex-work/review-strict-chain-20260916'),
 'remote_code':'/home/nc437/ladder-lite/code_pins/review_strict_50ceb6c0',
 'source_p1_manifest_sha256':sha(source),'source_master_sha256':sha(master),
 'source_status':case['source_status'],'source_status_sha256':case['source_status_sha256'],
 'source_journal':'/home/nc437/ladder-lite/chain_extension_31_32_20260915/cases/w5_k31/cg/228610_r0/cg.json.columns.jsonl',
 'source_journal_sha256':case['source_journal_sha256'],
 'source_ordered_pool_sha256':case['comparator']['ordered_pool_sha256'],
 'source_ordered_pool_columns':case['comparator']['pool_columns'],
 'source_cg_commit':case['source_cg_commit'],'instance':'inputs/w5_k31.csv','instance_sha256':sha(instance),
 'group_map':'inputs/groups.json','group_map_sha256':sha(HERE/'inputs/groups.json'),'group_inputs':group_inputs,
 'trip_count':len(input_rows),'arms':ARMS,
 'common':{'soc_step_kwh':2.5,'event_block_min':5,'other_charger_kw':240,'tariff':'hourly_prices_flat.csv',
 'charge_start_fee':5,'bus_cost':100000,'master':'set_covering','start_soc':'full',
 'end_soc':'free above the arm reserve; no separate terminal-energy target',
 'station_capacity':False,'vehicle_type_availability_limits':False,'max_station_wait_min':1560},
 'static_hashes':{Path(p).name:h for p,h in case['static_hashes'].items() if Path(p).suffix=='.csv'},
 'battery_factor_note':'Two homogeneous battery sensitivities, not GIRO class assignments. Offering both without compatibility or quotas reduces to the larger battery in the physical model; source-group segregation is a separate structural arm.',
 'certificate_scope':'Cheapest charging or no path for a fixed ordered sequence in the frozen event graph. Never a full-model pricing certificate or continuous-model infeasibility proof.',
 'proposed_resources':{'partition':'default_partition','exclude':'scaglione-compute-01','extraction_cpus':1,'extraction_memory_gb':16,'replay_cpus':1,'replay_memory_gb':8,
 'replay_sequence_timeout_s':120,'chunk_size':128,'default_concurrency':50,'pilot_sequences_per_arm':20,
 'pilot_allocation_minutes':45,'production_chunk_allocation_hours':6,'cg_followup_hours_per_arm':4,'mip_followup_hours_per_arm':1},
 'plan':'After confirmation only: extract frozen original pool; deterministic 20-sequence/arm pilot; inspect timing and memory; replay all sequences in resumable shards; gate coverage and unresolved counts; then use survivors plus explicitly identified singleton routes for new CG. Preserve all six arms and report preprocessing cost separately.',
 'tooling_sha256':{p.name:sha(p) for p in HERE.glob('*.py')}}
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':manifest['status'],'arms':list(ARMS),'trips':len(input_rows),'jobs_submitted':0}))
