from pathlib import Path
import sys,json,random,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from make_duty_pair_instances import load_duty_frames,merge_duties,_base_task,_peak_concurrency
frames=load_duty_frames(); unique={}
for duty in sorted(frames): unique.setdefault(_base_task(duty),duty)
selected=sorted(random.Random(20260912).sample(sorted(unique.values()),32))
out=ROOT/'data/overnight_decomposition_20260912';out.mkdir(exist_ok=False)
def emit(name,duties):
 f=out/(name+'.csv');df=merge_duties(frames,duties);df.to_csv(f,index=False,lineterminator='\n')
 assert len(set(df.Ordered_Trip_ID))==len(df)
 return {'id':name,'csv':str(f.relative_to(ROOT/'data')),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'duties':duties,'trip_count':len(df),'peak_overlap':_peak_concurrency(df)}
parent=emit('parent32',selected);parts=[]
for variant in range(10):
 if variant==0:
  order=sorted(selected,key=lambda d:(len(frames[d]),d));groups=[order[i*8:(i+1)*8] for i in range(4)];method='trip_count_contiguous'
 elif variant==1:
  groups=[[] for _ in range(4)];loads=[0]*4
  for duty in sorted(selected,key=lambda d:(-len(frames[d]),d)):
   i=min((i for i in range(4) if len(groups[i])<8),key=lambda i:(loads[i],i));groups[i].append(duty);loads[i]+=len(frames[duty])
  method='trip_count_balanced'
 else:
  order=selected.copy();random.Random(20260912+variant).shuffle(order);groups=[order[i*8:(i+1)*8] for i in range(4)];method='seeded_random'
 assert sorted(sum(groups,[]))==selected
 records=[emit(f'd{variant:02d}_g{i}',g) for i,g in enumerate(groups)]
 parts.append({'partition':variant,'method':method,'seed':20260912+variant if variant>=2 else None,'groups':records})
manifest={'schema':'evsp-dr-decomposition-poc-v1','selection_seed':20260912,'parent':parent,'partitions':parts,'source_sha256':hashlib.sha256((ROOT/'data/Par_VehicleDetails_Updated.csv').read_bytes()).hexdigest(),'knowledge_used':'GIRO duty membership selects disjoint trip subsets; no GIRO route columns injected','physics':{'battery_kwh':240,'charge_kw':240,'reserve_fraction':0,'shared_capacity':False,'terminal_soc_floor':0,'master_sense':'cover','tariff':'flat','soc_step_kwh':2.5,'event_block_min':5}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'parent_trips':parent['trip_count'],'groups':40,'min_group_trips':min(r['trip_count'] for p in parts for r in p['groups']),'max_group_trips':max(r['trip_count'] for p in parts for r in p['groups'])}))
