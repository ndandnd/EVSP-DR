"""Freeze a random trip-group control with the same final input as chain1 stage15."""
from pathlib import Path
import csv,json,random,hashlib,io,datetime
P=Path(__file__).resolve().parent;ROOT=P.parents[3];SRC=ROOT/'outputs/chain_extension_20260913/inputs/sources';SEED=20260916

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(p.open()))
final_path=SRC/'Practice_Custom_DutyUnion_k15_p01_20260908.csv'; final=read(final_path)
master=read(SRC/'Par_VehicleDetails_Updated.csv'); duty={int(r['Ordered_Trip_ID']):r['VehicleTask'] for r in master if r['Identifier']=='Regular'}
order=sorted([r for r in read(SRC/'chain_order.csv') if r['family_replicate']=='1'],key=lambda r:int(r['addition_rank']))
assert len(order)==15
original_groups=[[r for r in final if duty[int(r['Ordered_Trip_ID'])]==x['duty_id']] for x in order]
assert sum(map(len,original_groups))==len(final)
rows=sorted(final,key=lambda r:int(r['Ordered_Trip_ID']));random.Random(SEED).shuffle(rows)
fields=list(final[0]);cases={};cursor=0;assignment=[]
for stage,group in enumerate(original_groups,1):
 for r in rows[cursor:cursor+len(group)]:assignment.append(dict(ordered_trip_id=int(r['Ordered_Trip_ID']),random_group=stage,original_giro_duty=duty[int(r['Ordered_Trip_ID'])]))
 cursor+=len(group)
 if stage<2:continue
 selected=[dict(r) for r in rows[:cursor]];selected.sort(key=lambda r:(sum(int(a)*b for a,b in zip(r['Start1'].split(':'),[60,1])),int(r['Ordered_Trip_ID'])))
 for i,r in enumerate(selected):r['count_trip_id']=str(i)
 cid=f'r1_s{stage:02d}';path=P/'inputs'/f'{cid}.csv'
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(selected)
 cases[cid]=dict(id=cid,chain=1,stage=stage,trip_count=len(selected),added_trips=len(group) if stage>2 else cursor,reference_original_chain_stage=stage,target_buses=None,final_giro_reference_count=15 if stage==15 else None,input_sha256=sha(path),csv=path.name,previous_case=f'r1_s{stage-1:02d}' if stage>2 else None,distinct_original_giro_duties_represented=len({duty[int(r['Ordered_Trip_ID'])] for r in selected}))
assert {int(r['Ordered_Trip_ID']) for r in selected}=={int(r['Ordered_Trip_ID']) for r in final}
# All physical trip attributes match the original final instance; local row IDs may differ.
attrs=lambda rr:{int(r['Ordered_Trip_ID']):{k:v for k,v in r.items() if k!='count_trip_id'} for r in rr}
assert attrs(selected)==attrs(final)
with (P/'trip_group_assignment.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(assignment[0]));w.writeheader();w.writerows(assignment)
manifest=dict(schema='evsp-random-trip-groups-input-v1',prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=SEED,random_method='Sort final input by Ordered_Trip_ID, shuffle with Python random.Random(20260916), cut into original C1 duty trip counts; solve cumulative groups2..15.',source_hashes={str(x.relative_to(ROOT)):sha(x) for x in [final_path,SRC/'Par_VehicleDetails_Updated.csv',SRC/'chain_order.csv']},final_instance_sha256=sha(final_path),final_trip_attributes_identical=True,groups=[dict(group=i+1,size=len(g),matched_original_duty=order[i]['duty_id']) for i,g in enumerate(original_groups)],cases=cases,warm_chains={'1':list(cases)},interpretation='stage index is NOT a fleet target; intermediate sets mix trips from many GIRO duties. Only stage15 has the original 15-duty reference.',prepare_script_sha256=sha(Path(__file__)))
(P/'inputs/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print([(x['stage'],x['trip_count'],x['distinct_original_giro_duties_represented']) for x in cases.values()])
