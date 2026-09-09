#!/usr/bin/env python3
"""Continue the original probability sampler through p20; publish k3/k5/k6."""
from __future__ import annotations
import argparse,csv,json,random,shutil,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from scripts.event_uniform_envelope import build_small_threshold_inputs as base
SEED=20260904
SCALES=(3,5,6)
REPLICATES=tuple(range(7,21))
SOURCE=REPO/'data/scale_ladder/instances/threshold_9_15_20260904'
ORIGINAL=REPO/'data/scale_ladder/instances/nested_probability_k2_15_20260908'
DEFAULT_OUTPUT=REPO/'data/scale_ladder/instances/nested_replication_p7_20_20260909'
def read(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def draw_orders(available,excluded):
    # Match the original probability phase exactly, including its k9--15
    # rejection domain. Structural-candidate generation is not rerun here.
    seen={k:set() for k in range(9,16)};orders=[];attempts=[]
    for replicate in range(1,21):
        rng=random.Random(SEED*1000+replicate)
        for attempt in range(1,100001):
            ordered=rng.sample(available,15)
            if len({base._base_task(d) for d in ordered})!=15:continue
            prefixes={k:tuple(sorted(ordered[:k])) for k in range(9,16)}
            if any(v in excluded or v in seen[k] for k,v in prefixes.items()):continue
            break
        else:raise ValueError('unable to draw chain '+str(replicate))
        for k,v in prefixes.items():seen[k].add(v)
        orders.append(ordered);attempts.append(attempt)
    return orders,attempts

def build(output):
    if output.exists():raise FileExistsError(output)
    if not output.is_relative_to(REPO):raise ValueError('output must be inside repository')
    source_plan_path=SOURCE/'input_plan.json';source_plan=json.loads(source_plan_path.read_text())
    if source_plan['schema']!='evsp-dr-threshold-9-15-inputs-v1':raise ValueError('wrong source plan')
    for name,digest in source_plan['files'].items():
        if base.sha256(SOURCE/name)!=digest:raise ValueError('source artifact mismatch: '+name)
    for key,path in [('source_master_sha256',REPO/source_plan['source_master']),('reference_sha256',REPO/'data/Ref_dict.csv'),('deadhead_sha256',REPO/'data/par_ref_dhd.csv'),('tariff_sha256',REPO/source_plan['tariff'])]:
        if base.sha256(path)!=source_plan[key]:raise ValueError('source physics input changed: '+key)
    certificates={r['duty_id']:r for r in read(SOURCE/'known_duty_continuous_240_240.csv')}
    if any(r['continuous_physical_feasible_240_240']!='True' or r['physical_replay_status']!='validated' for r in certificates.values()):raise ValueError('uncertified eligibility entry')
    exclusion_names=['excluded_existing_scale_ladder_manifest.csv','excluded_small_threshold_manifest.csv']
    excluded=set().union(*(base.existing_duty_sets(SOURCE/name) for name in exclusion_names))
    orders,attempts=draw_orders(sorted(certificates),excluded)
    old_plan=json.loads((ORIGINAL/'input_plan.json').read_text())
    for name in ('selection_manifest.csv','chain_order.csv'):
        if base.sha256(ORIGINAL/name)!=old_plan['files'][name]:raise ValueError('original nested artifact mismatch')
    old_orders={r:[] for r in range(1,7)}
    for row in read(ORIGINAL/'chain_order.csv'):old_orders[int(row['family_replicate'])].append(row['duty_id'])
    if any(orders[r-1]!=old_orders[r] for r in old_orders):raise ValueError('original six ordered draws did not replay')
    original_rows=read(ORIGINAL/'selection_manifest.csv')
    frames=base.load_duty_frames();output.mkdir(parents=True)
    for name in ['known_duty_continuous_240_240.csv',*exclusion_names]:shutil.copyfile(SOURCE/name,output/name)
    rows=[];chain_rows=[]
    for replicate,ordered in enumerate(orders,1):
        for rank,duty in enumerate(ordered,1):chain_rows.append(dict(chain_id=f'nested_probability_{replicate}',family_replicate=replicate,addition_rank=rank,duty_id=duty,chain_seed=SEED*1000+replicate,accepted_draw_attempt=attempts[replicate-1],cohort='replayed_original' if replicate<=6 else 'new_replication'))
        if replicate not in REPLICATES:continue
        for scale in SCALES:
            duties=tuple(sorted(ordered[:scale]));destination=output/f'Practice_Custom_DutyUnion_k{scale:02d}_p{replicate:02d}_20260909.csv'
            base.merge_duties(frames,list(duties)).to_csv(destination,index=False,lineterminator='\n')
            rows.append(dict(cell_id=f'k{scale:02d}_p{replicate}',scale=scale,selection_replicate=replicate,sample_family='probability',family_replicate=replicate,selection_role=f'fixed_seed_probability_{replicate}',nested_chain_id=f'nested_probability_{replicate}',nested_parent_cell_id=f'k{scale-1:02d}_p{replicate}',addition_rank_duty=ordered[scale-1],relative_path=str(destination.relative_to(REPO)),instance_file_sha256=base.sha256(destination),duties_json=json.dumps(duties,separators=(',',':')),duty_set_sha256=base.canonical_sha(duties),target_fleet=scale,known_partition_continuous_physical_upper_bound=True,known_duty_certificate_set_sha256=base.canonical_sha(sorted(certificates[d]['certificate_sha256'] for d in duties)),**base.candidate_features(frames,list(duties)),direct_compatibility_density=round(base.direct_compatibility_density(destination),12)))
    rows.sort(key=lambda r:(r['scale'],r['family_replicate']))
    duplicates=[]
    for scale in SCALES:
        seen={tuple(json.loads(r['duties_json'])):r['cell_id'] for r in original_rows if int(r['scale'])==scale}
        for row in [r for r in rows if r['scale']==scale]:
            key=tuple(json.loads(row['duties_json']))
            if key in seen:duplicates.append([seen[key],row['cell_id']])
            seen[key]=row['cell_id']
    write(output/'selection_manifest.csv',rows);write(output/'chain_order.csv',chain_rows)
    plan=dict(schema='evsp-dr-nested-replication-p7-20-inputs-v1',created_date='2026-09-09',generator_seed=SEED,scales=list(SCALES),replicates=list(REPLICATES),probability_per_scale=14,selected_rows=42,full_order_replicates=list(range(1,21)),selection_uses_solver_outcomes=False,sampling_rule='exact original ordered15 draw; reject duplicate base IDs and k9..15 prefixes present in original exclusions or preceding probability chains',original_six_orders_replayed=True,lower_prefix_duplicate_audit=duplicates,lower_prefix_duplicates_resampled=False,source_input_plan_sha256=base.sha256(source_plan_path),original_nested_plan_sha256=base.sha256(ORIGINAL/'input_plan.json'),generator_script_sha256=base.sha256(Path(__file__)),shared_builder_sha256=base.sha256(Path(base.__file__)),known_partition_scope=source_plan['known_partition_scope'],known_partition_caveat=source_plan['known_partition_caveat'],physics=source_plan['physics'],source_master_sha256=source_plan['source_master_sha256'],reference_sha256=source_plan['reference_sha256'],deadhead_sha256=source_plan['deadhead_sha256'],tariff_sha256=source_plan['tariff_sha256'],files={name:base.sha256(output/name) for name in ['selection_manifest.csv','chain_order.csv','known_duty_continuous_240_240.csv',*exclusion_names]})
    (output/'input_plan.json').write_text(json.dumps(plan,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),new_chains=14,full_orders=20,original_orders_replayed=6,lower_prefix_duplicates=duplicates,output=str(output))))
    return plan
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();build(a.output_dir.resolve())
