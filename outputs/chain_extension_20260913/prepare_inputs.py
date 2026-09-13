#!/usr/bin/env python3
"""Deterministically extend frozen GIRO prefixes. Python standard library only."""
import csv, hashlib, io, itertools, json, random, re
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'inputs'
SRC = OUT / 'sources'
SEED = 20260913
PREPARATION_BASE_COMMIT = 'd3aba8787e097e7da482c7fbcc5e22e83b395df4'
COLS = ['Identifier','From1','Start1','End1','To1','Distance1','Usage kWh','count_trip_id','Ordered_Trip_ID']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(newline='') as f: return list(csv.DictReader(f))
def base(d): return re.match(r'\d+',d).group()
def minute(t):
    h,m = map(int,t.split(':')); return h*60+m

def main():
    master = read(SRC/'Par_VehicleDetails_Updated.csv')
    frames = {}
    for r in master:
        if r['Identifier']=='Regular' and r['Ordered_Trip_ID']:
            frames.setdefault(r['VehicleTask'],[]).append({c:r[c] for c in COLS})
    variants={b:sorted(d for d in frames if base(d)==b) for b in sorted({base(d) for d in frames})}
    assert len(frames)==42 and len(variants)==40
    original=read(SRC/'chain_order.csv')
    plan=json.loads((SRC/'input_plan.json').read_text())
    for f,h in plan['files'].items(): assert sha(SRC/f)==h
    parent_manifest=json.loads((SRC/'parent_campaign_manifest.json').read_text())
    results={r['case']:r for r in read(SRC/'parent_integer_results.csv')}
    manifest={'schema':'evsp-chain-extension-inputs-v1','preparation_base_commit':PREPARATION_BASE_COMMIT,
      'preparation_script_sha256':sha(__file__),'solver_execution_commit':'set in separate launch manifest before execution',
      'source_remote_root':'/home/nc437/ladder-lite/full_pool_recovery_20260912/code',
      'source_hashes':{p.name:sha(p) for p in sorted(SRC.iterdir()) if p.is_file()},
      'selection_seed':SEED,'random_method':'Python random.Random(20260913 + chain).shuffle(sorted unused numeric base duties)',
      'base_duty_universe':variants,'variant_policy':'Preserve parent variants; select lexicographically first pair for bases 13316 and 13324 whose suffix character sets intersect. This infers compatible service-day labels, without asserting a uniquely identified weekday when parent evidence is ambiguous.',
      'scientific_settings':parent_manifest['scientific_settings'],'launch_through_k':25,'membership_through_k':40,
      'input_only':True,'chains':{},'cases':{},'warm_chains':{}}
    audit=[]
    for chain in range(1,7):
        cid=f'w{chain}_k15'; parent=parent_manifest['cases'][cid]
        pf=SRC/Path(parent['csv']).name
        assert sha(pf)==parent['input_sha256']==results[cid]['input_sha256']
        prefix=[r['duty_id'] for r in sorted((r for r in original if int(r['family_replicate'])==chain),key=lambda r:int(r['addition_rank']))]
        assert len(prefix)==15 and len({base(d) for d in prefix})==15
        options=[pair for pair in itertools.product(variants['13316'],variants['13324']) if set(pair[0][5:]) & set(pair[1][5:]) and all(d in pair for d in prefix if base(d) in ('13316','13324'))]
        assert options
        choice=min(options)
        selected={b:ds[0] for b,ds in variants.items()}
        selected.update({base(d):d for d in choice})
        assert all(selected[base(d)]==d for d in prefix)
        unused=sorted(set(variants)-{base(d) for d in prefix})
        random.Random(SEED+chain).shuffle(unused)
        additions=[selected[b] for b in unused]
        rows=read(pf); previous={r['Ordered_Trip_ID']:r for r in rows}
        assert len(previous)==len(rows)
        source_prefix={r['Ordered_Trip_ID']:r for d in prefix for r in frames[d]}
        assert set(source_prefix)==set(previous)
        for tid,r in previous.items():
            for c in COLS:
                if c!='count_trip_id': assert r[c]==source_prefix[tid][c],(chain,tid,c)
        manifest['chains'][str(chain)]={'parent_case':cid,'parent_csv':str(pf.relative_to(OUT)),
          'parent_remote_csv':manifest['source_remote_root']+'/data/'+parent['csv'],'parent_sha256':sha(pf),
          'parent_status':f'/home/nc437/ladder-lite/full_pool_recovery_20260912/cases/{cid}/cg.json',
          'parent_cg_payload_sha256':results[cid]['cg_payload_sha256'],
          'original_addition_order':prefix,'extension_seed':SEED+chain,'unused_base_order':unused,'added_duty_order':additions,
          'compatible_variant_options':options,'selected_variant_pair':choice,'variant_choice_ambiguous':len(options)>1,
          'resolved_40_duty_universe':[selected[b] for b in sorted(selected)]}
        manifest['warm_chains'][str(chain)]=[]
        for k,duty in enumerate(additions,16):
            added=frames[duty]
            assert not set(previous)&{r['Ordered_Trip_ID'] for r in added}
            rows=rows+[dict(r) for r in added]
            rows.sort(key=lambda r:(minute(r['Start1']),int(r['Ordered_Trip_ID'])))
            for n,r in enumerate(rows): r['count_trip_id']=str(n)
            now={r['Ordered_Trip_ID']:dict(r) for r in rows}
            assert len(now)==len(rows)
            assert set(previous)<set(now)
            for tid,r in previous.items():
                assert all(r[c]==now[tid][c] for c in COLS if c!='count_trip_id')
            duties=prefix+additions[:k-15]
            assert len(duties)==len({base(d) for d in duties})==k
            cid=f'w{chain}_k{k:02d}'; fp=OUT/(cid+'.csv')
            stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=COLS,lineterminator='\n');writer.writeheader();writer.writerows(rows)
            payload=stream.getvalue().encode()
            if fp.exists(): assert fp.read_bytes()==payload, f'Refuse changed existing input {fp}'
            else: fp.write_bytes(payload)
            rec={'id':cid,'chain':chain,'k':k,'target_duties':k,'csv':fp.name,'input_sha256':sha(fp),'trip_count':len(rows),'added_duty':duty,'added_base_duty':base(duty),'added_trip_count':len(added),'duties':duties,'previous_case':f'w{chain}_k{k-1:02d}','previous_input_sha256':sha(pf) if k==16 else manifest['cases'][f'w{chain}_k{k-1:02d}']['input_sha256'],'stage_enabled':k<=25,'stable_trip_id_nesting_verified':True,'previous_trip_attributes_unchanged_except_count_trip_id':True,'unique_base_duties_verified':True,'unique_trip_ids_verified':True}
            manifest['cases'][cid]=rec
            if k<=25: manifest['warm_chains'][str(chain)].append(cid)
            audit.append({x:rec[x] for x in ('id','chain','k','added_duty','added_trip_count','trip_count','input_sha256','previous_case','stage_enabled')})
            previous={t:dict(r) for t,r in now.items()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (OUT/'membership.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(audit[0]),lineterminator='\n');w.writeheader();w.writerows(audit)
    (OUT/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(OUT)}\n' for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    print(json.dumps({'generated_cases':len(audit),'staged_cases':sum(r['stage_enabled'] for r in audit),'manifest_sha256':sha(OUT/'manifest.json'),'validations':'all passed'}))
if __name__=='__main__': main()
