from pathlib import Path
import json,hashlib,os,time
root=Path('/home/nc437/ladder-lite/mip_structure_20260922');m=json.loads((root/'manifest.json').read_text());case=next(c for c in m['cases'] if c['id']=='c3_k15_fresh');end=json.loads(Path(case['endpoint']).read_text());path=Path(end['source_journal']);h=hashlib.sha256();pool={};raw=0
for b in path.open('rb'):
 h.update(b)
 if not b.strip():continue
 r=json.loads(b);raw+=1;key=frozenset(r['trips']);cost=float(r['cost'])
 if key not in pool or cost<pool[key]['total']-1e-9:pool[key]={'ordinal':raw,'total':cost,'variable':cost-100000.0,'cost_repr':repr(cost),'trip_count':len(r['trips']),'fields':{k:r.get(k) for k in ['cost_semantics','master_cost_semantics','expanded_grid_cost','continuous_realized_cost','origin','found_iter']},'charging_stops':r.get('charging_stops')}
assert h.hexdigest()==end['source_journal_sha256'];negative=[r for r in pool.values() if r['variable']<0];out={'source':str(path),'source_sha256':h.hexdigest(),'raw_records':raw,'unique_incidence_columns':len(pool),'negative_count':len(negative),'min_variable_cost':min(r['variable'] for r in pool.values()),'max_variable_cost':max(r['variable'] for r in pool.values()),'negative_max':max((r['variable'] for r in negative),default=None),'representative_negative':sorted(negative,key=lambda r:r['variable'])[:5],'nominal_subtracted_bus_cost':100000,'scope':'Exact source coefficients; no clamping or model mutation','created_unix':time.time()};dest=root/'diagnostics'/('costs_'+os.environ['SLURM_JOB_ID']+'.json');dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
