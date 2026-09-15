"""Exact native-incidence union with donor witness and immutable source audits."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math,resource,sqlite3,time
import common as w
IDENTITY=('csv','trip_ids','g_kwh','charge_kw','min_soc_frac','soc_step','block_min','time_model','master_sense','prices_csv')
HASHES=('instance_sha256','prices_sha256','reference_sha256','deadhead_sha256','git_commit')
PRODUCERS={'e091a4dba549510238507ef5e5367abea958bd30','a0e0bb7681c8451e3cbbbfa06aef390026d9af4b'}
BASELINE={'g_kwh':240,'charge_kw':240,'min_soc_frac':0,'soc_step':2.5,'block_min':5,'time_model':'event','master_sense':'cover','prices_csv':'hourly_prices_flat.csv'}
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def incidence(r):return canonical(sorted(r['trips']))
def route_hash(r):return digest({k:r.get(k) for k in ['trips','route_nodes','charging_stops','cost']})
def identities(values):
 first=values[0]
 for v in values:
  for k in IDENTITY:
   if v.get(k)!=first.get(k) or k not in v:raise ValueError('source identity mismatch '+k)
  for k in HASHES:
   if not v['provenance'].get(k) or v['provenance'][k]!=first['provenance'][k]:raise ValueError('provenance mismatch '+k)
  if v['provenance']['git_commit'] not in PRODUCERS:raise ValueError('unreviewed producer')
  for k,x in BASELINE.items():
   if v.get(k)!=x:raise ValueError('nonbaseline physics '+k)
  for k in ['capacity_enforced','shared_station_capacity','charger_counts','terminal_soc_floor','station_power_kw']:
   if v.get(k) not in (None,False,{},[]):raise ValueError('unsupported constraint '+k)
 return first

def validate_source(s):
 for p,h in [('cg_completion_path','cg_completion_sha256'),('mip_completion_path','mip_completion_sha256'),('status_path','status_sha256'),('journal_path','journal_sha256'),('mip_path','mip_sha256')]:w.require_hash(s[p],s[h])
 v=w.read(s['status_path']);m=w.read(s['mip_path']);pa=m['physical_pool_audit']
 if Path(v['columns_journal']).resolve()!=Path(s['journal_path']).resolve():raise ValueError('journal path mismatch')
 if m['source_result_sha256']!=s['status_sha256'] or m['source_journal_sha256']!=s['journal_sha256']:raise ValueError('source binding mismatch')
 if m['physical_replay_validated'] is not True or pa['rejected_columns'] or pa['deterministically_repaired']:raise ValueError('unadmitted donor')
 if pa['base_pool_column_count']!=pa['post_augmentation_columns'] or pa['base_pool_ordered_sha256']!=pa['augmented_pool_ordered_sha256']:raise ValueError('donor source was augmented')
 return v,m

def stream_source(db,table,path,trips):
 allowed=set(trips);count=0
 db.execute(f'CREATE TABLE {table}(id INTEGER PRIMARY KEY, key TEXT UNIQUE, cost REAL, rh TEXT, payload TEXT)')
 with Path(path).open() as f:
  for line in f:
   if not line.strip():continue
   r=json.loads(line);ts=r['trips'];cost=float(r['cost'])
   if not ts or len(ts)!=len(set(ts)) or not set(ts)<=allowed or any(not isinstance(t,int) or isinstance(t,bool) for t in ts) or not math.isfinite(cost):raise ValueError('invalid column')
   key=incidence(r);old=db.execute(f'SELECT cost FROM {table} WHERE key=?',(key,)).fetchone()
   payload=line.rstrip('\n');rh=route_hash(r)
   if old is None:db.execute(f'INSERT INTO {table}(key,cost,rh,payload) VALUES(?,?,?,?)',(key,cost,rh,payload))
   elif cost<old[0]-1e-9:db.execute(f'UPDATE {table} SET cost=?,rh=?,payload=? WHERE key=?',(cost,rh,payload,key))
   count+=1
   if count%2000==0:db.commit()
 db.commit();return count

def table_hashes(db,table):
 rows=list(db.execute(f'SELECT key,cost,rh FROM {table} ORDER BY key'))
 # Sorted scientific incidence/cost and recorded physical-route set hashes; raw bytes have their own hash.
 return {'native_unique_columns':len(rows),'incidence_set_sha256':digest([r[0] for r in rows]),'incidence_cost_set_sha256':digest([[r[0],r[1]] for r in rows]),'recorded_route_set_sha256':digest(sorted(r[2] for r in rows))}

def donor_witness(db,own,union,source,m,trips):
 routes=m['selected_routes'];counts=Counter();mapped_cost=0;identical=0;details=[]
 for r in routes:
  key=incidence(r);row=db.execute(f'SELECT cost,rh FROM {own} WHERE key=?',(key,)).fetchone();merged=db.execute(f'SELECT cost,rh FROM {union} WHERE key=?',(key,)).fetchone()
  recorded=r['physical_realization']['recorded_route_sha256']
  if row is None or row[1]!=recorded or not math.isclose(row[0],r['expanded_grid_cost'],rel_tol=1e-10,abs_tol=1e-6):raise ValueError('donor selected route absent/different in own native pool')
  if merged is None or merged[0]>row[0]+1e-9:raise ValueError('donor incidence lost or made more expensive by union')
  if r['master_cost_semantics']!='expanded_grid_cost':raise ValueError('wrong donor cost semantics')
  counts.update(r['trips']);mapped_cost+=merged[0];identical+=merged[1]==recorded
  details.append({'incidence_sha256':digest(key),'source_recorded_route_sha256':recorded,'union_recorded_route_sha256':merged[1],'source_cost':row[0],'union_cost':merged[0],'exact_record_retained':merged[1]==recorded})
 if set(counts)!=set(trips) or len(routes)!=m['buses']:raise ValueError('donor not a full covering witness')
 return {'arm':source['arm'],'source_mip_sha256':source['mip_sha256'],'buses':len(routes),'donor_expanded_grid_objective':sum(r['cost'] for r in routes),'union_incidence_substitution_objective':mapped_cost,'exact_record_retained_count':identical,'same_incidence_cheaper_or_tied_substitution_count':len(routes)-identical,'all_own_selected_records_matched_exactly':True,'all_union_incidences_present_no_higher_cost':True,'overcovered_trip_count':sum(n>1 for n in counts.values()),'route_membership':details,'interpretation':'Independent feasible upper bound from admitted donor columns under covering; no start supplied to new solver. A union-dominating physical route may substitute for a donor route with the same incidence.'}

def construct(spec,out):
 start=time.monotonic();out=Path(out);sources=spec['sources'];assert len(sources)==2
 checked=[validate_source(s) for s in sources];first=identities([x[0] for x in checked]);trips=first['trip_ids']
 for p,h in spec['static_hashes'].items():w.require_hash(p,h)
 w.require_hash(spec['input_path'],spec['input_sha256'])
 journal=Path(str(out)+'.columns.jsonl');database=Path(str(out)+'.sqlite');db=sqlite3.connect(database)
 db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF');db.execute('PRAGMA cache_size=-32768')
 try:
  counts=[stream_source(db,'s'+str(i),s['journal_path'],trips) for i,s in enumerate(sources)]
  summaries=[table_hashes(db,'s'+str(i)) for i in range(2)]
  for s,n in zip(sources,summaries):
   if s['native_pool_columns']!=n['native_unique_columns']:raise ValueError('source native pool count changed')
  db.execute('CREATE TABLE u(id INTEGER PRIMARY KEY,key TEXT UNIQUE,cost REAL,rh TEXT,payload TEXT)')
  replaced=ties=0
  for i in range(2):
   for key,cost,rh,payload in db.execute(f'SELECT key,cost,rh,payload FROM s{i} ORDER BY id'):
    old=db.execute('SELECT cost FROM u WHERE key=?',(key,)).fetchone()
    if old is None:db.execute('INSERT INTO u(key,cost,rh,payload) VALUES(?,?,?,?)',(key,cost,rh,payload))
    elif cost<old[0]-1e-9:db.execute('UPDATE u SET cost=?,rh=?,payload=? WHERE key=?',(cost,rh,payload,key));replaced+=1
    else:ties+=abs(cost-old[0])<=1e-9
   db.commit()
  union_summary=table_hashes(db,'u');overlap=db.execute('SELECT COUNT(*) FROM s0 JOIN s1 USING(key)').fetchone()[0]
  witnesses=[donor_witness(db,'s'+str(i),'u',s,m,trips) for i,(s,(_,m)) in enumerate(zip(sources,checked))]
  with journal.open('x') as f:
   for (payload,) in db.execute('SELECT payload FROM u ORDER BY id'):f.write(payload+'\n')
 finally:db.close()
 for s in sources:
  w.require_hash(s['status_path'],s['status_sha256']);w.require_hash(s['journal_path'],s['journal_sha256']);w.require_hash(s['mip_path'],s['mip_sha256'])
 details={'schema':'evsp-compact-native-union-v1','kind':'pool_construction','optimization_run':False,'full_model_lp_certified':False,'source_order':['core','core512'],'sources':sources,'source_record_counts':counts,'source_pool_sets':summaries,'union_pool_set':union_summary,'shared_incidence_count':overlap,'second_source_cheaper_wins':replaced,'source_cost_ties':ties,'union_journal_sha256':w.sha(journal),'route_identity':'frozenset(trips)','dedup_cost_policy':'Exact frozen-native policy: cheapest cost by >1e-9; first record/source wins ties; raw winning record preserved','donor_witnesses':witnesses,'best_independent_fleet_upper_bound':min(x['buses'] for x in witnesses),'supplied_solver_start':False,'combined_child_cg_wall_s':sum(s['cg_wall_s'] for s in sources),'prior_cg_accounting_scope':'Combined child native wall shown; source_audit records shared parent CG/graph history separately; no same-computation performance claim','constructor_wall_s':time.monotonic()-start,'peak_rss_native':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'constructor_sha256':w.sha(__file__)}
 status={k:first[k] for k in IDENTITY};status.update(schema='evsp-finite-pool-union-mip-input-v3',artifact_kind='finite_pool_union',optimization_run=False,provenance={k:first['provenance'][k] for k in HASHES},columns_journal=str(journal),certified_rc_optimal=False,stop_reason='pool_construction_no_cg',wall_s=0,final={'artificials':0,'iter':0,'pool_columns':union_summary['native_unique_columns']},pool_construction=details)
 status['provenance']['scope']='Route-source model hashes, not a new CG execution or pricing certificate'
 w.save(out,status);w.save(out.parent/'construction.json',details);database.unlink()
 return details
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True);a=p.parse_args();construct(w.read(a.spec),a.out)
