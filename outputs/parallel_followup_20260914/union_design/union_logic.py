"""Disk-backed union of physically admitted, identical baseline source pools."""
from pathlib import Path
import argparse,hashlib,json,math,sqlite3,time,resource

PRODUCER='e091a4dba549510238507ef5e5367abea958bd30'
IDENTITY=('csv','trip_ids','g_kwh','charge_kw','min_soc_frac','soc_step','block_min','time_model','master_sense','prices_csv')
HASHES=('instance_sha256','prices_sha256','reference_sha256','deadhead_sha256','git_commit')
BASELINE={'g_kwh':240,'charge_kw':240,'min_soc_frac':0,'soc_step':2.5,'block_min':5,'time_model':'event','master_sense':'cover','prices_csv':'hourly_prices_flat.csv'}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def checkhash(p,h):
 if not h or sha(p)!=h:raise ValueError('Source hash mismatch: '+str(p))
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def identities(values):
 first=values[0]
 for v in values:
  for key in IDENTITY:
   if key not in first or key not in v or v[key]!=first[key]:raise ValueError('Identity mismatch: '+key)
  for key in HASHES:
   if not first['provenance'].get(key) or v['provenance'].get(key)!=first['provenance'][key]:raise ValueError('Provenance mismatch: '+key)
  if v['provenance']['git_commit']!=PRODUCER:raise ValueError('Unreviewed route producer')
  for key,value in BASELINE.items():
   if v.get(key)!=value:raise ValueError('Nonbaseline source: '+key)
  for key in ['capacity_enforced','shared_station_capacity','charger_counts','terminal_soc_floor','station_power_kw']:
   if v.get(key) not in (None,False,{},[]):raise ValueError('Unsupported source constraint: '+key)
 return first

def merge(journals,trips,database,out):
 allowed=set(trips);db=sqlite3.connect(database)
 db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF');db.execute('PRAGMA cache_size=-32768')
 db.execute('CREATE TABLE cols (id INTEGER PRIMARY KEY, tripkey TEXT UNIQUE, cost REAL, payload TEXT)')
 counts=[]
 try:
  for journal in journals:
   count=0
   with Path(journal).open() as f:
    for line in f:
     if not line.strip():continue
     r=json.loads(line);t=r['trips'];cost=float(r['cost'])
     if not t or any(not isinstance(x,int) or isinstance(x,bool) for x in t) or len(t)!=len(set(t)) or not set(t)<=allowed or not math.isfinite(cost):raise ValueError('Invalid source column')
     key=json.dumps(sorted(t),separators=(',',':'));old=db.execute('SELECT cost FROM cols WHERE tripkey=?',(key,)).fetchone()
     if old is None:db.execute('INSERT INTO cols(tripkey,cost,payload) VALUES(?,?,?)',(key,cost,line.rstrip('\n')))
     elif cost<old[0]-1e-9:db.execute('UPDATE cols SET cost=?,payload=? WHERE tripkey=?',(cost,line.rstrip('\n'),key))
     count+=1
     if count%2000==0:db.commit()
   db.commit();counts.append(count)
  n=db.execute('SELECT COUNT(*) FROM cols').fetchone()[0]
  with Path(out).open('x') as f:
   for (payload,) in db.execute('SELECT payload FROM cols ORDER BY id'):f.write(payload+'\n')
  return n,counts
 finally:db.close()

def construct(spec,out):
 start=time.monotonic();sources=spec['sources'];values=[]
 assert len(sources)==3
 for s in sources:
  for path,key in [('status_path','status_sha256'),('journal_path','journal_sha256'),('mip_evidence_path','mip_evidence_sha256')]:checkhash(s[path],s[key])
  v=read(s['status_path']);mip=read(s['mip_evidence_path']);audit=mip['physical_pool_audit']
  assert Path(v['columns_journal']).resolve()==Path(s['journal_path']).resolve()
  assert mip['source_result_sha256']==s['status_sha256'] and mip['source_journal_sha256']==s['journal_sha256']
  assert audit['rejected_columns']==audit['deterministically_repaired']==0 and mip['physical_replay_validated'] is True
  assert audit['input_hashes']['instance_sha256']==v['provenance']['instance_sha256']==spec['input_sha256']
  values.append(v)
 first=identities(values)
 assert spec['baseline_scope']=={'shared_station_capacity':False,'terminal_soc_floor':False,'bus_coefficient':100000,'charge_start_fee':5,'configuration_bound_by':PRODUCER}
 for p,h in spec['static_hashes'].items():checkhash(p,h)
 out=Path(out);journal=Path(str(out)+'.columns.jsonl');database=Path(str(out)+'.sqlite')
 n,counts=merge([s['journal_path'] for s in sources],first['trip_ids'],database,journal)
 for s in sources:
  checkhash(s['status_path'],s['status_sha256']);checkhash(s['journal_path'],s['journal_sha256'])
 construction=dict(schema='evsp-finite-pool-construction-v1',kind='pool_construction',optimization_run=False,
  source_order=['original','c200','complementary'],sources=sources,source_record_counts=counts,
  union_columns=n,union_journal_sha256=sha(journal),input_sha256=spec['input_sha256'],
  route_identity='frozenset(trips)',cost_policy='Preserve complete raw winner; lowercost by>1e-9 wins; first-source ties retained',
  baseline_scope=spec['baseline_scope'],constructor_sha256=sha(__file__),
  constructor_wall_s=time.monotonic()-start,peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  full_model_lp_certified=False,source_admission_evidence='All18bound sourceMIPs have zero rejected/repaired columns; actual unionMIP replays the pool again before optimization')
 status={k:first[k] for k in IDENTITY}
 status.update(schema='evsp-finite-pool-union-mip-input-v2',artifact_kind='finite_pool_union',optimization_run=False,
  provenance={k:first['provenance'][k] for k in HASHES},columns_journal=str(journal),
  certified_rc_optimal=False,stop_reason='pool_construction_no_cg',wall_s=0,
  final={'artificials':0,'iter':0,'pool_columns':n},pool_construction=construction)
 status['provenance']['scope']='Hashes identify unchanged route-source model, not a new CG execution or pricing certificate'
 save(out,status);save(out.parent/'construction.json',construction)
 database.unlink()
 return construction

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True);a=p.parse_args();construct(read(a.spec),a.out)
