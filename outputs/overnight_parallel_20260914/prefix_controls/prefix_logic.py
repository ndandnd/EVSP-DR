"""Reconstruct an immutable historical pre-insertion iteration pool, not CG."""
from pathlib import Path
import argparse,csv,hashlib,json,math,time,resource
PRODUCER='e091a4dba549510238507ef5e5367abea958bd30'
IDENTITY=('csv','trip_ids','g_kwh','charge_kw','min_soc_frac','soc_step','block_min','time_model','master_sense','prices_csv')
HASHES=('instance_sha256','prices_sha256','reference_sha256','deadhead_sha256','git_commit')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def require(p,h):
 if not h or sha(p)!=h:raise ValueError('Frozen hash mismatch: '+str(p))
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def extract(source,destination,cutoff,allowed):
 """Stream all source bytes for hash/monotonic audit; preserve prefix bytes."""
 h=hashlib.sha256();pool={};total=selected=0;last=-1;origins=set()
 with Path(source).open('rb') as f,Path(destination).open('xb') as out:
  for line in f:
   h.update(line)
   if not line.strip():continue
   r=json.loads(line);i=r.get('found_iter');t=r.get('trips');cost=float(r['cost'])
   if not isinstance(i,int) or isinstance(i,bool) or i<0 or i<last:raise ValueError('Missing/nonmonotonic found_iter')
   last=i;total+=1
   if not isinstance(t,list) or not t or any(not isinstance(x,int) or isinstance(x,bool) for x in t) or len(t)!=len(set(t)) or not set(t)<=allowed or not math.isfinite(cost):raise ValueError('Invalid source route')
   if i>=cutoff:continue
   out.write(line if line.endswith(b'\n') else line+b'\n');selected+=1;origins.add(r.get('origin'))
   key=frozenset(t)
   if key not in pool or cost<pool[key]-1e-9:pool[key]=cost
 return {'source_journal_sha256':h.hexdigest(),'source_records':total,'selected_records':selected,'unique_pool_columns':len(pool),'selected_origins':sorted(origins,key=str),'last_source_found_iter':last}
def construct(spec,out):
 start=time.monotonic();s=spec['source'];out=Path(out)
 for p,h in spec['static_hashes'].items():require(p,h)
 require(s['status_path'],s['status_sha256']);require(s['iters_path'],s['iters_sha256'])
 parent=read(s['status_path']);assert parent['provenance']['git_commit']==PRODUCER and parent['provenance']['instance_sha256']==spec['input_sha256']
 args=parent['provenance']['args'];assert args['initial_pool']=='singletons' and not args.get('inherit_event_pool_from') and not args.get('validated_seed_routes') and not args.get('resume') and not args.get('diversify_rounds')
 for k,v in {'g_kwh':240,'charge_kw':240,'min_soc_frac':0,'soc_step':2.5,'block_min':5,'time_model':'event','master_sense':'cover','prices_csv':'hourly_prices_flat.csv'}.items():assert parent[k]==v
 assert Path(parent['columns_journal']).resolve()==Path(s['journal_path']).resolve()
 with open(s['iters_path']) as f:rows=list(csv.DictReader(f))
 before=[r for r in rows if float(r['elapsed_s'])<14400];row=before[-1];assert row==s['cutoff_row']
 cutoff=int(row['iteration']);journal=Path(str(out)+'.columns.jsonl');audit=extract(s['journal_path'],journal,cutoff,set(parent['trip_ids']))
 assert audit['source_journal_sha256']==s['journal_sha256'];assert audit['unique_pool_columns']==int(row['pool_columns'])
 require(s['status_path'],s['status_sha256']);require(s['iters_path'],s['iters_sha256'])
 construction={'schema':'evsp-retrospective-iteration-prefix-v1','kind':'pool_construction','optimization_run':False,'full_model_lp_certified':False,'source':s,'cutoff_found_iter_exclusive':cutoff,'historical_log_elapsed_s':float(row['elapsed_s']),'prefix_journal_sha256':sha(journal),'input_sha256':spec['input_sha256'],'constructor_sha256':sha(__file__),'constructor_wall_s':time.monotonic()-start,'peak_rss_native':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**audit,'semantics':'Pool before insertion of the batch found at cutoff iteration. Historical iteration CSV is written after pricing and before insertion. Not exact240minute terminalCG, not resumedCG, no inherited finalLP/certificate. Native MIP will physically replay all admitted routes.'}
 status={k:parent[k] for k in IDENTITY};status.update(schema='evsp-retrospective-prefix-mip-input-v1',artifact_kind='retrospective_iteration_prefix',optimization_run=False,provenance={k:parent['provenance'][k] for k in HASHES},columns_journal=str(journal),certified_rc_optimal=False,stop_reason='constructed_historical_iteration_prefix',final={'iter':0,'artificials':None,'pool_columns':audit['unique_pool_columns']},pool_construction=construction)
 status['provenance']['scope']='Identity of unchanged source model; no new CG execution or LPcertificate.'
 save(out,status);save(out.parent/'construction.json',construction);return construction
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True);a=p.parse_args();construct(read(a.spec),a.out)
