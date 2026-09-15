from pathlib import Path
import hashlib,json,math,sqlite3,sys,time,resource
import worker as w
import native_union as n
B=Path(__file__).resolve().parent
def key(ts):return json.dumps(sorted(ts),separators=(',',':'))
def load_db(s,p):
 for a,b in [('status_path','status_sha256'),('journal_path','journal_sha256'),('mip_evidence_path','mip_evidence_sha256')]:w.require_hash(s[a],s[b])
 v=w.read(s['status_path']);m=w.read(s['mip_evidence_path']);assert m['source_result_sha256']==s['status_sha256'] and m['source_journal_sha256']==s['journal_sha256'];assert m['physical_pool_audit']['rejected_columns']==m['physical_pool_audit']['deterministically_repaired']==0
 journal=p.with_suffix('.canonical.jsonl');n.merge([s['journal_path']],v['trip_ids'],p,journal)
 return sqlite3.connect(p),v,journal
def positive(v,db):
 lp=v['final_lp'];assert lp['artificial_total']==0 and v['certified_rc_optimal'];cover={t:0. for t in v['trip_ids']};weights={};objective=0.;rw=0.
 for r in lp['positive_routes']:
  value=float(r['value']);assert math.isfinite(value) and value>=0
  if value==0:continue
  k=key(r['trips']);assert k not in weights;row=db.execute('SELECT cost,payload FROM cols WHERE tripkey=?',(k,)).fetchone();assert row
  raw=json.loads(row[1]);assert raw['trips']==r['trips'] and abs(float(row[0])-float(r['cost']))<=1e-6
  weights[k]=value;objective+=value*row[0];rw+=value
  for t in r['trips']:cover[t]+=value
 assert min(cover.values())>=1-1e-6 and abs(objective-lp['objective'])<=2e-5 and abs(rw-lp['route_weight'])<=1e-6
 return weights,dict(positive_columns=len(weights),minimum_trip_coverage=min(cover.values()),reconstructed_objective=objective,recorded_objective=lp['objective'],objective_difference=objective-lp['objective'],route_weight=rw,artificial_total=0,source_final_lp_sha256=hashlib.sha256(json.dumps(lp,sort_keys=True,separators=(',',':')).encode()).hexdigest())
def emit(cid,first,sources,base_journal,add_payloads,audit):
 p=B/'pools'/cid;p.mkdir(exist_ok=False);j=p/'pool.columns.jsonl';count=0
 with j.open('x') as f:
  if base_journal:
   with open(base_journal) as inp:
    for line in inp:f.write(line);count+=1
  for payload in add_payloads:f.write(payload+'\n');count+=1
 construction=dict(schema='evsp-lp-support-pool-construction-v1',kind='pool_construction',optimization_run=False,certified_rc_optimal=False,sources=sources,selection=audit,union_columns=count,union_journal_sha256=w.sha(j),input_sha256=first['provenance']['instance_sha256'],source_order='recipient native pool first, then novel donor incidences; donor-only support has source journal order',cost_policy='Unchanged complete cheapest native incidence records; no synthesized route/cost',constructor_sha256=w.sha(__file__))
 status={k:first[k] for k in n.IDENTITY};status.update(schema='evsp-finite-pool-union-mip-input-v2',artifact_kind='finite_pool_union',optimization_run=False,provenance={k:first['provenance'][k] for k in n.HASHES},columns_journal=str(j),certified_rc_optimal=False,stop_reason='pool_construction_no_cg',wall_s=0,final={'artificials':0,'iter':0,'pool_columns':count},pool_construction=construction)
 w.save(p/'pool.json',status);w.save(p/'construction.json',construction)
 return dict(case_id=cid,status_path=str(p/'pool.json'),status_sha256=w.sha(p/'pool.json'),journal_path=str(j),journal_sha256=w.sha(j),construction_path=str(p/'construction.json'),construction_sha256=w.sha(p/'construction.json'),selection=audit,pool_columns=count)
def main(pair):
 start=time.monotonic();ix=w.read(B/'source_index.json')
 for f,h in ix['tooling_sha256'].items():w.require_hash(B/f,h)
 g=ix['groups'][pair];scratch=B/'pools'/('_build_'+pair);scratch.mkdir(exist_ok=False);d,v,dj=load_db(g['donor'],scratch/'donor.sqlite');weights,lp=positive(v,d);outputs=[]
 n.identities([v]);payloads=[payload for k,payload in d.execute('SELECT tripkey,payload FROM cols ORDER BY id') if k in weights]
 outputs.append(emit(pair+'_support_only',v,[g['donor']],None,payloads,dict(treatment='donor_final_positive_support_only',lp_support=lp,added_columns=len(payloads))))
 for r in g['recipients']:
  db,old,oj=load_db(r,scratch/(r['case_id']+'.sqlite'));n.identities([v,old]);oldkeys={k for k, in db.execute('SELECT tripkey FROM cols')}
  positivekeys=[k for k in weights if k not in oldkeys];N=len(positivekeys);assert N>0,'No novel positive support for '+r['case_id']
  zero=[k for k, in d.execute('SELECT tripkey FROM cols ORDER BY id') if k not in weights and k not in oldkeys];assert len(zero)>=N,'Insufficient matched LP-zero donor incidences'
  zero=sorted(zero,key=lambda k:(hashlib.sha256(k.encode()).hexdigest(),k))[:N]
  for arm,ks in [('positive_added',positivekeys),('matched_zero_added',zero)]:
   selected=set(ks);payloads=[payload for k,payload in d.execute('SELECT tripkey,payload FROM cols ORDER BY id') if k in selected];assert len(payloads)==N
   audit=dict(treatment=arm,added_columns=N,novel_positive_count=N,novel_zero_available=sum(1 for k, in d.execute('SELECT tripkey FROM cols') if k not in weights and k not in oldkeys),selected_incidence_keys=ks,selector='SHA256 canonical tripset JSON ascending for matched-zero; positive uses frozen final LP value>0; emitted donor source order',lp_support=lp,recipient_proved_fleet=r['buses'],target_k=g['target_k'])
   outputs.append(emit(r['case_id']+'_'+arm,old,[r,g['donor']],oj,payloads,audit))
  db.close()
 d.close();w.save(B/'pools'/(pair+'_prepared.json'),dict(pair_id=pair,source_index_sha256=w.sha(B/'source_index.json'),outputs=outputs,wall_s=time.monotonic()-start,peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));print(json.dumps({'pair':pair,'pools':len(outputs),'added':[x['selection']['added_columns'] for x in outputs]}))
if __name__=='__main__':main(sys.argv[1])
