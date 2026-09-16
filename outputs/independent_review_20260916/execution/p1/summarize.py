"""Read-only P1 result collector. Runs on Unicorn; does not retry jobs."""
from pathlib import Path
import argparse,csv,datetime,json,hashlib

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main(root,out):
 b=Path(root);o=Path(out);o.mkdir(parents=True,exist_ok=True);m=read(b/'manifest.json');rows=[]
 def row(cid,c,result,stage):
  base=dict(case_id=cid,item=c['review_item'],findings='/'.join(c['findings']),chain=c['chain'],k=c.get('target_k',c.get('k')),seed=c['seed'],fleet_seconds=c.get('stage1_budget_s',10800),status=stage)
  if result and Path(result).exists():
   r=read(result);t=r.get('two_stage',{});base.update(result_path=str(result),result_sha256=sha(result),buses=r.get('buses'),pool_fleet_bound=r.get('fleet_bound'),fleet_proven_in_pool=r.get('fleet_proven'),target_matched=r.get('buses',999)<=base['k'],physical_route_replay=r.get('physical_replay_validated'),duplicate_cleanup=r.get('duplicate_trip_removal_validated'),ordered_pool_sha256=r.get('physical_pool_audit',{}).get('mip_ordered_pool_sha256'),runtime_s=r.get('runtime_s'),seed_observed=r.get('mip_provenance',{}).get('gurobi_parameters',{}).get('Seed'))
  rows.append(base)
 for cid,c in m['cases'].items():
  p=b/'cases'/cid/'completion.json'
  comp=read(p) if p.exists() else {}
  row(cid,c,comp.get('result_path'),comp.get('status','not_completed'))
 for c in m['reused_cells']:
  p=Path(c['campaign'])/'cases'/c['case']/'completion.json';comp=read(p) if p.exists() else {}
  row(c['case_id'],c,comp.get('result_path'),('reused_'+comp.get('status','not_completed')))
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (o/'results.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 fresh=[r for r in rows if r['item']==7]; matched=sorted({r['chain'] for r in fresh if r.get('target_matched')}); finished=sum('buses' in r for r in fresh)
 report=dict(collected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_sha256=sha(b/'manifest.json'),cells=len(rows),completed=sum('buses' in r for r in rows),F5=dict(status='unresolved' if finished<18 else 'tested',completed_fresh_cells=finished,matched_chains=matched,review_headline_trigger=len(matched)>=4,interpretation='At least one seed matching in at least four of six chains triggers reviewer headline revision. Finite-pool outcomes alone cannot identify causal effects of warm starting; baseline fleet allowance was 1800s, repeated allowance10800s.'),rows=rows)
 (o/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.root,a.out)
