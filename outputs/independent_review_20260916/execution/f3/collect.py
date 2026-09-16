from pathlib import Path
import json,csv,hashlib,sys
rows=list(csv.DictReader(sys.stdin))
out=[]
for r in rows:
 p=Path(r['cg_path']); b=p.read_bytes();j=json.loads(b)
 root=Path('/home/nc437/ladder-lite')/r['campaign'];inp=root/'code/data'/j['csv']; ib=inp.read_bytes(); trips=list(csv.DictReader(ib.decode().splitlines()))
 def minutes(x):
  h,m=map(float,x.split(':'));return h*60+m
 intervals=sorted((minutes(t['End1']),minutes(t['Start1'])) for t in trips)
 assert all(a>b for a,b in intervals)
 end=-1;count=0
 for e,s in intervals:
  if s>=end:count+=1;end=e
 fl=j.get('final_lp') or {};last=j.get('final') or {};dual=fl.get('trip_duals') or {}
 prices=root/'code/data'/j['prices_csv'];pb=prices.read_bytes(); pp=list(csv.DictReader(pb.decode().splitlines()));maxprice=max(float(x['cost']) for x in pp)
 out.append(dict(case=r['case_id'],chain=int(r['chain']),k=int(r['target_buses']),n=len(trips),cg_path=str(p),cg_sha256=hashlib.sha256(b).hexdigest(),input_path=str(inp),input_sha256=hashlib.sha256(ib).hexdigest(),prices_path=str(prices),prices_sha256=hashlib.sha256(pb).hexdigest(),max_price=maxprice,max_nonoverlapping_trips=count,last_pricing_iteration=last,certified=j['certified_rc_optimal'],final_lp_scalars={k:v for k,v in fl.items() if not isinstance(v,(dict,list))},dual_sum=sum(dual.values()),dual_min=min(dual.values(),default=0),dual_count=len(dual),provenance=j['provenance'],physics={k:j.get(k) for k in ('g_kwh','charge_kw','min_soc_frac','master_sense','soc_step','block_min','time_model')},network_metrics=j.get('network_metrics')))
print(json.dumps(out,indent=2))
