"""No-solver service-overlap lower bound with and without mandatory group separation."""
import collections,csv,hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent;ROOT=B.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(Path(p).open()))
def minute(value):
 parts=str(value).split(':');assert len(parts)==2
 h,m=map(int,parts);assert h>=0 and 0<=m<60
 return 60*h+m

def overlap(trips):
 # Ends occur before starts at equal time: conservative half-open intervals.
 events=collections.defaultdict(lambda:{'start':set(),'end':set()})
 for tid,s,e in trips:
  assert e>s,(tid,s,e);events[s]['start'].add(tid);events[e]['end'].add(tid)
 active=set();best=0;witness=[];at=None
 for t,event in sorted(events.items()):
  active.difference_update(event['end']);active.update(event['start'])
  if len(active)>best:best=len(active);witness=sorted(active);at=t
 return best,at,witness

def main():
 source=read(B.parent/'audited_chain_results.csv');master=ROOT/'outputs/chain_extension_20260913/inputs/sources/Par_VehicleDetails_Updated.csv';lookup={int(r['Ordered_Trip_ID']):r['VehicleTask'] for r in read(master) if r['Identifier']=='Regular'}
 out=[]
 for r in source:
  p=ROOT/'outputs'/r['campaign']/'inputs'/(r['case_id']+'.csv');assert sha(p)==r['input_sha256'];groups={'18E1':[],'18E2':[]}
  for trip in read(p):
   tid=int(trip['Ordered_Trip_ID']);duty=lookup[tid];assert duty.startswith(('133','134'));groups['18E1' if duty.startswith('134') else '18E2'].append((tid,minute(trip['Start1']),minute(trip['End1'])))
  row={k:r[k] for k in ['case_id','chain','target_buses','fractional_route_weight','input_sha256']};row['input_path']=str(p);row['cohort']='k_minus_1' if abs(float(r['fractional_route_weight'])-(int(r['target_buses'])-1))<1e-6 else 'k'
  for g,ts in groups.items():
   n,t,w=overlap(ts);row[g+'_overlap_lower_bound']=n;row[g+'_witness_minute']=t;row[g+'_witness_ordered_trip_ids']=json.dumps(w)
  row['separated_groups_lp_fleet_lower_bound']=sum(row[g+'_overlap_lower_bound'] for g in groups);row['all_groups_service_overlap_lower_bound']=overlap(groups['18E1']+groups['18E2'])[0];row['separated_bound_equals_k']=row['separated_groups_lp_fleet_lower_bound']==int(r['target_buses']);row['separated_bound_excludes_k_minus_1']=row['separated_groups_lp_fleet_lower_bound']>int(r['target_buses'])-1;out.append(row)
 with (B/'interval_bound_check.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
 summary={'source_table_sha256':sha(B.parent/'audited_chain_results.csv'),'master_sha256':sha(master),'script_sha256':sha(__file__),'result_sha256':sha(B/'interval_bound_check.csv'),'solver_calls':0,'group_kminus1':[r for r in out if r['cohort']=='k_minus_1'],'separated_bound_equals_k_total':sum(r['separated_bound_equals_k'] for r in out),'all_cases':len(out),'proof':'For group g choose all service trips overlapping its witness time. Every feasible route of that group covers at most one witness trip, so covering constraints imply sum(lambda on group g)>=witness count. Disjoint group-specific route sets allow summing across group-specific witness times. Nonnegative lambdas therefore have total mass at least n18E1+n18E2. This is an exact combinatorial bound for the recorded integer-minute intervals; no pricing certificate is required.','time_semantics':'Production build_problem maps Start1/End1 to start_min/end_min using _total_minutes. Half-open intervals [start,end); ignore all deadheads and charging. Hours beyond24 retained, no modulo24.'}
 (B/'interval_bound_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'all_equal_k':summary['separated_bound_equals_k_total'],'kminus1':[(r['case_id'],r['18E1_overlap_lower_bound'],r['18E2_overlap_lower_bound'],r['separated_groups_lp_fleet_lower_bound'],r['all_groups_service_overlap_lower_bound']) for r in summary['group_kminus1']]}))
if __name__=='__main__':main()
