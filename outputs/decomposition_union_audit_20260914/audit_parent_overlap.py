"""Read-only exact service interval overlap witness on the frozen parent."""
import csv,datetime,hashlib,io,json
from collections import defaultdict
from pathlib import Path
P=Path('/home/nc437/ladder-lite/overnight_extension_20260912/code/data/overnight_decomposition_20260912/parent32.csv')
EXPECTED='4367335166098c6c50fb283b1cd3307a72720ea0b70fff4567b085af9a37e66e'
def seconds(value):
 p=value.split(':');assert len(p) in (2,3)
 h,m=map(int,p[:2]);s=int(p[2]) if len(p)==3 else 0
 assert h>=0 and 0<=m<60 and 0<=s<60
 return h*3600+m*60+s
def clock(value):return f'{value//3600:02d}:{value//60%60:02d}:{value%60:02d}'
def audit(path=P):
 raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest();assert digest==EXPECTED
 rows=list(csv.DictReader(io.StringIO(raw.decode())));assert len(rows)==750
 trips={};starts=defaultdict(set);ends=defaultdict(set)
 for row in rows:
  tid=row['Ordered_Trip_ID'];a=seconds(row['Start1']);b=seconds(row['End1']);assert b>a and tid not in trips
  trips[tid]={'stable_trip_id':tid,'local_trip_id':row['count_trip_id'],'start':row['Start1'],'end':row['End1'],'start_seconds':a,'end_seconds':b}
  starts[a].add(tid);ends[b].add(tid)
 events=sorted(starts.keys()|ends.keys());active=set();peak=0;witnesses=[]
 for i,t in enumerate(events):
  active.difference_update(ends[t]);active.update(starts[t])
  if i==len(events)-1:assert not active;continue
  if len(active)>peak:peak=len(active);witnesses=[]
  if len(active)==peak and peak:
   ids=sorted(active,key=int);witnesses.append({'start':clock(t),'end_exclusive':clock(events[i+1]),'start_seconds':t,'end_seconds_exclusive':events[i+1],'stable_trip_ids':ids})
 first=witnesses[0];ts=first['start_seconds']
 direct=sorted([tid for tid,r in trips.items() if r['start_seconds']<=ts<r['end_seconds']],key=int)
 assert direct==first['stable_trip_ids'] and len(direct)==peak
 # Independent O(n²) check on every service start, without event bookkeeping.
 direct_peak=max(sum(r['start_seconds']<=t<r['end_seconds'] for r in trips.values()) for t in starts)
 assert direct_peak==peak
 return dict(schema='evsp-service-overlap-lower-bound-v1',audited_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),input_path=str(path),input_sha256=digest,trip_count=len(rows),included_rows='All750frozeninputrows; no additional filtering or exclusions.',interval_semantics='Actual Start1/End1 service-clock timestamps; half-open[start,end), departures at a common endpoint processed after arrivals finish; no rounding, deadhead or charging padding.',fleet_lower_bound=peak,peak_windows=witnesses,first_peak_trip_witness=[trips[k] for k in direct],independent_direct_check_peak=direct_peak,full_model_pricing_certificate=False,cg_run=False,finite_pool_mip_proof=False,conclusion=('Any physically valid32busfull-inputschedule proves global fleet optimality by matching this independent lowerbound.' if peak==32 else 'A physically valid32busfull-inputschedule would not close this overlap lowerbound.'),proof_scope='Every bus can serve at most one service trip at a time; therefore every feasible integral fleet needs at least the peak simultaneous-trip count. This does not prove weighted cost optimality or certify a pricing problem.')
if __name__=='__main__':print(json.dumps(audit(),indent=2))
