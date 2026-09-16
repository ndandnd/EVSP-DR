"""Read-only conversion of GIRO FDL workbooks to a declared exploratory model."""
from pathlib import Path
import pandas as pd,json,csv,hashlib,random,math
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3];DATA=OUT/'data';DATA.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
raw=ROOT/'data/FDL_VehicleDetails.xlsx';dhd=ROOT/'data/FDL_DHD.xlsm'
x=pd.read_excel(raw,dtype=str).fillna('');d=pd.read_excel(dhd,sheet_name='Deadhead').fillna('')
def token(v):
 s=str(v).strip()
 try:
  f=float(s)
  return str(int(f)) if f.is_integer() else s
 except ValueError:return s
aliases={}
for a,b in [('From1','Refer.'),('To1','Refer.1')]:
 for loc,ref in zip(x[a],x[b]):
  if loc and ref:
   loc,ref=token(loc),token(ref);assert loc not in aliases or aliases[loc]==ref;aliases[loc]=ref
places={token(z) for z in d['Start Place']}|{token(z) for z in d['End Place']}
for p in places:aliases.setdefault(p,p)
locations={token(z) for z in x['From1']}|{token(z) for z in x['To1']};missing=locations-set(aliases);assert not missing,missing
pairs={};omitted=[]
for _,row in d.iterrows():
 a,b=aliases.get(token(row['Start Place']),token(row['Start Place'])),aliases.get(token(row['End Place']),token(row['End Place']))
 if a==b:continue
 durations=[float(row[k]) for k in ['Base Duration','1st Interval Duration','2nd Interval Duration','3rd Interval Duration'] if row[k]!='']
 distances=[float(row[k]) for k in ['Base Distance','1st Interval Distance','2nd Interval Distance','3rd Interval Distance'] if row[k]!='']
 if not durations or not distances:
  omitted.append({'from':a,'to':b,'reason':'missing duration or distance; arc omitted, not imputed'});continue
 assert min(durations)>=0 and min(distances)>=0
 key=tuple(sorted([a,b]));old=pairs.get(key,(0,0));pairs[key]=(max(old[0],max(durations)),max(old[1],max(distances)*2.0))
with (DATA/'Ref_dict.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['Location','Ref']);w.writerows(sorted(aliases.items()))
with (DATA/'par_ref_dhd.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['Start Place','End Place','Base Duration','Energy used']);w.writerows([(*key,*val) for key,val in sorted(pairs.items())])
reg=x[x.Identifier.eq('Regular')].copy();reg['Ordered_Trip_ID']=range(1,len(reg)+1);duties=sorted(reg.VehicleTask.unique());assert len(duties)==61 and len(reg)==1393
random.Random(20260916).shuffle(duties)
levels=[1,2,3,5,8,10,15];cases={}
cols=['Identifier','From1','Start1','End1','To1','Distance1','Usage kWh','count_trip_id','Ordered_Trip_ID']
for k in levels:
 t=reg[reg.VehicleTask.isin(duties[:k])].sort_values('Ordered_Trip_ID').copy();t['count_trip_id']=range(len(t));p=DATA/f'fdl_k{k:02d}.csv';t[cols].to_csv(p,index=False)
 for c in ['Start1','End1']:
  times=t[c].map(lambda s:sum(int(v)*m for v,m in zip(s.split(':'),[60,1])));assert times.min()>=0 and times.max()<=1560
 cases[f'fdl_k{k:02d}']={'k':k,'csv':p.name,'input_sha256':sha(p),'trip_count':len(t),'duties':duties[:k]}
chargers=sorted({token(z) for z in x[x.Identifier.eq('Recharge')].From1});assert 'KEX' in chargers
# Preserve and explicitly extend the constant original tariff through hour25.
original_flat=OUT/'original_hourly_prices_flat.csv'
original_rows=list(csv.DictReader(original_flat.open()))
assert len(original_rows)==25 and {float(r['cost']) for r in original_rows}=={0.09920000000000001}
with (DATA/'hourly_prices_flat.csv').open('w',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['time_block','cost']);w.writerows((h,'0.09920000000000001') for h in range(26))
metadata={'schema':'review-fdl-inputs-v1','finding':'F8','review_item':14,'source_hashes':{str(raw):sha(raw),str(dhd):sha(dhd)},'regular_trips':len(reg),'duty_count':len(duties),'selection':'seed20260916 random ordering, no feasibility/easiness screening','duty_order':duties,'chargers':chargers,'depot':'KEX','deadhead_conversion':'symmetric maximum of BOTH directions and ALL available time-interval durations/distances; energy=2.0kWh/km; no silent missing arcs or shortest-path imputation','energy_scope':'service-trip Usage kWh retained exactly; deadhead2.0kWh/km is declared constant-model assumption, not inferred actualFDLtraction','horizon_minutes':1560,'cases':cases,'derived_hashes':{p.name:sha(p) for p in DATA.glob('*.csv')},'missing_location_aliases':sorted(missing),'omitted_deadhead_rows':omitted,'status':'inputs_prepared_not_results'}
metadata['tariff_extension']={'source_sha256':sha(original_flat),'source_saved_as':'original_hourly_prices_flat.csv','source_hours':'0..24','new_hours':'0..25','price_per_kwh':.0992,'reason':'Explicit complete26hour coverage required by continuous optimizer; extend constant tariff by same price, no shape change','new_sha256':sha(DATA/'hourly_prices_flat.csv')}
(OUT/'inputs.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps({'cases':{c:(s['k'],s['trip_count']) for c,s in cases.items()},'aliases':len(aliases),'deadhead_pairs':len(pairs),'chargers':chargers},indent=2))
