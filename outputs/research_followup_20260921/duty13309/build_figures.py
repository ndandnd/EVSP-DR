#!/usr/bin/env python3
"""Reproduce whole-day spatial overviews and itinerary from frozen saved witnesses; no solving."""
from pathlib import Path
import csv,json,hashlib,subprocess,collections,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
P=Path(__file__).resolve().parent;ROOT=P.parents[2];OLD=ROOT/'outputs/week_20260921/complex_route_graphs';INP=ROOT/'outputs/week_20260921/cleanup_physics/inputs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:list(csv.DictReader(p.open()))
snapshot=json.loads((P/'sources/campaign_snapshot.json').read_text());original=next(s for s in json.loads((OLD/'schedules.json').read_text())['schedules'] if s['duty_id']=='13309');target=set(original['source_trip_ids'])
ref={r['Location']:r['Ref'] for r in read(INP/'Ref_dict.csv')};dhd={}
for r in read(INP/'par_ref_dhd.csv'):
 a,b=r['Start Place'],r['End Place'];dhd[a,b]=dhd[b,a]=(float(r['Base Duration']),float(r['Energy used']))
area=lambda p:ref.get(p,p)
def link(a,b):return (0.,0.) if area(a)==area(b) else dhd[area(a),area(b)]
def minute(t):
 a,b=t.split(':');return int(a)*60+float(b)
def clock(t):
 x=round(t*60);return f'{x//3600:02d}:{x%3600//60:02d}'+(f':{x%60:02d}' if x%60 else '')
def outcsv(name,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (P/name).open('w') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
rank=[]
master={r['Ordered_Trip_ID']:r for r in read(ROOT/'data/Par_VehicleDetails_Updated.csv') if r['Identifier']=='Regular'}
for key,a in snapshot['arms'].items():
 result=a['files']['result.json']['content'];case=key.rsplit('_',1)[0];rows=snapshot['inputs'][case]['rows'];assert sha(INP/'par_ref_dhd.csv')==result['physical_pool_audit']['input_hashes']['deadhead_sha256'];assert sha(INP/'Ref_dict.csv')==result['physical_pool_audit']['input_hashes']['reference_sha256'];assert snapshot['inputs'][case]['sha256']==result['physical_pool_audit']['input_hashes']['instance_sha256']
 for r in rows:
  m=master[r['Ordered_Trip_ID']]
  if r['Ordered_Trip_ID'] in target:
   for k in ['From1','To1','Start1','End1','Usage kWh']:assert r[k]==m[k],(r,k,m[k])
 for i,r in enumerate(result['selected_routes']):
  ids=[rows[int(j)]['Ordered_Trip_ID'] for j in r['trips']];common=target&set(ids)
  rank.append(dict(case=key,route_index_zero_based=i,trip_count=len(ids),shared=len(common),target_denominator=22,coverage=len(common)/22,jaccard=len(common)/len(target|set(ids)),union_denominator=len(target|set(ids)),source_trip_ids=ids,shared_trip_ids=sorted(common,key=int),charge_starts=r['charges_started'],fleet_size=result['buses'],fleet_overcovered_trips=result['overcovered_trips'],input_hash=snapshot['inputs'][case]['sha256']))
rank.sort(key=lambda r:(-r['jaccard'],-r['coverage'],r['case'],r['route_index_zero_based']));outcsv('overlap_ranking.csv',rank)
outcsv('search_coverage.csv',[dict(case=k,trip_count=v['row_count'],target_overlap=len(v['overlap']),sha256=v['sha256'],remote_source=v['path']) for k,v in snapshot['inputs'].items()])
def ev(kind,start,end,a,b,**kw):return dict(kind=kind,start_min=start,end_min=end,start_time=clock(start),end_time=clock(end),from_code=a,to_code=b,from_ref=area(a),to_ref=area(b),**kw)
def modeled(key,index):
 result=snapshot['arms'][key]['files']['result.json']['content'];route=result['selected_routes'][index];rows=snapshot['inputs'][key.rsplit('_',1)[0]]['rows'];anchors=[];ci=0
 for node in route['route_nodes'][1:-1]:
  if isinstance(node,int):
   r=rows[node];anchors.append(ev('service',minute(r['Start1']),minute(r['End1']),r['From1'],r['To1'],source_trip_id=r['Ordered_Trip_ID'],energy_consumed_kwh=float(r['Usage kWh'])))
  else:
   z=route['charging_stops'];assert node==z['stations'][ci];code=node.rsplit('_',1)[0];anchors.append(ev('charge',z['cst'][ci],z['cet'][ci],code,code,energy_charged_kwh=z['kwh'][ci]));ci+=1
 assert ci==route['charges_started'];events=[]
 first=anchors[0];dur,en=link('PARX',first['from_code']);events.append(ev('deadhead',first['start_min']-dur,first['start_min'],'PARX',first['from_code'],energy_consumed_kwh=en,timing_basis='latest feasible departure for first anchor'))
 for a in anchors:
  b=events[-1];dur,en=link(b['to_code'],a['from_code']);start=b['end_min'];assert start+dur<=a['start_min']+1e-7,(key,index,b,a,dur)
  if b['to_code']!=a['from_code'] or dur:events.append(ev('deadhead',start,start+dur,b['to_code'],a['from_code'],energy_consumed_kwh=en,timing_basis='earliest departure after previous fixed anchor; reconstructed'))
  if start+dur<a['start_min']-1e-7:events.append(ev('wait',start+dur,a['start_min'],a['from_code'],a['from_code'],timing_basis='residual feasible gap'))
  events.append(a)
 last=events[-1];dur,en=link(last['to_code'],'PARX');events.append(ev('deadhead',last['end_min'],last['end_min']+dur,last['to_code'],'PARX',energy_consumed_kwh=en,timing_basis='earliest feasible return after final anchor'))
 for a,b in zip(events,events[1:]):assert a['to_code']==b['from_code'] and abs(a['end_min']-b['start_min'])<1e-7
 rr=next(r for r in rank if r['case']==key and r['route_index_zero_based']==index)
 return dict(schedule_id=key,title=f"Start fee {key[-1]} · C{key[1]} k{int(key[4:6])} · route {index+1}",events=events,overlap=rr,physics=result['physics'],source_result=snapshot['arms'][key]['files']['result.json']['path'],source_result_sha256=snapshot['arms'][key]['files']['result.json']['sha256'],validation_scope=result['physical_replay_scope'],physical_replay_validated=result['physical_replay_validated'],shared_capacity_validated=result['cross_route_charger_capacity_validated'],fleet_proven=result['fleet_proven'],pricing_scope=result['pricing_certificate_scope'],original_route_payload=route)
schedules=[dict(schedule_id='original_13309',title='Recorded GIRO duty 13309',events=original['events'],overlap=dict(trip_count=22,shared=22,coverage=1,jaccard=1,fleet_size=None),physics=dict(g_kwh=239.01,charging='recorded varying charging; 18E2',min_soc_frac=None))]+[modeled(k,i) for k,i in [('w6_k05_fee0',3),('w6_k05_fee5',2),('w6_k15_fee5',1)]]
for s in schedules:
 counts=collections.Counter()
 for e in s['events']:
  prefix={'service':'L','charge':'C','deadhead':'M','wait':'W','prep':'P','same_reference_gap':'G'}[e['kind']];counts[prefix]+=1;e['label']=prefix+str(counts[prefix])
 s['metrics']=dict(service_trips=counts['L'],charge_starts=counts['C'],charged_kwh=sum(e.get('energy_charged_kwh',0) for e in s['events']),start=s['events'][0]['start_time'],end=s['events'][-1]['end_time'])
outcsv('events.csv',[dict(schedule_id=s['schedule_id'],**e) for s in schedules for e in s['events']]);outcsv('comparison.csv',[{'schedule':s['schedule_id'],**s['overlap'],**s['metrics'],'physics':s['physics']} for s in schedules]);(P/'schedules.json').write_text(json.dumps(schedules,indent=2)+'\n')
# A stable geographic-proxy layout, with depot and unknown13722 explicitly schematic.
POS={'3127':(0,0),'7581':(1.76,2.84),'13215':(7.82,4.01),'13410':(11.93,4.83),'13801':(10.7,-.4),'13722':(14,1.7)}
NAMES={'3127':'Heden','7581':'Gamlestads Torg','13215':'Partille centrum','13410':'Jons väg','13801':'PARX depot','13722':'13722\n(unlocated)'}
BLUE='#286493';GREY='#8a98a3';ORANGE='#ca7024';DARK='#253b4a'
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,'font.size':10})
outcsv('display_layout.csv',[dict(area=k,name=NAMES[k],x=v[0],y=v[1],basis='schematic' if k in ['13722','13801'] else 'approximate geographic proxy') for k,v in POS.items()])
def compact(v):return ', '.join(v)
def draw(s,ax):
 ax.set_aspect('equal');ax.axis('off');ax.set_xlim(-1.3,15.2);ax.set_ylim(-2.0,6.3)
 groups=collections.defaultdict(list)
 for e in s['events']:
  if e['from_ref']!=e['to_ref']:groups[e['kind'],e['from_ref'],e['to_ref']].append(e)
 used={e[k] for e in s['events'] for k in ['from_ref','to_ref']};charging={e['from_ref'] for e in s['events'] if e['kind']=='charge'}
 for (kind,a,b),es in sorted(groups.items()):
  x,y=POS[a],POS[b];rad=.16 if kind=='service' else -.14
  if kind=='deadhead' and '13801' in [a,b]:rad=.22
  if kind=='deadhead' and set([a,b])=={'3127','13215'}:rad=-.30
  if kind=='deadhead' and set([a,b])=={'13410','13215'}:rad=-.60
  if kind=='service' and set([a,b])=={'3127','13410'}:rad=.37
  col=BLUE if kind=='service' else GREY
  arr=FancyArrowPatch(x,y,arrowstyle='-|>',mutation_scale=14,lw=1.7 if kind=='service' else 1.15,linestyle='-' if kind=='service' else '--',color=col,connectionstyle=f'arc3,rad={rad}',shrinkA=14,shrinkB=14,zorder=2);ax.add_patch(arr)
  lx=(x[0]+y[0])/2+rad*(y[1]-x[1])/2;ly=(x[1]+y[1])/2-rad*(y[0]-x[0])/2
  labs=[e['label'] for e in es];lab=compact(labs)
  if len(labs)>5:lab=compact(labs[:math.ceil(len(labs)/2)])+'\n'+compact(labs[math.ceil(len(labs)/2):])
  ax.text(lx,ly,lab,ha='center',va='center',fontsize=8.5,color=col,bbox=dict(fc='white',ec='none',pad=1.5),zorder=5)
 offsets={'3127':(-.1,-.7),'7581':(-.4,.58),'13215':(-.7,.57),'13410':(.5,.59),'13801':(.1,-.67),'13722':(.1,-.8)}
 for a in used:
  x,y=POS[a]
  if a in charging:ax.scatter(x,y,s=510,facecolors='none',edgecolors=ORANGE,lw=2.4,zorder=6)
  ax.scatter(x,y,s=230,c='white',edgecolors=DARK,lw=1.5,marker='D' if a=='13801' else 'o',zorder=7)
  dx,dy=offsets[a];ax.text(x+dx,y+dy,NAMES[a],ha='center',va='center',fontsize=10,fontweight='bold',color=DARK,zorder=9,bbox=dict(fc='white',ec='none',pad=1))
def figure(s):
 fig=plt.figure(figsize=(14,8.6));ax=fig.add_axes([.025,.32,.95,.56]);draw(s,ax)
 fig.text(.045,.95,s['title']+' · whole day',fontsize=19,fontweight='bold',color=DARK)
 o=s['overlap'];m=s['metrics'];qual='Recorded 18E2 schedule · 239.01 kWh battery' if s['schedule_id'].startswith('original') else f"Saved historical model · 240 kWh / 240 kW · zero reserve · {o['shared']}/22 original trips shared"
 fig.text(.045,.905,f"{m['start']}–{m['end']}  |  {m['service_trips']} passenger trips  |  {m['charge_starts']} charging starts\n{qual}",fontsize=11,color=DARK,linespacing=1.55)
 charges=[e for e in s['events'] if e['kind']=='charge']
 fig.text(.045,.31,'Charging connections',fontsize=12,fontweight='bold',color=ORANGE)
 for i,e in enumerate(charges):
  col=i//4;row=i%4
  fig.text(.045+col*.48,.272-row*.037,f"{e['label']}  {NAMES[e['from_ref']].splitlines()[0]} ({e['from_code']})  {e['start_time']}–{e['end_time']}  +{e.get('energy_charged_kwh',0):.1f} kWh",fontsize=10.4,color=ORANGE)
 fig.text(.045,.085,'Blue arrows: passenger legs L1, L2, … in time order. Dashed arrows: empty moves M1, M2, …\nRepeated directed edges are grouped; the itinerary retains every clock, platform, wait and charging connection.',fontsize=9.5,color=DARK,linespacing=1.5)
 fig.text(.045,.031,'Approximate place layout, not road geometry. PARX displaced; 13722 unlocated when shown. Orange rings mark used charging sites.',fontsize=8.6,color=GREY)
 for ext in ['png','pdf','svg']:fig.savefig(P/(s['schedule_id']+'_full_day.'+ext),dpi=180,facecolor='white')
 plt.close(fig)
for s in schedules:figure(s)
# One full day per panel; three panels compare schedules, not morning/afternoon halves.
fig,axes=plt.subplots(3,1,figsize=(13,20))
for ax,s in zip(axes,schedules[:3]):
 draw(s,ax);m=s['metrics'];o=s['overlap'];ax.set_title(s['title']+f"  |  {m['service_trips']} trips · {o['shared']}/22 shared · {m['charge_starts']} starts",loc='left',fontsize=14,fontweight='bold',pad=8)
fig.suptitle('Duty 13309 and saved C6 k5 algorithm routes',fontsize=19,fontweight='bold',y=.985)
fig.text(.06,.953,'Whole-day views · fee0 and fee5 come from the same 79-trip input; each saved fleet has five routes.\nOriginal: recorded 239.01 kWh 18E2. Algorithm: historical 240 kWh / 240 kW, zero reserve, no shared charger limits.',fontsize=10.5,linespacing=1.5)
fig.subplots_adjust(top=.92,bottom=.045,hspace=.12);fig.text(.06,.02,'L: chronological passenger leg. M: empty move. Orange rings: used chargers. PARX displaced; 13722 schematic and unlocated.',fontsize=9)
for ext in ['png','pdf','svg']:fig.savefig(P/('comparison_same_cohort.'+ext),dpi=150,facecolor='white')
plt.close(fig)
# Native text PDF gives full chronological details without overloading spatial edges.
with PdfPages(P/'complete_itineraries.pdf') as pdf:
 for s in schedules:
  es=s['events']
  for start in range(0,len(es),32):
   chunk=es[start:start+32];fig,ax=plt.subplots(figsize=(12,10));ax.axis('off');fig.text(.05,.955,s['title']+' · complete itinerary',fontsize=15,fontweight='bold');fig.text(.05,.928,'Model empty-move clocks are feasible reconstructions between fixed trip/charge anchors; recorded GIRO clocks are copied.',fontsize=8.7)
   cell=[]
   for e in chunk:
    detail=(('Trip '+str(e.get('source_trip_id'))) if e['kind']=='service' else (f"+{e.get('energy_charged_kwh',0):.3f} kWh" if e['kind']=='charge' else ''))
    cell.append([e['label'],e['kind'],e['start_time'],e['end_time'],e['from_code']+' → '+e['to_code'],detail])
   tb=ax.table(cellText=cell,colLabels=['Order','Activity','Start','End','Exact source locations','Detail'],colWidths=[.07,.17,.11,.11,.32,.22],loc='upper center',cellLoc='left',bbox=[0,.04,1,.94]);tb.auto_set_font_size(False);tb.set_fontsize(9)
   for (r,c),cellobj in tb.get_celld().items():
    cellobj.set_edgecolor('#d7dfe5');cellobj.set_linewidth(.35)
    if r==0:cellobj.set_facecolor('#e8eff3');cellobj.set_text_props(weight='bold')
   fig.subplots_adjust(top=.9,bottom=.05,left=.05,right=.95);pdf.savefig(fig);plt.close(fig)
manifest=dict(execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),new_optimization=False,resource_requests='local extraction/Matplotlib only; read-only SSH snapshot; no cluster job',dependencies=[],source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [P/'sources/campaign_snapshot.json',OLD/'schedules.json',INP/'Ref_dict.csv',INP/'par_ref_dhd.csv',ROOT/'data/Par_VehicleDetails_Updated.csv',ROOT/'outputs/zero_charge_start_fee_20260913/manifest.json']},source_campaign_commit='06b5cb86d6c24df0ec0a5ca7189fa9552f527dd0',source_campaign_master_sense='cover',source_campaign_initialization='same-k inherited sequences reoptimized under destination fee, plus singletons',source_objective='100000*fleet + flat-tariff electricity + fee*starts; two-stage finite pool MIP',checks={'all_12_arm_input_hashes_match':True,'all_22_target_rows_exact_master_identity':True,'modeled_chronology_continuous':True,'constant_static_deadhead_reference_hash_verified':True,'no_new_physical_certificate':True},ranked_routes=len(rank))
manifest['outputs']={p.name:sha(p) for p in P.iterdir() if p.is_file() and p.name!='figure_manifest.json'};(P/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps([{'schedule':s['schedule_id'],'metrics':s['metrics'],'overlap':s['overlap']['shared']} for s in schedules],indent=2))
