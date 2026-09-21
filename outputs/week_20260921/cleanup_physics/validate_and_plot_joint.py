from pathlib import Path
import json,csv,hashlib,sys,math,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import matched_fixed_windows as f
from giro_partille_physics import PARTILLE_PROFILES,charge_soc_after_minutes
pr=PARTILLE_PROFILES['18E1'];idle=pr.idle_kw/60
I=f.ROOT/'outputs/meeting_20260917/route_explainer/inputs';origdoc=json.loads((I/'original.json').read_text());ref=next(r for r in origdoc['routes'] if r['duty_id']=='13414');baseline=next(r for r in json.loads((P/'baseline.json').read_text())['routes'] if r['duty']=='13414')
TR={int(r['count_trip_id']):dict(start=f.minutes(r['Start1']),end=f.minutes(r['End1']),energy=float(r['Usage kWh']),source_id=r['Ordered_Trip_ID']) for r in csv.DictReader((I/'k05.csv').open())}
replay=json.loads((P/'saved_sequence_replay.json').read_text());all_validation={};panels=[dict(label='Recorded GIRO',trips=ref['trips'],**baseline)];summary=[]
for arm in ['saved_joint_fee0','saved_joint_fee5']:
 s=json.loads((P/f'{arm}.json').read_text());routeval=[];covered=collections.Counter();events={}
 for r,source in zip(s['routes'],replay[arm]['routes']):
  soc=pr.usable_capacity_kwh-source['actions'][0]['deadhead_kwh'];minimum=soc;fail=[];ci=0
  for a in source['actions'][1:]:
   t=TR[a['from_trip']];soc-=t['energy'];minimum=min(minimum,soc)
   if a['kind']=='direct':soc-=a['deadhead_kwh']+a['idle_kwh']
   else:
    arr=a['arrival_min'];end=a['latest_departure_min'];soc-=a['inbound_kwh'];minimum=min(minimum,soc)
    matching=[c for c in r['charges'] if c['station']==a['station'] and c['start']>=arr-1e-5 and c['end']<=end+1e-5]
    assert len(matching)<=1
    if matching:
     c=matching[0];soc-=(c['start']-arr)*idle;minimum=min(minimum,soc);new=charge_soc_after_minutes(pr,c['station'],soc,c['end']-c['start']);assert abs(new-soc-c['kwh'])<1e-4;assert c['end']-c['start']>=3-1e-5;soc=new-idle*(end-c['end']);ci+=1
    else:soc-=idle*(end-arr)
    soc-=a['outbound_kwh']
   minimum=min(minimum,soc)
  assert abs(soc-r['terminal_kwh'])<1e-4
  assert soc>=r['target_kwh']-1e-5
  assert minimum>=pr.reserve_kwh-1e-5
  assert ci==len(r['charges'])
  covered.update(r['trips']);routeval.append(dict(source_index=r['source_index'],valid=True,min_soc_kwh=minimum,terminal_kwh=soc,target_kwh=r['target_kwh']))
  for c in r['charges']:events.setdefault(c['station'],[]).extend([(c['start'],1),(c['end'],-1)])
 assert covered==collections.Counter({t:1 for t in TR})
 peaks={}
 for st,ev in events.items():
  cur=peak=0
  for t,v in sorted(ev,key=lambda z:(round(z[0],5),z[1])):cur+=v;peak=max(peak,cur)
  peaks[st]=peak
  assert st=='PARX' or peak<=1
 all_validation[arm]=dict(valid_for_declared_scope=True,exact_once_trips=len(covered),fleet=len(s['routes']),routes=routeval,charger_peaks=peaks,not_validated=['time-dependent DHD','platform/departure blocking','FIFO at4808','crew constraints'],source_hash=hashlib.sha256((P/f'{arm}.json').read_bytes()).hexdigest())
 selected=max(s['routes'],key=lambda r:len(set(r['trips'])&set(ref['trips'])))
 panels.append(dict(label='Our saved CG trips · fee 0' if s['fee']==0 else 'Our saved pool trips · fee 5',**selected))
 summary.append(dict(arm=arm,fee=s['fee'],electricity_cost=sum(r['electricity_cost'] for r in s['routes']),starts=sum(r['starts'] for r in s['routes']),objective=s['objective'],bound=s['bound'],gap=(s['objective']-s['bound'])/s['objective'],status=s['status'],seconds=s['wall_seconds'],selected_source_index=selected['source_index']))
(P/'joint_validation.json').write_text(json.dumps(all_validation,indent=2));(P/'joint_figure_metrics.json').write_text(json.dumps(summary,indent=2))
ordered=sorted({t for r in panels for t in r['trips']},key=lambda t:TR[t]['start']);pos={t:i for i,t in enumerate(ordered)}
fig,axs=plt.subplots(2,3,figsize=(15,8),gridspec_kw={'height_ratios':[3,1]},layout='constrained');export=[]
for col,r in enumerate(panels):
 ax=axs[0,col];seq=r['trips']
 for t in seq:
  q=TR[t];ax.plot([q['start'],q['end']],[pos[t]]*2,color='#126d9d',lw=5,solid_capstyle='butt');export.append(dict(arm=r['label'],event='service',start_min=q['start'],end_min=q['end'],source_trip_id=q['source_id'],station='',kwh=''))
 for a,b in zip(seq,seq[1:]):
  ta,tb=TR[a]['end'],TR[b]['start'];ya,yb=pos[a],pos[b];ax.plot([ta,tb],[ya,yb],color='#126d9d',lw=1.2)
  for c in r['charges']:
   if ta-1e-5<=c['start'] and c['end']<=tb+1e-5:
    ax.plot([c['start'],c['end']],[ya+(yb-ya)*(x-ta)/(tb-ta) for x in [c['start'],c['end']]],color='#e88d20',lw=5,solid_capstyle='butt')
 for c in r['charges']:export.append(dict(arm=r['label'],event='charging',start_min=c['start'],end_min=c['end'],source_trip_id='',station=c['station'],kwh=c['kwh']))
 ax.set_title(r['label'],fontsize=12,loc='left');ax.set_ylim(-.8,len(ordered)-.2);ax.set_xlim(240,1500);ax.set_xticks(range(240,1501,240),[f'{x//60:02}:00' for x in range(240,1501,240)]);ax.set_yticks(range(len(ordered)),[TR[t]['source_id'] for t in ordered]);ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
 if col==0:ax.set_ylabel('Source trip ID (departure order)')
 sx=axs[1,col];tx,sy=zip(*r['trace']);sx.plot(tx,sy,color='#126d9d',lw=1.7);sx.axhline(pr.reserve_kwh,color='#bd3e3e',ls='--',lw=1);sx.set_ylim(0,245);sx.set_xlim(240,1500);sx.set_xticks(range(240,1501,240),[f'{x//60:02}:00' for x in range(240,1501,240)]);sx.set_yticks([0,120,236.44],['0','120','236.44']);sx.set_xlabel('Time of day');sx.grid(alpha=.15);sx.spines[['top','right']].set_visible(False)
 if col==0:sx.set_ylabel('Battery (kWh)')
fig.legend(handles=[Line2D([0],[0],color='#126d9d',lw=5,label='Service trip'),Line2D([0],[0],color='#126d9d',lw=1,label='Same bus connection'),Line2D([0],[0],color='#e88d20',lw=5,label='Charging'),Line2D([0],[0],color='#bd3e3e',ls='--',lw=1,label='15% reserve')],loc='outside lower center',ncol=4,frameon=False)
fig.savefig(P/'one_bus_k5_joint_matched.png',dpi=200);fig.savefig(P/'one_bus_k5_joint_matched.pdf')
with (P/'one_bus_k5_joint_matched.csv').open('w') as out:
 w=csv.DictWriter(out,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
print(json.dumps(summary,indent=2));print([(r['label'],len(r['trips']),len(r['charges'])) for r in panels])
