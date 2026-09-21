from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
minutes=lambda s:sum(int(x)*m for x,m in zip(s.split(':'),(60,1)))
rows=[r for r in csv.DictReader((ROOT/'data/Par_VehicleDetails_Updated.csv').open()) if r['VehicleTask']=='13414']
trips=[r for r in rows if r['Identifier']=='Regular'];order={r['Ordered_Trip_ID']:i for i,r in enumerate(trips)}
files=['baseline.json','fee0.json','fee5.json'];labels=['Recorded GIRO','Fixed original windows · fee 0','Fixed original windows · fee 5']
fig,axs=plt.subplots(2,3,figsize=(15,7.3),gridspec_kw={'height_ratios':[3,1]},layout='constrained')
flat=[]
for col,(file,label) in enumerate(zip(files,labels)):
 r=next(r for r in json.loads((P/file).read_text())['routes'] if r['duty']=='13414');ax=axs[0,col]
 for t in trips:
  a,b=minutes(t['Start1']),minutes(t['End1']);i=order[t['Ordered_Trip_ID']];ax.plot([a,b],[i,i],color='#126d9d',lw=5,solid_capstyle='butt');flat.append(dict(arm=label,event='service',start_min=a,end_min=b,trip_id=t['Ordered_Trip_ID'],station='',kwh=''))
 for i,(a,b) in enumerate(zip(trips,trips[1:])):
  ta,tb=minutes(a['End1']),minutes(b['Start1']);ax.plot([ta,tb],[i,i+1],color='#126d9d',lw=1.3)
  for c in r['charges']:
   if ta<=c['start'] and c['end']<=tb:
    for seg in c['segments']:
     st=seg['start'];et=st+seg['active_min']
     if et-st<1e-4:continue
     yy=[i+(x-ta)/(tb-ta) for x in [st,et]];ax.plot([st,et],yy,color='#e88d20',lw=5,solid_capstyle='butt')
 for c in r['charges']:flat.append(dict(arm=label,event='charging_connection',start_min=c['start'],end_min=c['end'],trip_id='',station=c['station'],kwh=c['kwh']))
 ax.set_title(label,fontsize=13,loc='left');ax.set_yticks(range(len(trips)),[t['Ordered_Trip_ID'] for t in trips]);ax.set_xlim(390,1310);ax.set_xticks(range(420,1321,120),[f'{x//60:02}:00' for x in range(420,1321,120)]);ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
 if col==0:ax.set_ylabel('Source trip ID (departure order)')
 sx=axs[1,col];tx,sy=zip(*r['trace']);sx.plot(tx,sy,color='#126d9d',lw=1.7);sx.axhline(.15*236.44,color='#bd3e3e',ls='--',lw=1);sx.set_ylim(0,245);sx.set_xlim(390,1310);sx.set_xticks(range(420,1321,120),[f'{x//60:02}:00' for x in range(420,1321,120)]);sx.set_yticks([0,120,236.44],['0','120','236.44']);sx.set_xlabel('Time of day');sx.grid(alpha=.15);sx.spines[['top','right']].set_visible(False)
 if col==0:sx.set_ylabel('Battery (kWh)')
fig.legend(handles=[Line2D([0],[0],color='#126d9d',lw=5,label='Service trip'),Line2D([0],[0],color='#126d9d',lw=1,label='Same bus connection'),Line2D([0],[0],color='#e88d20',lw=5,label='Charging'),Line2D([0],[0],color='#bd3e3e',ls='--',lw=1,label='15% reserve')],loc='outside lower center',ncol=4,frameon=False)
fig.savefig(P/'one_bus_k5_matched.png',dpi=200);fig.savefig(P/'one_bus_k5_matched.pdf')
with (P/'one_bus_k5_matched.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
