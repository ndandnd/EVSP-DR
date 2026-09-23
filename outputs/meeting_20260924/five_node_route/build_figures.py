"""Meeting figures from two frozen, previously validated schedules; no solver."""
from pathlib import Path
import json,hashlib,csv,ast,collections,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
P=Path(__file__).resolve().parent;R=P.parents[2]
S=R/'outputs/research_followup_20260921/duty13309'
ss=json.loads((S/'schedules.json').read_text())[:2]
assert [s['schedule_id'] for s in ss]==['original_13309','w6_k05_fee0']
def service(s):return [(str(e['source_trip_id']),e['start_min'],e['end_min'],e['from_code'],e['to_code']) for e in s['events'] if e['kind']=='service']
assert service(ss[0])==service(ss[1]) and len(service(ss[0]))==22
assert [len([e for e in s['events'] if e['kind']=='charge']) for s in ss]==[4,7]
assert not any(e['from_code']=='PARX' for e in ss[1]['events'] if e['kind']=='charge')
# Reuse the archived graph renderer without executing its extraction/writing code.
source=ast.parse((S/'build_figures.py').read_text());selected=[]
for node in source.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['POS','NAMES','BLUE','GREY','ORANGE','DARK'] for t in node.targets):selected.append(node)
 if isinstance(node,ast.FunctionDef) and node.name in ['compact','draw']:selected.append(node)
exec(compile(ast.Module(body=selected,type_ignores=[]),str(S/'build_figures.py'),'exec'),globals())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(14,6.3))
for ax,s,title in zip(axes,ss,['Recorded GIRO duty 13309','Our saved route: C6 k5, fee 0']):
 draw(s,ax);ax.set_xlim(-1.2,13.2);ax.set_ylim(-1.35,5.8)
 for t in ax.texts:
  t.set_fontsize(11 if t.get_fontweight()=='bold' else 10)
 ax.set_title(title,fontsize=15,pad=18)
leg=[Line2D([0],[0],color=BLUE,lw=2,label='Passenger trips (L = order in the day)'),Line2D([0],[0],color=GREY,lw=1.7,ls='--',label='Empty driving (M)'),Line2D([0],[0],marker='o',ms=10,mfc='white',mec=ORANGE,mew=2,color='none',label='Charging site used')]
fig.legend(handles=leg,loc='lower center',ncol=3,frameon=False,fontsize=11,bbox_to_anchor=(.5,.015))
fig.subplots_adjust(left=.025,right=.985,top=.9,bottom=.13,wspace=.12)
for ext in ['png','pdf','svg']:fig.savefig(P/f'13309_node_comparison.{ext}',dpi=190,facecolor='white')
plt.close(fig)
# Trip-ID ladder above, charging-site timing below. Same row for each source trip.
fig=plt.figure(figsize=(14,6.3));gs=fig.add_gridspec(2,2,height_ratios=[3.8,1.05],hspace=.15,wspace=.17)
ids=[r[0] for r in service(ss[0])];row={t:i+1 for i,t in enumerate(ids)}
for j,(s,title) in enumerate(zip(ss,['Recorded GIRO duty 13309','Our saved route: C6 k5, fee 0'])):
 ax=fig.add_subplot(gs[0,j]);cx=fig.add_subplot(gs[1,j],sharex=ax);prev=None;lastrow=0
 for e in s['events']:
  t0,t1=e['start_min']/60,e['end_min']/60
  if e['kind']=='service':
   y=row[str(e['source_trip_id'])]
   if prev is not None:ax.plot([prev[0],t0],[prev[1],y],lw=.8,color=BLUE,alpha=.5)
   ax.plot([t0,t1],[y,y],lw=3,color=BLUE,solid_capstyle='butt');prev=(t1,y);lastrow=y
  elif e['kind']=='charge':
   y=lastrow+.43
   ax.plot([t0,t1],[y,y],lw=3.4,color=ORANGE,solid_capstyle='butt')
   site={'3127':'Heden','13410':'Jons väg','13801':'PARX'}[e['from_ref']];cy={'Heden':2,'Jons väg':1,'PARX':0}[site]
   cx.plot([t0,t1],[cy,cy],lw=8,color=ORANGE,solid_capstyle='butt')
 ax.set_title(title,fontsize=15,pad=12);ax.set_yticks(range(1,23),ids,fontsize=9);ax.set_ylim(.1,22.9)
 ax.grid(axis='x',color='#e4e8ec',lw=.6);ax.tick_params(axis='x',labelbottom=False);ax.set_xlim(5,19.5)
 for a in (ax,cx):a.spines[['top','right']].set_visible(False);a.spines[['bottom','left']].set_color('#aab5bd')
 cx.set_ylim(-.55,2.55);cx.set_yticks([0,1,2],['PARX','Jons väg','Heden'],fontsize=10);cx.set_xticks([6,8,10,12,14,16,18],[f'{h:02d}:00' for h in [6,8,10,12,14,16,18]]);cx.grid(axis='x',color='#e4e8ec',lw=.6);cx.set_xlabel('Time of day')
 if j==0:ax.set_ylabel('Source trip ID (departure order)');cx.annotate('10:45–12:30',xy=(11.6,0),xytext=(11.6,.55),ha='center',fontsize=10,color=ORANGE)
fig.legend(handles=[Line2D([0],[0],color=BLUE,lw=3,label='Passenger trip'),Line2D([0],[0],color=BLUE,lw=.8,alpha=.5,label='Same bus, next trip'),Line2D([0],[0],color=ORANGE,lw=4,label='Charging')],loc='lower center',ncol=3,frameon=False,fontsize=11,bbox_to_anchor=(.5,-.005))
fig.subplots_adjust(left=.075,right=.985,top=.92,bottom=.115)
for ext in ['png','pdf','svg']:fig.savefig(P/f'13309_time_comparison.{ext}',dpi=190,facecolor='white')
plt.close(fig)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'source_schedule_sha256':sha(S/'schedules.json'),'source_renderer_sha256':sha(S/'build_figures.py'),'script_sha256':sha(Path(__file__)),'schedules':[s['schedule_id'] for s in ss],'checks':{'same22ordered_trip_records':True,'charge_counts':[4,7],'our_route_no_depot_charge':True},'scope':'Historical saved schedules; different physics. No new optimizer/physical validation, no cost saving claim. Graph shows all cross-area movements; local moves remain in archived itinerary. Trip labels are prepared source IDs.','sources':str(S.relative_to(R)),'figures':{p.name:sha(p) for p in P.glob('13309_*') if p.suffix in ['.png','.pdf','.svg']}}
(P/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Validated 22 identical passenger trip records; generated paired node/time figures.')
