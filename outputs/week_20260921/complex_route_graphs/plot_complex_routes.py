#!/usr/bin/env python3
"""Recorded GIRO multi-terminal duties in a geographic node/edge layout."""
from pathlib import Path
import argparse,csv,json,math
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BLUE='#246b9a';ORANGE='#cb711f';DARK='#253d4e';GREY='#71808c';LIGHT='#e8edf1'

def clock(x):
 s=round(float(x)*60);h=s//3600;m=s%3600//60;s=s%60
 return f'{h:02}:{m:02}' if not s else f'{h:02}:{m:02}:{s:02}'
def interval(a,b):return clock(a)+'–'+clock(b)
def coords():
 c={x['area']:{**x,'xy':(float(x['east_km']),float(x['north_km']))} for x in csv.DictReader((HERE/'coordinates.csv').open())}
 c['13801']['xy']=(11.2,1.1)  # Explicit display displacement; actual coordinates remain in CSV.
 return c

def arc(ax,a,b,rad,color=BLUE,ls='-',lw=1.25,gid=None):
 p=FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,color=color,lw=lw,ls=ls,connectionstyle=f'arc3,rad={rad}',shrinkA=15,shrinkB=15,zorder=2)
 if gid:p.set_gid(gid)
 ax.add_patch(p)
 return ((a[0]+b[0])/2+rad*(b[1]-a[1])/2,(a[1]+b[1])/2-rad*(b[0]-a[0])/2)

def read_schedule(duty='13309'):
 d=json.loads((HERE/'schedules.json').read_text());s=next(s for s in d['schedules'] if s['duty_id']==duty)
 s['duty']=duty
 for e in s['events']:e['from_area']=e['from_ref'];e['to_area']=e['to_ref']
 return s

def panel(ax,s,part):
 C=coords();alltrips=[e for e in s['events'] if e['kind']=='service'];allmoves=[e for e in s['events'] if e['kind']=='deadhead' and e['from_ref']!=e['to_ref']]
 lower,upper=(0,642) if part=='morning' else ((763,1500) if part=='afternoon' else (0,1500))
 ax.axis('off');ax.set_aspect('equal');ax.set_xlim(-2.1,14.6);ax.set_ylim(-2.6,7.6)
 if part=='full':ax.set_ylim(-3.6,8.5)
 counts=defaultdict(int)
 for i,e in enumerate(alltrips,1):
  if not lower<=e['start_min']<upper:continue
  a,b=C[e['from_ref']]['xy'],C[e['to_ref']]['xy'];key=e['from_ref'],e['to_ref'];n=counts[key];counts[key]+=1
  rad=.14+.27*n
  x,y=arc(ax,a,b,rad,gid=f'leg-{i}')
  label=ax.text(x,y,f'L{i}',ha='center',va='center',fontsize=9.4,color=BLUE,bbox=dict(boxstyle='circle,pad=.18',fc='white',ec=BLUE,lw=.75),zorder=5);label.set_gid(f'leg-label-{i}')
 for i,e in enumerate(allmoves,1):
  if not lower<=e['start_min']<upper:continue
  a,b=C[e['from_ref']]['xy'],C[e['to_ref']]['xy'];rad={1:-.30,2:-.10,3:.35,4:-.77,5:.50}[i]
  x,y=arc(ax,a,b,rad,GREY,'--',1.5,gid=f'move-{i}')
  if i==5:
   t=.25;control=((a[0]+b[0])/2+rad*(b[1]-a[1]),(a[1]+b[1])/2-rad*(b[0]-a[0]));x=(1-t)**2*a[0]+2*(1-t)*t*control[0]+t*t*b[0];y=(1-t)**2*a[1]+2*(1-t)*t*control[1]+t*t*b[1]
  ax.text(x,y,f'M{i}',ha='center',va='center',fontsize=9,color=GREY,bbox=dict(boxstyle='circle,pad=.18',fc='white',ec=GREY,lw=.7),zorder=5)
 names={'3127':'Heden','7581':'Gamlestads Torg','13215':'Partille centrum','13410':'Jons väg','13801':'PARX depot'}
 labels={'3127':(-.25,-.66),'7581':(.8,4.30),'13215':(6.4,6.3),'13410':(12.5,5.55),'13801':(12.15,.25)}
 for code,v in C.items():
  xy=v['xy'];charging=code in ['3127','13410','13801']
  if charging:ax.scatter(*xy,s=420,facecolor='none',edgecolor=ORANGE,lw=2.3,zorder=7)
  ax.scatter(*xy,s=215,color='white',edgecolor=DARK,lw=1.5,marker='D' if code=='13801' else 'o',zorder=8)
  ax.annotate(names[code],xy,xytext=labels[code],fontsize=10.4,ha='center',fontweight='bold',color=DARK,arrowprops=dict(arrowstyle='-',color=GREY,lw=.7) if code in ['7581','13215'] else None,bbox=dict(fc='white',ec='none',pad=.25),zorder=9)

def split_graph(s,pathbase):
 fig=plt.figure(figsize=(17.4,8.7));left=fig.add_axes([.025,.33,.46,.56]);right=fig.add_axes([.515,.33,.46,.56])
 panel(left,s,'morning');panel(right,s,'afternoon')
 fig.text(.04,.956,'GIRO duty 13309 · five places, three charging sites',fontsize=17,fontweight='bold',color=DARK)
 fig.text(.04,.917,'Recorded schedule · same approximate layout in both panels · depot displaced for clarity · 22 passenger trips',fontsize=10.6,color=GREY)
 fig.text(.25,.87,'Morning · 05:19–10:42',ha='center',fontsize=13,fontweight='bold',color=DARK)
 fig.text(.75,.87,'Afternoon · 12:43–19:02',ha='center',fontsize=13,fontweight='bold',color=DARK)
 # Empty-movement clocks form short keys outside the edges, ordered across the day.
 fig.text(.045,.327,'M1  PARX → Heden  05:19–05:37\nM2  Partille centrum → PARX  10:37–10:42',ha='left',va='top',fontsize=9.6,color=GREY,linespacing=1.6)
 fig.text(.535,.327,'M3  PARX → Jons väg  12:43–12:54\nM4  Partille centrum → Heden  16:07–16:22\nM5  Heden → PARX  18:47–19:02',ha='left',va='top',fontsize=9.6,color=GREY,linespacing=1.6)
 cards=[(.04,'Morning charging','C1  Heden · 3127L  06:42–06:56  +40.6 kWh\nC2  Jons väg · JON_A  09:15–09:25  +44.8 kWh'),(.397,'Midday depot charging','C3  PARX  10:45–12:30\n+105.0 kWh · 105 minutes'),(.715,'Afternoon charging','C4  Heden · 3127L\n16:22–16:49  +130.7 kWh')]
 for x,title,body in cards:fig.text(x,.207,title+'\n'+body,ha='left',va='top',fontsize=10.3,color=ORANGE,linespacing=1.58,bbox=dict(boxstyle='round,pad=.55',fc='#fff8f1',ec='#ecd2b8',lw=.8))
 fig.text(.5,.072,'L = passenger-leg order, not Trip ID · M = inter-area empty-move order · orange rings identify charging sites',ha='center',fontsize=9.5,color=DARK)
 fig.text(.5,.047,'The editable itinerary preserves all trip clocks, numbered visits, platform movements, waits and preparation. This is not a newly optimized schedule.',ha='center',fontsize=8.8,color=GREY)
 fig.text(.5,.024,'Approximate layout, not road paths; depot displaced for clarity. Stop-area proxies © OpenStreetMap contributors; actual coordinates retained in CSV.',ha='center',fontsize=8.5,color=GREY)
 for ext in ['png','pdf','svg']:fig.savefig(pathbase.with_suffix('.'+ext),dpi=190,facecolor='white')
 plt.close(fig)

def full_graph(s,pathbase):
 fig=plt.figure(figsize=(15.6,9));ax=fig.add_axes([.03,.24,.94,.67]);panel(ax,s,'full')
 fig.text(.04,.955,'GIRO duty 13309 · optional full-day multigraph',fontsize=17,fontweight='bold',color=DARK)
 fig.text(.04,.917,'22 service legs · use the split view and editable itinerary for clearer chronology',fontsize=11,color=GREY)
 charges=[e for e in s['events'] if e['kind']=='charge']
 for code,x in [('3127',.04),('13410',.40),('13801',.71)]:
  local=[(i,e) for i,e in enumerate(charges,1) if e['from_ref']==code]
  text=[{'3127':'Heden · 3127L','13410':'Jons väg · JON_A','13801':'PARX · depot'}[code]]
  text += [f"C{i}  "+interval(e['start_min'],e['end_min'])+f"  +{e['energy_charged_kwh']:.1f} kWh" for i,e in local]
  fig.text(x,.185,'\n'.join(text),ha='left',va='top',fontsize=10.3,color=ORANGE,linespacing=1.6,bbox=dict(boxstyle='round,pad=.55',fc='#fff8f1',ec='#ecd2b8',lw=.8))
 fig.text(.5,.06,'L = service-leg order · M = inter-area empty-move order · C = charging-session order · full clocks and platforms in the itinerary',ha='center',fontsize=9.5,color=DARK)
 fig.text(.5,.032,'Approximate layout, depot displaced for clarity; edges are not road paths. Geographic proxies © OpenStreetMap contributors.',ha='center',fontsize=9,color=GREY)
 for ext in ['png','pdf','svg']:fig.savefig(pathbase.with_suffix('.'+ext),dpi=190,facecolor='white')
 plt.close(fig)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prototype',action='store_true');args=ap.parse_args()
 plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'evsp-complex-duty-20260921','pdf.fonttype':42})
 s=read_schedule()
 prefix='prototype_' if args.prototype else ''
 split_graph(s,HERE/(prefix+'duty_13309_graph'))
 full_graph(s,HERE/(prefix+'duty_13309_full_day'))
 print('Rendered duty 13309')
if __name__=='__main__':main()
