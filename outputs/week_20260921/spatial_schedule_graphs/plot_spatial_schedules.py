#!/usr/bin/env python3
"""Exact-data spatial multigraphs and visit-expanded itinerary diagrams."""
from pathlib import Path
import argparse, csv, json, math, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
BLUE='#246b9a'; ORANGE='#cb711f'; DARK='#253d4e'; GREY='#83949e'; LIGHT='#e8edf1'
ARM_NAMES={'original':'Original GIRO','fee0':'Reoptimized · fee 0','fee5':'Reoptimized · fee 5'}
# Approximate geographic L2 positions: equirectangular projection at57.73N,
# normalized by west-to-east distance from Eketrägatan to Merkuriusgatan.
POS={'2190':(0.,0.),'2190L':(0.,0.),'4808':(1.,.526),'PARX':(1.100,.241)}

def clock(x):
    sec=round(float(x)*60); h=sec//3600; m=(sec%3600)//60; s=sec%60
    return f'{h:02}:{m:02}' if s==0 else f'{h:02}:{m:02}:{s:02}'

def interval(a,b):return f'{clock(a)}–{clock(b)}'

def compact_minutes(x):
    return f'{x:.0f}' if abs(x-round(x))<.005 else f'{x:.1f}'

def legacy_example():
    d=json.loads((HERE.parent/'geography_map/model_geography.json').read_text());schedules=[]
    for raw,arm in [('original_giro_13414','original'),('saved_joint_fee0','fee0'),('saved_joint_fee5','fee5')]:
        row=next(x for x in d['selected_buses'] if x['arm']==raw); events=[]
        for e in d['trips']:
            if e['arm']==raw:events.append(dict(kind='service',start_min=e['start_min'],end_min=e['end_min'],from_code=e['from_code'],to_code=e['to_code'],source_trip_id=e['source_trip_id'],energy_consumed_kwh=e['service_energy_kwh']))
        for e in d['movements']:
            if e['arm']==raw:events.append(dict(kind='deadhead',start_min=e['start_min'],end_min=e['end_min'],from_code=e['from_code'],to_code=e['to_code'],energy_consumed_kwh=e['energy_kwh']))
        for e in d['charging']:
            if e['arm']==raw:events.append(dict(kind='charge',start_min=e['start_min'],end_min=e['end_min'],from_code=e['station'],to_code=e['station'],energy_charged_kwh=e['energy_kwh']))
        events.sort(key=lambda e:(e['start_min'],e['end_min'],e['kind']!='deadhead'))
        for i,e in enumerate(events):e['event_index']=i
        travel=[e for e in events if e['kind'] in ['service','deadhead']]
        schedules.append(dict(schedule_id=f'duty_13414_{arm}',pair_id='duty_13414',arm=arm,source_index=row['source_bus_index'],metrics=dict(depot_departure_min=min(e['start_min'] for e in travel),depot_return_min=max(e['end_min'] for e in travel),starts=row['charging_starts'],charged_kwh=sum(e.get('energy_charged_kwh',0) for e in events),trip_count=row['service_trip_count']),events=events))
    return dict(pairs=[dict(pair_id='duty_13414',original_duty='13414')],schedules=schedules)

def normalize_arm(s):
    arm=s['arm'].lower()
    if 'original' in arm or 'giro' in arm:return 'original'
    if 'fee0' in arm or 'fee_0' in arm:return 'fee0'
    if 'fee5' in arm or 'fee_5' in arm:return 'fee5'
    raise ValueError(arm)

def events(s,kind):return sorted([e for e in s['events'] if e['kind']==kind],key=lambda e:(e['start_min'],e['end_min']))

def geographic_site(code):
    return '2190' if code=='2190L' else code

def spatial_panel(fig,grid,s,common_n,graph_only=False):
    if graph_only:
        ax=fig.add_subplot(grid);ax.axis('off')
    else:
        sub=grid.subgridspec(3,1,height_ratios=[3.6,3.2,2.7],hspace=.06)
        ax=fig.add_subplot(sub[0]); leg=fig.add_subplot(sub[1]); charge=fig.add_subplot(sub[2])
        for a in [ax,leg,charge]:a.axis('off')
    trips=events(s,'service');charges=events(s,'charge');deadheads=events(s,'deadhead');m=s['metrics'];arm=normalize_arm(s)
    ax.text(.5,1.05 if graph_only else 1.10,ARM_NAMES[arm],ha='center',va='top',transform=ax.transAxes,fontsize=13,fontweight='bold',color=DARK)
    ax.text(.5,.995 if graph_only else 1.015,f"{len(trips)} passenger trips · {len(charges)} charge starts",ha='center',va='top',transform=ax.transAxes,fontsize=10,color=GREY)
    per_direction={}
    for i,e in enumerate(trips,1):
        key=(geographic_site(e['from_code']),geographic_site(e['to_code']));n=per_direction.get(key,0);per_direction[key]=n+1
        a=POS[key[0]];b=POS[key[1]];rad=.13+.19*n
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,color=BLUE,lw=1.25,connectionstyle=f'arc3,rad={rad}',shrinkA=13,shrinkB=13,alpha=.82,zorder=2))
        # arc3 quadratic midpoint, offset to the same side as its curved edge.
        mx=(a[0]+b[0])/2+rad*(b[1]-a[1])/2;my=(a[1]+b[1])/2-rad*(b[0]-a[0])/2
        ax.text(mx,my,f'L{i}',ha='center',va='center',fontsize=7.6,color=BLUE,bbox=dict(boxstyle='circle,pad=.16',fc='white',ec=BLUE,lw=.65),zorder=5)
    # The exact outbound and return clocks are separate from the service arcs.
    out=next(e for e in deadheads if e['from_code']=='PARX')
    back=next(e for e in reversed(deadheads) if e['to_code']=='PARX')
    for e,rad,word,offset in [(out,-.85,'OUT',(.20,.02)),(back,.55,'BACK',(.16,-.16))]:
        a=POS[geographic_site(e['from_code'])];b=POS[geographic_site(e['to_code'])]
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,color=GREY,lw=1.4,ls='--',connectionstyle=f'arc3,rad={rad}',shrinkA=12,shrinkB=12,zorder=1))
        if e['from_code']=='2190':pos=(1.43,-.27) if trips[0]['from_code']=='2190' else (.83,-.43)
        elif e['to_code']=='2190':pos=(1.40,.02)
        elif word=='OUT':pos=(1.28,.57)
        else:pos=(1.43,-.17)
        ax.text(*pos,word+'\n'+interval(e['start_min'],e['end_min']),ha='center',va='center',fontsize=8,color=GREY,bbox=dict(boxstyle='round,pad=.2',fc='white',ec='none'),zorder=6)
    # Stop+charger within Eketrägatan are grouped only in this overview.
    # Detailed actual local movements remain in the separate expanded graph.
    for code,(x,y) in POS.items():
        if code=='2190L':continue
        local=[(i,c) for i,c in enumerate(charges,1) if geographic_site(c['from_code'])==code]
        ax.scatter(x,y,s=220,marker='D' if code=='PARX' else 'o',color='white',ec=DARK,lw=1.6,zorder=8)
        if local:ax.scatter(x,y,s=420,facecolor='none',edgecolor=ORANGE,lw=2.4,zorder=7)
        text={'2190':'Eketrägatan\n2190 / 2190L','4808':'Merkuriusgatan\n4808','PARX':'PARX\nDepot'}[code]
        xytext={'2190':(-12,-32),'4808':(-4,22),'PARX':(21,0)}[code]
        ax.annotate(text,(x,y),xytext=xytext,textcoords='offset points',ha='center',fontsize=9.2,color=DARK,zorder=9)
        if local and not graph_only:
            inds=', '.join(f'C{i}' for i,_ in local)
            ax.annotate(inds,(x,y),xytext=(-1,-63 if code=='2190' else 47),textcoords='offset points',ha='center',fontsize=7.8,color=ORANGE,zorder=9)
    rest=[e for e in deadheads if e['from_code']=='ET_R' or e['to_code']=='ET_R']
    if rest:
        R=(-.26,.28)
        ax.scatter(*R,s=90,facecolor='white',edgecolor=GREY,marker='s',lw=1.1,zorder=7)
        for rad,a,b in [(.2,POS['2190'],R),(.2,R,POS['2190'])]:
            ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->',mutation_scale=9,color=GREY,lw=1.0,ls=':',connectionstyle=f'arc3,rad={rad}',shrinkA=8,shrinkB=8,zorder=3))
        ax.annotate('ET_R\nUnlocated',R,xytext=(-3,14),textcoords='offset points',ha='center',fontsize=8,color=GREY,zorder=9)
    ax.set_xlim(-.42,1.70);ax.set_ylim(-.61,1.08);ax.set_aspect('equal')
    if graph_only:
        ax.set_ylim(-1.16,1.18)
        for site,x in [('2190L',-.26),('4808',.84)]:
            local=[(i,e) for i,e in enumerate(charges,1) if e['from_code']==site]
            lines=[site+' · charging / kWh']
            lines += [f"C{i}  "+interval(e['start_min'],e['end_min'])+f"  +{e['energy_charged_kwh']:.1f}" for i,e in local]
            if not local:lines+=['No charging']
            ax.text(x,-.59,'\n'.join(lines),ha='left',va='top',fontsize=8.2,color=ORANGE,linespacing=1.48,bbox=dict(boxstyle='round,pad=.55',fc='#fff8f1',ec='#ecd2b8',lw=.8),zorder=10)
        return
    leg.set_xlim(0,1);leg.set_ylim(0,common_n+1.4)
    leg.text(.01,common_n+1.0,'Leg',fontsize=9.5,fontweight='bold',color=DARK)
    leg.text(.13,common_n+1.0,'Source trip',fontsize=9.5,fontweight='bold',color=DARK)
    leg.text(.47,common_n+1.0,'Departure → arrival',fontsize=9.5,fontweight='bold',color=DARK)
    for i,e in enumerate(trips,1):
        y=common_n+.05-i
        if i%2==0:leg.axhspan(y-.35,y+.48,color='#f4f7f9',zorder=0)
        leg.text(.02,y,f'L{i}',fontsize=9.1,color=BLUE)
        leg.text(.18,y,str(e['source_trip_id']),fontsize=9.1,color=DARK)
        leg.text(.47,y,interval(e['start_min'],e['end_min']),fontsize=9.1,color=DARK)
    charge.set_xlim(0,1);charge.set_ylim(0,11.2)
    charge.text(.01,10.65,'Charge',fontsize=9.5,fontweight='bold',color=ORANGE)
    charge.text(.20,10.65,'Site',fontsize=9.5,fontweight='bold',color=DARK)
    charge.text(.39,10.65,'Connection interval',fontsize=9.5,fontweight='bold',color=DARK)
    charge.text(.96,10.65,'kWh',ha='right',fontsize=9.5,fontweight='bold',color=DARK)
    for i,e in enumerate(charges,1):
        y=10.0-i*.88
        charge.text(.035,y,f'C{i}',fontsize=8.9,color=ORANGE)
        charge.text(.20,y,e['from_code'],fontsize=8.9,color=DARK)
        charge.text(.39,y,interval(e['start_min'],e['end_min']),fontsize=8.4,color=DARK)
        charge.text(.98,y,f"{e['energy_charged_kwh']:.1f}",ha='right',fontsize=8.8,color=ORANGE)

def spatial_triptych(group,pathbase):
    fig=plt.figure(figsize=(17.4,11.6));grid=fig.add_gridspec(1,3,left=.035,right=.98,top=.94,bottom=.055,wspace=.18)
    maxn=max(len(events(s,'service')) for s in group)
    for i,s in enumerate(group):spatial_panel(fig,grid[i],s,maxn)
    fig.text(.5,.028,'L = passenger-leg order, not source Trip ID · grey depot moves · orange charging nodes · approximate geographic layout',ha='center',fontsize=10,color=DARK)
    fig.text(.5,.010,'Within-terminal movements, ET_R rest visits, waits and preparation appear in the full itinerary. Clocks rounded to the nearest second.',ha='center',fontsize=8.5,color=GREY)
    for ext in ['png','pdf','svg']:fig.savefig(pathbase.with_suffix('.'+ext),dpi=190,facecolor='white')
    plt.close(fig)

def graph_triptych(group,pathbase):
    fig=plt.figure(figsize=(17.4,7.5));grid=fig.add_gridspec(1,3,left=.035,right=.98,top=.93,bottom=.075,wspace=.18)
    maxn=max(len(events(s,'service')) for s in group)
    for i,s in enumerate(group):spatial_panel(fig,grid[i],s,maxn,graph_only=True)
    fig.text(.5,.034,'L = passenger-leg order, not source Trip ID · grey depot moves · orange charging nodes · approximate geographic layout',ha='center',fontsize=9.5,color=DARK)
    fig.text(.5,.014,'Source Trip IDs and clocked legs are in the editable key; complete waits, local movements, ET_R visits and preparation are in the itinerary.',ha='center',fontsize=8.5,color=GREY)
    for ext in ['png','pdf','svg']:fig.savefig(pathbase.with_suffix('.'+ext),dpi=190,facecolor='white')
    plt.close(fig)

def displayed_dwell(s,v):
    return [s['events'][i] for i in v['event_indices'] if not (s['events'][i]['kind']=='deadhead' and abs(s['events'][i]['end_min']-s['events'][i]['start_min'])<1e-8)]

def visit_positions(s):
    positions={};y=0.0
    for v in s['visits']:
        dwell=displayed_dwell(s,v)
        lines=max(1,len(dwell))
        positions[v['visit_index']]=y
        y+=1.35+.44*lines
    return positions,y

def expanded_panel(ax,s,total_height):
    ax.axis('off');arm=normalize_arm(s);xx={'ET_R':-.25,'2190':.0,'2190L':.22,'4808':1.0,'PARX':1.45};positions,_=visit_positions(s)
    ax.set_xlim(-.65,4.7);ax.set_ylim(total_height+.5,-2.6)
    ax.text(1.8,-2.4,ARM_NAMES[arm],ha='center',fontsize=14,fontweight='bold',color=DARK)
    for code,x in xx.items():
        ax.plot([x,x],[-.75,total_height-.4],color=LIGHT,lw=.7,zorder=0)
        ax.text(x,-1.25,code,ha='center',fontsize=8,color=GREY,rotation=35)
    service_order={e['event_index']:i for i,e in enumerate(events(s,'service'),1)}
    charge_order={e['event_index']:i for i,e in enumerate(events(s,'charge'),1)}
    for e in s['edges']:
        a=(xx[e['from_code']],positions[e['from_visit']]);b=(xx[e['to_code']],positions[e['to_visit']]);is_service=e['kind']=='service'
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,lw=1.6 if is_service else 1.0,color=BLUE if is_service else GREY,ls='-' if is_service else '--',shrinkA=5,shrinkB=5,zorder=1))
        label=(f"L{service_order[e['event_index']]} · Trip {e['source_trip_id']}" if is_service else 'Move '+compact_minutes(e['duration_min'])+' min')
        if is_service:label+='\n'+interval(e['start_min'],e['end_min'])
        tx=(a[0]+b[0])/2;ty=(a[1]+b[1])/2
        ax.text(tx,ty,label,ha='center',va='center',fontsize=7.6,color=BLUE if is_service else GREY,bbox=dict(boxstyle='round,pad=.16',fc='white',ec='none',alpha=.95),zorder=4)
    for v in s['visits']:
        idx=v['visit_index'];y=positions[idx];x=xx[v['location']];dwell=displayed_dwell(s,v);charging=bool(v['charge_indices']);color=ORANGE if charging else DARK
        ax.scatter(x,y,s=44 if charging else 26,facecolor='white',edgecolor=color,lw=1.2,zorder=6,marker='D' if v['location']=='PARX' else 'o')
        label=f"V{idx+1} · {v['location']} · "+interval(v['arrival_min'],v['departure_min'])
        if abs(v['arrival_min']-v['departure_min'])<1e-7:label=f"V{idx+1} · {v['location']} · "+clock(v['arrival_min'])
        ax.text(1.75,y,label,ha='left',va='center',fontsize=8.5,fontweight='bold' if charging else 'normal',color=color)
        for j,e in enumerate(dwell):
            yy=y+.40*(j+1)
            if e['kind']=='charge':text=f"C{charge_order[e['event_index']]} charge "+interval(e['start_min'],e['end_min'])+f" · +{e['energy_charged_kwh']:.1f} kWh";col=ORANGE
            elif e['kind']=='wait':text='Wait '+interval(e['start_min'],e['end_min'])+f" · {compact_minutes(e['duration_min'])} min";col=GREY
            elif e['kind']=='prep':text='Prep '+interval(e['start_min'],e['end_min'])+f" · {compact_minutes(e['duration_min'])} min";col=GREY
            else:text=e['kind']+' '+interval(e['start_min'],e['end_min']);col=GREY
            ax.text(1.78,yy,text,ha='left',va='center',fontsize=7.8,color=col)

def expanded_triptych(group,pathbase):
    height=max(visit_positions(s)[1] for s in group)
    fig,axes=plt.subplots(1,3,figsize=(20,max(14,height*.31)),gridspec_kw={'wspace':.10})
    fig.subplots_adjust(left=.025,right=.99,top=.96,bottom=.035)
    for ax,s in zip(axes,group):expanded_panel(ax,s,height)
    fig.text(.5,.013,'Visit-expanded graph: location order left → right, chronology top → bottom; vertical spacing is not elapsed time. V = visit order; L = passenger-leg order.',ha='center',fontsize=10,color=DARK)
    for ext in ['png','pdf','svg']:fig.savefig(pathbase.with_suffix('.'+ext),dpi=180,facecolor='white')
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prototype',action='store_true');args=parser.parse_args()
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'evsp-spatial-schedules-20260921','pdf.fonttype':42})
    data=json.loads((HERE/'schedules.json').read_text()) if (HERE/'schedules.json').exists() else legacy_example()
    for pair in data['pairs']:
        if args.prototype and pair['pair_id']!='duty_13414':continue
        group=sorted([s for s in data['schedules'] if s['pair_id']==pair['pair_id']],key=lambda s:['original','fee0','fee5'].index(normalize_arm(s)))
        if args.prototype:spatial_triptych(group,HERE/('prototype_'+pair['pair_id']+'_spatial'))
        graph_triptych(group,HERE/(('prototype_' if args.prototype else '')+pair['pair_id']+'_graph'))
        if 'visits' in group[0]:expanded_triptych(group,HERE/(('prototype_' if args.prototype else '')+pair['pair_id']+'_itinerary'))
    print('Rendered spatial triptychs')

if __name__=='__main__':main()
