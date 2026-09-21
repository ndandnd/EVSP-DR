#!/usr/bin/env python3
"""Geographic context for the exact buses in the matched k5 charging figure.

Coordinates locate stop/address proxies; connectors are not road polylines.
No coordinate is invented for ET_R or the unresolved 2190 passenger platform.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'outputs/post_meeting_20260910/decomposition'
SOURCES = HERE / 'sources'

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)

def osm_context(path):
    tree = ET.parse(path).getroot()
    nodes = {e.attrib['id']: (float(e.attrib['lon']), float(e.attrib['lat'])) for e in tree.findall('node')}
    ways = []
    for e in tree.findall('way'):
        tags = {t.attrib['k']: t.attrib['v'] for t in e.findall('tag')}
        pts = [nodes[n.attrib['ref']] for n in e.findall('nd') if n.attrib['ref'] in nodes]
        if pts:
            ways.append((tags, pts))
    return nodes, ways

def map_axes(ax):
    ax.set_facecolor('#f9fbfc')
    ax.grid(color='#9daeb9', alpha=.18, linewidth=.55)
    ax.set_aspect(1 / math.cos(math.radians(57.73)))
    for s in ax.spines.values():
        s.set_color('#ced8df')
    ax.tick_params(labelsize=9, colors='#50616e')

def scale(ax, x, y, km, label):
    length = km / (111.32 * math.cos(math.radians(57.73)))
    ax.plot([x, x+length], [y, y], lw=2.4, color='#263f50', zorder=10)
    ax.annotate(label, (x+length/2, y), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=9, color='#263f50')

def main():
    SOURCES.mkdir(exist_ok=True)
    for name in ('osm_roads_rivers_20260910.json', 'osm_named_stops_20260910.json', 'osm_named_stops_extra_20260910.json'):
        if not (SOURCES/name).exists():
            shutil.copy2(OLD / 'sources' / name, SOURCES / name)
    nodes = {}
    for name in ('osm_named_stops_20260910.json', 'osm_named_stops_extra_20260910.json'):
        for e in json.loads((SOURCES/name).read_text())['elements']:
            nodes[(e['type'], e['id'])] = e
    # Archived primary OSM data; these are stop positions, not surveyed chargers.
    a = nodes[('node', 241780711)]
    b = nodes[('node', 648217190)]
    address_url = 'https://transdev.se/wp-content/uploads/2024/12/Certifikat_Transdev-Sverige-AB_ms_2024-12-03-2026-05-31-002.pdf'
    geocode_url = 'https://www.hitta.se/v%C3%A4stra%2Bg%C3%B6talands%2Bl%C3%A4n/partille/j%C3%A4rnringen%2B5/omr%C3%A5de/57.737614:12.086805'
    coords = [
        dict(code='2190L', name='Eketrägatan departure charger', latitude=a['lat'], longitude=a['lon'], plotted=True, scope='OSM stop L proxy; not surveyed charger', confidence='high for named stop L; proxy for charger', source='https://www.openstreetmap.org/node/241780711', additional_source='GIRO_EMAIL_ATTACHMENT_AUDIT.md'),
        dict(code='4808', name='Merkuriusgatan terminal and charger', latitude=b['lat'], longitude=b['lon'], plotted=True, scope='OSM stop proxy; not surveyed charger', confidence='high for named stop; proxy for charger', source='https://www.openstreetmap.org/node/648217190', additional_source='GIRO_EMAIL_ATTACHMENT_AUDIT.md'),
        dict(code='PARX', name='Partille garage', latitude=57.737614, longitude=12.086805, plotted=True, scope='Transdev Järnringen 5 address proxy; exact historical PARX gate/charger not established', confidence='medium: address verified by operator; address geocode by Hitta; PARX binding is inferred', source=address_url, additional_source=geocode_url),
        dict(code='2190', name='Eketrägatan passenger terminal', latitude='', longitude='', plotted=False, scope='Terminal vicinity shown as an area; exact passenger platform unresolved', confidence='terminal identified; precise passenger platform unknown', source='https://www.openstreetmap.org/node/241780711', additional_source='GIRO_EMAIL_ATTACHMENT_AUDIT.md'),
        dict(code='ET_R', name='Original duty layover', latitude='', longitude='', plotted=False, scope='Unlocated; retained in travel table only', confidence='unknown; no geographic point plotted', source='data/Par_VehicleDetails_Updated.csv', additional_source=''),
    ]
    context=[]
    for code,name,kind,oid,ports in [('3127L','Heden','relation',2766184,2),('7880C','Östra Sjukhuset','relation',2208859,1),('JON_A','Jons väg','node',2097238852,1)]:
        e=nodes[(kind,oid)];point=e.get('center',e)
        context.append(dict(code=code,name=name,latitude=point['lat'],longitude=point['lon'],ports=ports))
        coords.append(dict(code=code,name=name,latitude=point['lat'],longitude=point['lon'],plotted=True,scope='Network context figure only; stop/stop-area proxy, not surveyed charger; not used by selected 18E1 buses',confidence='high for named stop/stop-area; proxy for charger',source=f'https://www.openstreetmap.org/{kind}/{oid}',additional_source='GIRO_EMAIL_ATTACHMENT_AUDIT.md'))
    write_csv(HERE/'coordinates.csv', coords)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(14.3, 7.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.35, 1], wspace=.17, left=.06, right=.985, bottom=.145, top=.93)
    ax = fig.add_subplot(gs[0, 0]); detail = fig.add_subplot(gs[0, 1])
    map_axes(ax); map_axes(detail)
    for e in json.loads((SOURCES/'osm_roads_rivers_20260910.json').read_text())['elements']:
        xy = e.get('geometry', [])
        if not xy: continue
        water = 'waterway' in e.get('tags', {})
        ax.plot([p['lon'] for p in xy], [p['lat'] for p in xy], color='#a4cddb' if water else '#d7dfe4', lw=3.0 if water else .8, zorder=0)
    A = (a['lon'], a['lat']); B = (b['lon'], b['lat']); D = (12.086805, 57.737614)
    blue='#246b9a'; orange='#ca762e'; dark='#263f50'
    # Two different movement types. No geographic path is asserted by these arcs.
    ax.add_patch(FancyArrowPatch(A, B, arrowstyle='<->', mutation_scale=13, color=blue, lw=2.8, connectionstyle='arc3,rad=-.10', zorder=3, shrinkA=9, shrinkB=9))
    ax.add_patch(FancyArrowPatch(B, D, arrowstyle='<->', mutation_scale=12, color=orange, lw=2.2, linestyle=(0,(5,3)), zorder=3, shrinkA=9, shrinkB=9))
    ax.add_patch(FancyArrowPatch(A, D, arrowstyle='->', mutation_scale=12, color=orange, lw=2.2, linestyle=(0,(5,3)), connectionstyle='arc3,rad=.12', zorder=3, shrinkA=9, shrinkB=9))
    box = dict(boxstyle='round,pad=.3', fc='white', ec='none', alpha=.96)
    ax.text(11.994,57.754,'Passenger service\nGIRO 53–63 min\nfee0 51–63 min',color=blue,ha='center',va='center',fontsize=11,bbox=box,zorder=6)
    ax.text(12.105,57.751,'7 min\n7.8 kWh',color=orange,ha='center',va='center',fontsize=10.5,bbox=box,zorder=6)
    ax.text(11.997,57.726,'19 min · 28.0 kWh\nfee0 return to depot',color=orange,ha='center',va='center',fontsize=10.5,bbox=box,zorder=6)
    ax.scatter(*A,s=135,marker='s',color=blue,edgecolors='white',lw=1.5,zorder=8)
    ax.scatter(*B,s=135,marker='s',color=blue,edgecolors='white',lw=1.5,zorder=8)
    ax.scatter(*D,s=135,marker='D',facecolors='white',edgecolors=orange,lw=2.2,zorder=8)
    ax.annotate('Eketrägatan\n2190 / 2190L · 1 charger', A, xytext=(-6,-25),textcoords='offset points',ha='left',va='top',fontsize=10.7,color=dark,bbox=box,zorder=9)
    ax.annotate('Merkuriusgatan\n4808 · 1 charger',B,xytext=(-8,14),textcoords='offset points',ha='center',fontsize=10.7,color=dark,bbox=box,zorder=9)
    ax.annotate('PARX · Partille garage\n60 kW · address proxy',D,xytext=(4,-21),textcoords='offset points',ha='center',va='top',fontsize=10.7,color=dark,bbox=box,zorder=9)
    ax.text(11.973,57.706,'Göteborg',fontsize=11,color='#8496a1',style='italic')
    ax.text(12.101,57.742,'Partille',fontsize=10,color='#8496a1',style='italic')
    ax.set_xlim(11.89,12.13);ax.set_ylim(57.699,57.78)
    ax.set_xlabel('Longitude (°E)',labelpad=8);ax.set_ylabel('Latitude (°N)',labelpad=8)
    scale(ax,11.903,57.768,2,'2 km')
    ax.annotate('N',xy=(12.12,57.773),xytext=(12.12,57.764),ha='center',fontsize=11,arrowprops=dict(arrowstyle='-|>',color=dark),color=dark)
    ax.text(.015,1.018,'A  Göteborg–Partille',transform=ax.transAxes,fontsize=12,fontweight='bold',color=dark)
    # Zoom shows known OSM stop positions. The uncertain service platform is an
    # area, not an invented coordinate. ET_R is deliberately absent.
    _, ways = osm_context(SOURCES/'osm_eketragatan_context_20260921.osm')
    for tags, pts in ways:
        xx,yy=zip(*pts)
        if 'building' in tags:
            detail.fill(xx,yy,fc='#e8edf0',ec='#dce3e8',lw=.4,zorder=0)
        elif 'highway' in tags or 'railway' in tags:
            detail.plot(xx,yy,color='#c8d4dc',lw=1.2 if tags.get('highway') not in ['footway','path','cycleway'] else .55,zorder=1)
    terminal_ids=[286385228,663680768,663680772,663680775,663682636,1753201873,1753201875,1753201876]
    terminal=[nodes[('node',n)] for n in terminal_ids]
    xmin=min(e['lon'] for e in terminal)-.00015;xmax=max(e['lon'] for e in terminal)+.00015
    ymin=min(e['lat'] for e in terminal)-.00012;ymax=max(e['lat'] for e in terminal)+.00012
    detail.add_patch(Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,facecolor=blue,alpha=.065,edgecolor=blue,lw=1.4,linestyle='--',zorder=2))
    for e in terminal:
        detail.scatter(e['lon'],e['lat'],s=24,color='#7e919c',edgecolor='white',lw=.5,zorder=4)
        detail.annotate(e['tags'].get('ref',''),(e['lon'],e['lat']),xytext=(4,0),textcoords='offset points',fontsize=8,color='#607783',zorder=5)
    detail.scatter(*A,s=120,marker='s',color=blue,edgecolors='white',lw=1.4,zorder=8)
    detail.annotate('2190L\nStop L / charger proxy',A,xytext=(0,15),textcoords='offset points',ha='center',fontsize=10.5,color=dark,bbox=box,zorder=9)
    detail.annotate('2190\nPassenger terminal\n(platform unresolved)',(11.91025,57.7162),xytext=(0,-73),textcoords='offset points',ha='center',va='top',fontsize=10.5,color=dark,arrowprops=dict(arrowstyle='-',color=blue,lw=1),bbox=box,zorder=9)
    detail.set_xlim(11.9088,11.9118);detail.set_ylim(57.7146,57.7182)
    detail.set_xticks([11.909,11.910,11.911]);detail.set_yticks([57.715,57.716,57.717,57.718]);detail.ticklabel_format(style='plain',useOffset=False)
    detail.set_xlabel('Longitude (°E)',labelpad=8)
    scale(detail,11.9090,57.71783,.05,'50 m')
    detail.text(.015,1.018,'B  Eketrägatan detail',transform=detail.transAxes,fontsize=12,fontweight='bold',color=dark)
    fig.legend(handles=[Line2D([0],[0],color=blue,lw=2.8,label='Passenger-service endpoints'),Line2D([0],[0],color=orange,lw=2.2,ls='--',label='Model empty driving'),Line2D([0],[0],color=orange,marker='D',mfc='white',lw=0,ms=8,label='Depot address proxy')],loc='lower center',bbox_to_anchor=(.50,.045),ncol=3,frameon=False,fontsize=10.5)
    fig.text(.5,.021,'Connectors are not road paths.  |  Map data © OpenStreetMap contributors (ODbL); depot address geocode: Hitta.',ha='center',fontsize=8.7,color='#536a78')
    for ext in ('png','pdf','svg'):
        fig.savefig(HERE/f'k5_geographic_context.{ext}',dpi=210,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    # Optional network overview: the three already geocoded other chargers are
    # muted context; no new bus movement or site eligibility is asserted.
    fig,ax=plt.subplots(figsize=(13.3,6.8))
    fig.subplots_adjust(left=.065,right=.985,top=.96,bottom=.16)
    map_axes(ax)
    for e in json.loads((SOURCES/'osm_roads_rivers_20260910.json').read_text())['elements']:
        xy=e.get('geometry',[])
        if xy:
            water='waterway' in e.get('tags',{})
            ax.plot([p['lon'] for p in xy],[p['lat'] for p in xy],color='#a4cddb' if water else '#d7dfe4',lw=3 if water else .8,zorder=0)
    ax.add_patch(FancyArrowPatch(A,B,arrowstyle='<->',mutation_scale=12,color=blue,lw=2.5,connectionstyle='arc3,rad=-.10',zorder=3,shrinkA=9,shrinkB=9))
    ax.add_patch(FancyArrowPatch(B,D,arrowstyle='<->',mutation_scale=11,color=orange,lw=2,linestyle=(0,(5,3)),zorder=3,shrinkA=9,shrinkB=9))
    ax.add_patch(FancyArrowPatch(A,D,arrowstyle='->',mutation_scale=11,color=orange,lw=2,linestyle=(0,(5,3)),connectionstyle='arc3,rad=.12',zorder=3,shrinkA=9,shrinkB=9))
    for point,marker,color,label,offset in [(A,'s',blue,'Eketrägatan\n2190L · 1 charger',(-5,-23)),(B,'s',blue,'Merkuriusgatan\n4808 · 1 charger',(-8,14)),(D,'D',orange,'PARX · 60 kW\nAddress proxy',(17,-27))]:
        ax.scatter(*point,s=115,marker=marker,color=color if marker=='s' else 'white',edgecolors='white' if marker=='s' else color,lw=1.6,zorder=7)
        ax.annotate(label,point,xytext=offset,textcoords='offset points',ha='center',va='top' if offset[1]<0 else 'bottom',fontsize=10.5,color=dark,bbox=box,zorder=8)
    offsets={'3127L':(0,-30),'7880C':(12,-20),'JON_A':(3,16)}
    for r in context:
        xy=(r['longitude'],r['latitude'])
        ax.scatter(*xy,s=90,marker='s',color='#8b9ba6',edgecolor='white',lw=1.4,zorder=6)
        ax.annotate(f"{r['name']}\n{r['code']} · {r['ports']} charger"+('s' if r['ports']>1 else ''),xy,xytext=offsets[r['code']],textcoords='offset points',ha='center',va='top' if offsets[r['code']][1]<0 else 'bottom',fontsize=10.2,color='#637581',bbox=box,zorder=7)
    ax.text(11.992,57.755,'Passenger service\nGIRO 53–63 min\nfee0 51–63 min',ha='center',fontsize=10.5,color=blue,bbox=box,zorder=8)
    ax.text(12.109,57.755,'7 min\n7.8 kWh',ha='center',fontsize=10.3,color=orange,bbox=box,zorder=8)
    ax.text(11.983,57.724,'19 min\n28.0 kWh',ha='center',fontsize=10.3,color=orange,bbox=box,zorder=8)
    ax.set_xlim(11.89,12.21);ax.set_ylim(57.689,57.78)
    ax.set_xlabel('Longitude (°E)',labelpad=8);ax.set_ylabel('Latitude (°N)',labelpad=8)
    scale(ax,11.903,57.770,2,'2 km')
    ax.annotate('N',xy=(12.195,57.773),xytext=(12.195,57.763),ha='center',fontsize=11,arrowprops=dict(arrowstyle='-|>',color=dark),color=dark)
    fig.legend(handles=[Line2D([0],[0],marker='s',color='white',markerfacecolor=blue,ms=9,label='Selected buses: opportunity charging'),Line2D([0],[0],marker='s',color='white',markerfacecolor='#8b9ba6',ms=9,label='Other network chargers'),Line2D([0],[0],color=orange,lw=2,ls='--',label='Model empty driving')],loc='lower center',bbox_to_anchor=(.50,.045),ncol=3,frameon=False,fontsize=10)
    fig.text(.5,.025,'Stop/address proxies; connectors are not road paths.  |  Map data © OpenStreetMap contributors (ODbL); depot geocode: Hitta.',ha='center',fontsize=8.5,color='#536a78')
    for ext in ('png','pdf','svg'):
        fig.savefig(HERE/f'k5_charger_network_context.{ext}',dpi=210,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    hashes={str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='geography_manifest.json' and 'doc_verification' not in p.parts and '__pycache__' not in p.parts and p.name!='osm_depot_context_20260921.osm'}
    (HERE/'geography_manifest.json').write_text(json.dumps({'schema':'geographic_context_v1','created':'2026-09-21','coordinate_crs':'WGS84 EPSG:4326','display_projection':'latitude/longitude with aspect corrected at 57.73 N','coordinate_scope':'current stop/address proxies, not surveyed historical charger/depot positions','movement_scope':'exact selected buses in saved matched k5 figure; connectors are endpoint links, not road paths','unknown_locations_not_plotted':['ET_R','exact 2190 passenger platform'],'sha256':hashes},indent=2,ensure_ascii=False)+'\n')
    print('Wrote map PNG/PDF/SVG, coordinates.csv and geography_manifest.json')

if __name__=='__main__':
    main()
