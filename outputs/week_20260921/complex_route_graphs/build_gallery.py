#!/usr/bin/env python3
"""Native editable itinerary and interactive SVG companion, without invented trips."""
from pathlib import Path
import csv,json,html,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
D=json.loads((HERE/'schedules.json').read_text());S=next(s for s in D['schedules'] if s['duty_id']=='13309')
N={x['area']:x['name'] for x in csv.DictReader((HERE/'coordinates.csv').open())}
N['7581']='Gamlestads Torg'
def esc(x):return html.escape(str(x))
def clock(x):
 sec=round(float(x)*60);h=sec//3600;m=sec%3600//60;s=sec%60
 return f'{h:02}:{m:02}' if not s else f'{h:02}:{m:02}:{s:02}'
def span(e):return clock(e['start_min'])+'–'+clock(e['end_min'])
def write_csv(name,rows):
 with (HERE/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
leg_counter=move_counter=charge_counter=0
labels={};legrows=[];charge=[];detail={}
for e in S['events']:
 if e['kind']=='service':leg_counter+=1;lab=f'L{leg_counter}'
 elif e['kind']=='deadhead' and e['from_ref']!=e['to_ref']:move_counter+=1;lab=f'M{move_counter}'
 elif e['kind']=='charge':charge_counter+=1;lab=f'C{charge_counter}';labels[e['event_index']]=lab;charge.append(dict(charge=lab,site=e['from_code'],area=N[e['from_ref']],start=clock(e['start_min']),end=clock(e['end_min']),kwh=e['energy_charged_kwh']));continue
 else:continue
 labels[e['event_index']]=lab
 row=dict(step=len(legrows)+1,diagram_label=lab,prepared_trip_id=e['source_trip_id'] if e['kind']=='service' else '',kind='Passenger service' if e['kind']=='service' else 'Empty movement',from_area=N[e['from_ref']],from_code=e['from_code'],to_area=N[e['to_ref']],to_code=e['to_code'],departure=clock(e['start_min']),arrival=clock(e['end_min']),duration_minutes=e['duration_min'],recorded_usage_kwh=e['energy_consumed_kwh'],source_pointer=e.get('source_pointer',''))
 legrows.append(row);detail[lab]=f"{lab} · "+(f"Trip {e['source_trip_id']} · " if e['kind']=='service' else '')+f"{N[e['from_ref']]} ({e['from_code']}) → {N[e['to_ref']]} ({e['to_code']}) · {span(e)} · {e['duration_min']:g} min"
vis=[]
for v in S['visits']:
 activity=[]
 for i in v['event_indices']:
  e=S['events'][i];lab=labels.get(i,'');activity.append(f"{lab+' ' if lab else ''}{e['kind'].replace('_',' ')} {e['from_code']}"+(f"→{e['to_code']}" if e['from_code']!=e['to_code'] else '')+f" {span(e)}"+(f" +{e['energy_charged_kwh']:.3f}kWh" if e['kind']=='charge' else ''))
 vis.append(dict(visit=f"V{v['visit_index']+1}",area=N[v['area']],raw_codes=', '.join(v['raw_codes']),arrival=clock(v['arrival_min']),departure=clock(v['departure_min']),activities='; '.join(activity) or 'Immediate departure'))
evrows=[]
for e in S['events']:
 evrows.append(dict(event=f"E{e['event_index']+1}",diagram_label=labels.get(e['event_index'],''),kind=e['kind'],start=clock(e['start_min']),end=clock(e['end_min']),from_code=e['from_code'],to_code=e['to_code'],prepared_trip_id=e['source_trip_id'] if e['source_trip_id'] is not None else '',recorded_recharge_kwh=e['energy_charged_kwh'] if e['kind']=='charge' else '',recorded_usage_kwh='' if e['energy_consumed_kwh'] is None else e['energy_consumed_kwh'],source_pointer=e.get('source_pointer',''),timing_basis=e['timing_basis']))
write_csv('diagram_leg_key.csv',legrows);write_csv('diagram_visit_key.csv',vis);write_csv('diagram_event_key.csv',evrows);write_csv('diagram_charging_key.csv',charge)
# Native SVG: labels and paths have browser tooltips and keyboard/click detail.
svg=(HERE/'duty_13309_graph.svg').read_text();svg=svg[svg.index('<svg'):];root=ET.fromstring(svg)
for g in root.iter():
 gid=g.attrib.get('id','');lab=None
 if gid.startswith('leg-label-'):lab='L'+gid.removeprefix('leg-label-')
 elif gid.startswith('leg-'):lab='L'+gid.removeprefix('leg-')
 elif gid.startswith('move-'):lab='M'+gid.removeprefix('move-')
 if lab and lab in detail:
  g.attrib.update({'data-leg':lab,'tabindex':'0','role':'button','aria-label':detail[lab]});title=ET.Element('{http://www.w3.org/2000/svg}title');title.text=detail[lab];g.insert(0,title)
ET.register_namespace('','http://www.w3.org/2000/svg');svg=ET.tostring(root,encoding='unicode')
(HERE/'duty_13309_interactive.svg').write_text(svg)
def table(headers,rows,ids=None):
 h='<table contenteditable="true"><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'
 for i,r in enumerate(rows):h+='<tr'+(f' id="row-{esc(ids[i])}"' if ids else '')+'>'+''.join('<td>'+esc(x)+'</td>' for x in r)+'</tr>'
 return h+'</tbody></table>'
legtable=table(['Order','Graph','Trip label','From → to','Departure','Arrival','min'],[[x['step'],x['diagram_label'],x['prepared_trip_id'],f"{x['from_area']} ({x['from_code']}) → {x['to_area']} ({x['to_code']})",x['departure'],x['arrival'],x['duration_minutes']] for x in legrows],[x['diagram_label'] for x in legrows])
visittable=table(['Visit','Area / platforms','Arrival','Departure','Activities within this visit'],[[x['visit'],x['area']+' · '+x['raw_codes'],x['arrival'],x['departure'],x['activities']] for x in vis])
eventtable=table(['Event','Graph','Kind','From → to','Start','End','Trip label','Recharge kWh','Source row'],[[x['event'],x['diagram_label'],x['kind'].replace('_',' '),x['from_code']+' → '+x['to_code'],x['start'],x['end'],x['prepared_trip_id'],x['recorded_recharge_kwh'],x['source_pointer']] for x in evrows])
chargetable=table(['Charge','Site','Start','End','Recorded kWh'],[[x['charge'],x['site'],x['start'],x['end'],f"{x['kwh']:.6f}"] for x in charge])
head='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GIRO duty 13309 · five-place bus day</title><style>body{max-width:1450px;margin:2rem auto;padding:0 1.5rem;font:16px/1.55 system-ui;color:#253d4e}h1{font-size:28px;margin-bottom:.3em}h2{font-size:22px;margin-top:2rem}p{max-width:1100px}a{color:#246b9a}svg,img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.4;margin:1em 0}td,th{border:1px solid #d8e0e6;padding:8px;text-align:left}th{background:#eef3f6}tr:nth-child(even){background:#f8fafb}td:first-child{white-space:nowrap}.detail{position:sticky;top:0;background:#fff8f1;border:1px solid #ecd2b8;padding:12px;z-index:4;border-radius:6px}.selected{background:#fff0d9!important;outline:2px solid #cb711f}g[data-leg]{cursor:pointer}g[data-leg]:hover path{stroke:#cb711f!important;stroke-width:2.5!important}.small{font-size:13px;color:#627784}details{margin:1em 0}summary{cursor:pointer;font-weight:bold}@media print{.detail{position:static}details{display:block}table{font-size:9px}tr{break-inside:avoid}}</style><body>'''
body='<h1>A recorded bus day through five places</h1><p>Duty <strong>13309</strong> serves 22 passenger trips and charges at Heden, Jons väg and the depot. The same approximate spatial layout appears twice, with the depot displaced into clear space, split around the real midday depot visit. It returns at <strong>10:42</strong>, charges at <strong>10:45–12:30 (+105 kWh)</strong>, and leaves again at <strong>12:43</strong>.</p><p><a href="duty_13309_graph.pdf">Vector graph PDF</a> · <a href="duty_13309_graph.png">Full-resolution PNG</a> · <a href="duty_13309_daybook.pdf">Graph + full itinerary PDF</a> · <a href="diagram_leg_key.csv">Editable leg CSV</a> · <a href="README.md">Sources and scope</a></p>'
body+=svg+'<div id="selection" class="detail" aria-live="polite">Select a numbered service arc (L) or empty-movement arc (M) to see its exact trip label, endpoints and clocks.</div><p class="small">L1–L22 are passenger-leg order; M1–M5 are inter-area empty-move order; C1–C4 are charges; V1–V28 are area-visit order. Trip numbers use stable labels from our prepared input, not GIRO-supplied journey numbers. Orange rings mark sites used for charging somewhere during this day, not necessarily in that panel.</p>'
body+='<h2>Clocked leg key</h2><p>The table is in chronological order across the full day. Its platform codes are the recorded codes; graph nodes group platforms only through the documented reference-area mapping.</p>'+legtable
body+='<h2>Charging</h2>'+chargetable+'<p>Recharge is copied from recorded GIRO cells and checked against this duty’s documented 18E2 capacity (239.01 kWh). These are not recalculated 18E1 values or newly optimized charging choices.</p>'
body+='<details open><summary>All numbered area visits</summary>'+visittable+'</details><details><summary>All 60 events, including preparation and platform gaps</summary>'+eventtable+'</details>'
body+='<h2>Scope and geographic sources</h2><p>This is a recorded GIRO schedule, not a fresh optimized solution. Same-area gaps sometimes have no separate platform-movement record; the itinerary preserves that uncertainty instead of inventing travel times or energy. Curved edges show activity connections, not road paths. The four passenger-area positions preserve approximate geographic spacing from documented proxies. PARX is displaced into the lower-right empty space so passenger arcs do not appear to pass through the depot; the actual coordinates are retained in coordinates.csv.</p><ul>'
for x in csv.DictReader((HERE/'coordinates.csv').open()):body+='<li>'+esc(x['name'])+': <a href="'+esc(x['source'])+'">source</a> — '+esc(x['coordinate_scope'])+'</li>'
body+='</ul><p>Coordinates contain © OpenStreetMap contributors data. The exact historical PARX gate/charger is not established; its position is an operator-address proxy. <a href="DATA_README.md">Extraction notes</a> and <a href="validation.json">checks</a> retain source workbook rows and hashes. The secondary extracted duty13320 includes unlocated area13722 and is not geographically plotted.</p><details><summary>Optional full-day multigraph</summary><img src="duty_13309_full_day.png" alt="All22 service arcs in a single five-node diagram"><p>This compact overview has more overlapping arcs; the split graph is the primary readable view.</p></details><p class="small">Tables are editable in this local HTML page for copying into a document. Local edits do not change the source CSV/JSON or solve a new schedule. GitHub displays HTML source; download this folder and open gallery.html for the interactive view.</p>'
js='<script>const details='+json.dumps(detail,ensure_ascii=False).replace('</',r'<\/')+''';function selectLeg(k){document.getElementById('selection').textContent=details[k];document.querySelectorAll('.selected').forEach(x=>x.classList.remove('selected'));const r=document.getElementById('row-'+k);if(r)r.classList.add('selected');}document.querySelectorAll('[data-leg]').forEach(g=>{g.addEventListener('click',()=>selectLeg(g.dataset.leg));g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectLeg(g.dataset.leg)}})});</script>'''
(HERE/'gallery.html').write_text(head+body+js+'</body></html>')
(HERE/'native_itinerary_fragment.html').write_text('<h2>GIRO duty13309: full-day leg key</h2>'+legtable+'<h2>Charging</h2>'+chargetable)
print(f'Wrote native gallery and keys: {len(legrows)} movements, {len(vis)} visits, {len(evrows)} events')
