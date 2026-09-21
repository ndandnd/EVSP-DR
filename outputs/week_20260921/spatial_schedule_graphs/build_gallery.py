#!/usr/bin/env python3
"""Build a browsable local gallery and native editable leg/charging tables."""
from pathlib import Path
import csv,html,json,hashlib
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent

def clock(x):
    sec=round(float(x)*60);h=sec//3600;m=sec%3600//60;s=sec%60
    return f'{h:02}:{m:02}' if not s else f'{h:02}:{m:02}:{s:02}'

def esc(x):return html.escape(str(x))

def table(headers,rows):
    return '<table contenteditable="true"><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'

def main():
    d=json.loads((HERE/'schedules.json').read_text());parts=[];summary=[];keys=[]
    for pair in sorted(d['pairs'],key=lambda p:p['visual_priority']):
        name=pair['pair_id'];duty=pair['original_duty'];group=sorted([s for s in d['schedules'] if s['pair_id']==name],key=lambda s:['original','fee0','fee5'].index(s['arm']))
        starts=' / '.join(str(s['metrics']['starts']) for s in group);trips=' / '.join(str(s['metrics']['trip_count']) for s in group)
        summary.append([duty,trips,starts,f"{pair['fee0_trip_overlap']} / {pair['fee5_trip_overlap']}",pair['visual_interest_reason']])
        parts.append(f'<section id="{name}"><h2>Original duty {duty} and paired schedules</h2><p class="metrics">Trips: {trips} · Charge starts: {starts} (original / fee 0 / fee 5)</p><p>{esc(pair["visual_interest_reason"])}</p><a href="{name}_graph.svg"><img src="{name}_graph.png" alt="Spatial graphs for original duty {duty} and paired fee schedules"></a><p class="links"><a href="{name}_graph.pdf">Spatial PDF</a> · <a href="{name}_graph.svg">Vector SVG</a> · <a href="{name}_itinerary.png">Complete itinerary image</a> · <a href="{name}_itinerary.pdf">Complete itinerary PDF</a></p>')
        parts.append('<details><summary>Open editable leg and charging keys</summary><p>These native HTML tables can be selected, edited and copied. Edits are local to this browser view; original CSV/JSON files remain unchanged.</p><div class="keys">')
        for s in group:
            label={'original':'Original GIRO','fee0':'Reoptimized · fee 0','fee5':'Reoptimized · fee 5'}[s['arm']]
            services=sorted([e for e in s['events'] if e['kind']=='service'],key=lambda e:e['start_min']);charges=sorted([e for e in s['events'] if e['kind']=='charge'],key=lambda e:e['start_min'])
            legs=[[f'L{i}',e['source_trip_id'],clock(e['start_min']),clock(e['end_min']),f"{e['duration_min']:.0f}",f"{e['energy_consumed_kwh']:.2f}"] for i,e in enumerate(services,1)]
            ch=[[f'C{i}',e['from_code'],clock(e['start_min']),clock(e['end_min']),f"{e['energy_charged_kwh']:.2f}"] for i,e in enumerate(charges,1)]
            for i,e in enumerate(services,1):keys.append(dict(pair_id=name,schedule_id=s['schedule_id'],arm=s['arm'],leg=f'L{i}',prepared_trip_id=e['source_trip_id'],from_code=e['from_code'],to_code=e['to_code'],departure=clock(e['start_min']),arrival=clock(e['end_min']),start_min=e['start_min'],end_min=e['end_min'],duration_min=e['duration_min'],energy_kwh=e['energy_consumed_kwh']))
            parts.append('<article><h3>'+label+'</h3><h4>Passenger legs</h4>'+table(['Leg','Trip','Depart','Arrive','min','kWh'],legs)+'<h4>Charging windows</h4>'+table(['Charge','Site','Start','End','kWh'],ch)+'</article>')
        parts.append('</div></details><details><summary>Open complete visit-expanded graph</summary><img class="trace" src="'+name+'_itinerary.png" alt="Every timed visit, service, movement, charge, wait and preparation interval"></details></section>')
    css='''body{font:16px/1.45 system-ui,sans-serif;color:#253d4e;margin:0;background:#f5f7f9}main{max-width:1500px;margin:auto;padding:28px}h1{font-size:30px;margin:8px 0}h2{font-size:24px}h3{font-size:19px}h4{margin:15px 0 6px}nav{display:flex;gap:18px;flex-wrap:wrap;background:#fff;padding:14px;position:sticky;top:0;z-index:2;border-bottom:1px solid #dae1e7}a{color:#216b9b}section{background:white;padding:24px;margin:26px 0;border:1px solid #dae1e7;border-radius:9px}img{width:100%;height:auto}img.trace{max-width:1700px}.metrics{color:#647786}.scope{background:#fff5e9;border-left:4px solid #cb711f;padding:16px}.keys{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}table{border-collapse:collapse;width:100%;font-size:13px;background:white;margin:10px 0}td,th{text-align:left;padding:7px;border-bottom:1px solid #d8e0e6;vertical-align:top}th{background:#edf3f7}summary{cursor:pointer;font-weight:600;padding:12px 0}.links{font-size:14px}@media(max-width:1100px){.keys{grid-template-columns:1fr}}@media print{nav{position:static}section{break-before:page}details{display:block}main{padding:0}body{background:white}}'''
    intro='''<h1>Five bus-day comparisons</h1><p>Each spatial graph keeps locations fixed and shows directed passenger legs, depot departure/return and every charging window. Open the itinerary for the complete chronological path, including waiting, preparation, local transfers and ET_R rest visits.</p><p class="scope"><strong>Comparison scope:</strong> these are the existing matched-physics recharging results on saved trip sequences, not new column-generation runs. Paired bus labels follow an optimal but nonunique overlap mapping, not a physical vehicle identity. Every fee-0/fee-5 pair has different passenger assignments, so differences are not fee-only causal effects. Only original duty13414 and its fee-5 counterpart share the same twelve passenger trips.</p><p><strong>Read the labels:</strong> L1, L2, … mean passenger-leg order. Trip numbers use stable labels from our prepared input, not GIRO-supplied journey numbers. C1, C2, … mean charging-session order; V1, V2, … in the itinerary mean visit order. Time labels are rounded to the nearest second; CSV/JSON retains full precision.</p><p>The spatial overview groups the Eketrägatan stop2190 and charger2190L. The itinerary separates them and retains the original1-minute transfer versus the model0-minute transfer. ET_R is an unlocated auxiliary node; its drawn position is schematic. Reoptimized empty-driving clocks follow a feasible reconstruction convention, not uniquely optimized departure choices.</p><p><a href="all_comparisons.pdf">Download all five comparisons and itineraries (PDF)</a> · <a href="comparison_summary.csv">Comparison table CSV</a> · <a href="leg_key.csv">Editable leg-key CSV</a> · <a href="events.csv">All751 events CSV</a> · <a href="README.md">Sources and scope</a> · <a href="ladder_path_dependence.md">Why the six k40 chains can differ</a></p>'''
    nav='<nav>'+''.join(f'<a href="#{p["pair_id"]}">Duty {p["original_duty"]}</a>' for p in sorted(d['pairs'],key=lambda p:p['visual_priority']))+'</nav>'
    doc='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EVSP–DR spatial schedule comparisons</title><style>'+css+'</style></head><body>'+nav+'<main>'+intro+table(['Duty','Trips O/0/5','Starts O/0/5','Shared original trips 0/5','What to compare'],summary)+''.join(parts)+'</main></body></html>'
    (HERE/'gallery.html').write_text(doc)
    with (HERE/'comparison_summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['original_duty','trips_original_fee0_fee5','starts_original_fee0_fee5','original_trip_overlap_fee0_fee5','visual_interest']);w.writerows(summary)
    with (HERE/'leg_key.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(keys[0]));w.writeheader();w.writerows(keys)
    thumbs=[]
    try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',27)
    except OSError:font=ImageFont.load_default()
    for p in sorted(d['pairs'],key=lambda p:p['visual_priority']):
        im=Image.open(HERE/(p['pair_id']+'_graph.png')).convert('RGB');im.resize((1500,round(im.height*1500/im.width)))
        im=im.resize((1500,round(im.height*1500/im.width)));tile=Image.new('RGB',(1540,im.height+66),'white');draw=ImageDraw.Draw(tile);draw.text((28,10),'Original duty '+p['original_duty']+' · charge starts '+str(p['original_starts'])+' / '+str(p['fee0_starts'])+' / '+str(p['fee5_starts']),font=font,fill='#253d4e');tile.paste(im,(20,53));thumbs.append(tile)
    sheet=Image.new('RGB',(1540,sum(t.height for t in thumbs)), '#e8edf1');y=0
    for im in thumbs:sheet.paste(im,(0,y));y+=im.height
    sheet.save(HERE/'contact_sheet.png')
    print('Built gallery HTML, editable tables and contact sheet')

if __name__=='__main__':main()
