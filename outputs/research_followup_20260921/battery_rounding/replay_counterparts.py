"""No-solver supplemental replay of all C6k5 historical fee0/5 routes.
Reuses only the definitions/input lookup prefix of replay.py, not its main run.
"""
from pathlib import Path
import json,hashlib,csv
P=Path(__file__).resolve().parent
ns={'__file__':str(P/'replay.py')}
exec((P/'replay.py').read_text().split('source_meta=')[0],ns)
arc=ns['arc'];ROOT=ns['ROOT']
f=ROOT/'outputs/research_followup_20260921/duty13309/sources/campaign_snapshot.json';d=json.loads(f.read_text());tr=d['inputs']['w6_k05']['rows'];out=[]
for arm,picked in [('w6_k05_fee0',3),('w6_k05_fee5',2)]:
 sol=d['arms'][arm]['files']['result.json']['content'];assert sol['physics']['g_kwh']==240 and sol['physics']['charge_kw']==240
 for j,r in enumerate(sol['selected_routes']):
  e=240.;values=[e];si=0;st=r['charging_stops']
  for pos,(prev,node) in enumerate(zip(r['route_nodes'],r['route_nodes'][1:]),1):
   left=tr[prev]['To1'] if isinstance(prev,int) else prev;right=tr[node]['From1'] if isinstance(node,int) else node
   e-=arc(left,right)[1];values.append(e)
   if isinstance(node,int):e-=float(tr[node]['Usage kWh']);values.append(e)
   elif pos<len(r['route_nodes'])-1 and si<len(st['stations']) and node==st['stations'][si]:
    e+=st['kwh'][si];values.append(e);si+=1
  assert si==len(st['stations']) and abs(e-r['physical_realization']['continuous_terminal_soc_kwh'])<1e-5
  assert min(values)>=-1e-5 and max(values)<=240+1e-5
  for cap in [240.,239.01,236.44]:
   out.append(dict(arm=arm,route_index=j,duty13309_counterpart=j==picked,battery_kwh=cap,min_soc_kwh=min(values)+cap-240,terminal_kwh=e+cap-240,frozen_energy_valid=min(values)+cap-240>=-1e-5,scope='capacity-only, preserved zero reserve/no terminal floor, recorded charge kWh fixed'))
with (P/'duty13309_counterparts.csv').open('w') as o:
 w=csv.DictWriter(o,list(out[0]));w.writeheader();w.writerows(out)
(P/'counterpart_provenance.json').write_text(json.dumps(dict(source=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),output_sha256=hashlib.sha256((P/'duty13309_counterparts.csv').read_bytes()).hexdigest()),indent=2))
print(json.dumps([r for r in out if r['duty13309_counterpart']],indent=2))
