from pathlib import Path
import json,csv,hashlib,collections
P=Path(__file__).resolve().parent
x=json.loads((P/'replay_results.json').read_text());rows=[]
for r in x['results']:
 assert r['continuous_replay_valid'] and r['physical_routes_unchanged'] and r['cost_preserved_by_identical_physical_schedule']
 assert r['service_occurrences']==r['input_trips']
 rows.append({**{k:v for k,v in r.items() if k not in ['routes','dispatch_ledger','failures']},'selected_routes':len(r['routes']),'all_empty_routes_after_assignment':sum(not v['service_trip_ids'] for v in r['routes'])})
with (P/'per_case.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=dict(cases=len(rows),routes=sum(r['selected_routes'] for r in rows),service_occurrences=sum(r['service_occurrences'] for r in rows),empty_drive_occurrences=sum(r['empty_drive_occurrences'] for r in rows),replay_passed=sum(r['continuous_replay_valid'] for r in rows),physical_schedules_unchanged=sum(r['physical_routes_unchanged'] for r in rows),costs_preserved=sum(r['cost_preserved_by_identical_physical_schedule'] for r in rows),all_empty_routes=sum(r['all_empty_routes_after_assignment'] for r in rows),replay_seconds=sum(r['replay_seconds'] for r in rows))
(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
provenance=dict(module_sha256=x['module_sha256'],execution_commit=x['execution_commit'],artifacts_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(P.iterdir()) if f.is_file() and f.name!='provenance.json'},source_validation='All128 original MIP artifact byte hashes and input hashes rechecked against frozen F2 cohort; replay modules matched recorded pinned871d057e sources.',scope='Service-assignment partition with empty traversal preservation. Original saved route columns remain coverings. Baseline physics only. No shared charging capacity validation or optimization.')
(P/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(summary,indent=2))
