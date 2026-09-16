"""Read-only inventory of repeated service trips in saved k32 selections."""
from pathlib import Path
from collections import Counter
import json,hashlib,csv,datetime
H=Path('/home/nc437/ladder-lite');E=H/'chain_extension_31_32_20260915';manifest=json.loads((E/'manifest.json').read_text())
sources={1:E/'cases/w1_k32/mip_result.json',2:H/'continuation_gaps14_20260916/cases/w2_k32_longmip/mip_result.json',3:H/'continuation_gaps12_20260916/cases/w3_k32_longmip/mip_result.json',4:H/'continuation_gaps14_20260916/cases/w4_k32_longmip/mip_result.json',5:H/'continuation_gaps12_20260916/cases/w5_k32_longmip/mip_result.json',6:H/'continuation_gaps13_20260916/cases/w6_k32_longmip/mip_result.json'}
rows=[]
for chain,p in sources.items():
 raw=p.read_bytes();r=json.loads(raw);routes=r['selected_routes'];assert len(routes)==r['buses']
 c=manifest['cases'][f'w{chain}_k32'];f=E/'code/data'/c['csv'];h=hashlib.sha256(f.read_bytes()).hexdigest();assert h==c['input_sha256']==r['physical_pool_audit']['input_hashes']['instance_sha256']
 with f.open() as stream: n=len(list(csv.DictReader(stream)))
 assert all(len(route['trips'])==len(set(route['trips'])) for route in routes)
 counts=Counter(t for route in routes for t in route['trips']);assert len(counts)==n
 per_route=[sum(counts[t]>1 for t in route['trips']) for route in routes]
 rows.append(dict(chain=chain,target=32,buses=r['buses'],input_sha256=h,input_rows=n,unique_covered_trips=len(counts),repeated_trip_ids=sum(v>1 for v in counts.values()),extra_service_occurrences=sum(v-1 for v in counts.values()),max_multiplicity=max(counts.values()),per_route_repeated_trip_counts=per_route,max_repeated_trips_per_route=max(per_route),routes_exceeding_existing_cleanup_limit10=sum(v>10 for v in per_route),naive_subset_candidates=sum(2**v for v in per_route),selected_route_set_sha256=r['selected_route_set_sha256'],selected_charging_block_set_sha256=r['selected_charging_block_set_sha256'],source_journal_sha256=r['source_journal_sha256'],physics=r['physics'],source=str(p),source_sha256=hashlib.sha256(raw).hexdigest(),recorded_duplicate_cleanup_validated=r['duplicate_trip_removal_validated'],recorded_shared_capacity_validated=r['cross_route_charger_capacity_validated']))
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Read-only coverage inventory, not a cleanup, feasibility replay, or optimality proof.',rows=rows),indent=2))
