from pathlib import Path
import sys,json,hashlib
P=Path(__file__).resolve().parent;ROOT=P.parents[2];CODE=ROOT/'.codex-work/review-strict-chain-20260916/src';sys.path.insert(0,str(CODE))
from audit_giro_duty_recovery import fixed_sequence_recovery,charger_overlap_audit
from audit_giro_known_columns import build_problem
from giro_partille_physics import PARTILLE_PROFILES
problem=build_problem(P/'inputs',str(P/'inputs/Practice_Custom_DutyUnion_original_eligible_k05_20260908.csv'),max_station_to_trip_wait_min=1560)
I=ROOT/'outputs/meeting_20260917/route_explainer/inputs'
original=json.loads((I/'original.json').read_text())['routes'];zero=json.loads((I/'cg_selected_peak08.json').read_text());five=json.loads((I/'fee5_peak08_comparison.json').read_text())['joint_pool_optimized']['selected_routes']
result={}
for label,routes in [('original',original),('saved_joint_fee0',zero),('saved_joint_fee5',five)]:
 out=[]
 for i,r in enumerate(routes):
  rr=fixed_sequence_recovery(problem,tuple(r['trips']),PARTILLE_PROFILES['18E1'],horizon_min=1560);rr.update(source_index=i,trips=r['trips']);out.append(rr)
 result[label]=dict(routes=out,all_individually_recovered=all(r['feasible'] for r in out),capacity_audit=charger_overlap_audit(out),scope='individual fixed tripsequence, documented18E1 nonlinearphysics, static referenceDHD, maximalearlycharging heuristic; no tariffoptimization,no platform/FIFO,no equalterminalconstraint')
(P/'saved_sequence_replay.json').write_text(json.dumps(result,indent=2))
for k,v in result.items():print(k,v['all_individually_recovered'],[(x['source_index'],x['feasible'],x.get('reason')) for x in v['routes']],v['capacity_audit'])
