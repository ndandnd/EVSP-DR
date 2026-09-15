"""Frozen previous-k core and deterministic native-incidence filler; no CG."""
import math

def choose(parent,mip,journal,cap=512):
 allowed=set(parent['trip_ids'])
 def clean(r):
  ts=r['trips'];cost=float(r['cost'])
  assert ts and all(type(t) is int for t in ts) and len(set(ts))==len(ts) and set(ts)<=allowed and math.isfinite(cost)
  return {'trips':list(ts),'cost':cost}
 assert mip['physical_replay_validated'] is True
 assert len(mip['selected_routes'])==mip['buses']>0
 assert parent['final_lp']['artificial_total']==0
 core={};coverage={t:0. for t in allowed};integer_covered=set()
 for r in mip['selected_routes']:
  integer_covered.update(clean(r)['trips'])
 assert integer_covered==allowed
 positive=[]
 for r in parent['final_lp']['positive_routes']:
  value=float(r['value']);assert math.isfinite(value) and value>=0
  if value>0:
   rr=clean(r);positive.append(rr)
   for t in rr['trips']:coverage[t]+=value
 assert min(coverage.values())>=1-1e-6
 for r in list(mip['selected_routes'])+positive:
  rr=clean(r);key=tuple(sorted(rr['trips']))
  if key in core:
   assert core[key]['trips']==rr['trips'],'Conflicting mandatory ordered paths for native incidence'
   if rr['cost']<core[key]['cost']-1e-9:core[key]=rr
  else:core[key]=rr
 winners={}
 for r in journal:
  rr=clean(r);key=tuple(sorted(rr['trips']))
  if key not in winners or rr['cost']<winners[key]['cost']-1e-9:winners[key]=rr
 for key,rr in core.items():
  assert key in winners
  assert winners[key]['trips']==rr['trips'] and abs(winners[key]['cost']-rr['cost'])<=1e-6,'Core differs from native full-journal winner'
 ordered=sorted(core.values(),key=lambda r:tuple(r['trips']))
 filler=sorted((r for key,r in winners.items() if key not in core),key=lambda r:(-len(r['trips']),r['cost']/len(r['trips']),tuple(r['trips'])))
 expanded=ordered+filler[:max(0,cap-len(ordered))]
 def audit(rs):return dict(selected_sequence_count=len(rs),selected_unique_tripsets=len({frozenset(r['trips']) for r in rs}),distinct_parent_trips=len({t for r in rs for t in r['trips']}))
 return {'core':ordered,'core512':expanded},dict(core_count=len(core),integer_routes=len(mip['selected_routes']),positive_lp_routes=len(positive),minimum_source_lp_coverage=min(coverage.values()),native_journal_tripsets=len(winners),cap=cap,core_exceeds_cap=len(core)>cap,filler_rank='length descending, cost/trip ascending, ordered local sequence ascending; native cheapest incidence first; ties retain first journal record',arms={a:audit(rs) for a,rs in [('core',ordered),('core512',expanded)]})

def status(parent,journal,arm,audit):
 return dict(artifact_kind='previous_instance_sequence_subset',optimization_run=False,certified_rc_optimal=False,csv=parent['csv'],trip_ids=parent['trip_ids'],columns_journal=str(journal),columns=audit['selected_sequence_count'],provenance={'instance_sha256':parent['provenance']['instance_sha256']},treatment=arm,seed_construction=audit)
