#!/usr/bin/env python3
"""Emit compact hash-verified construction/MIP records for root integration."""
import argparse,hashlib,json
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main(root):
 root=Path(root).resolve();m=read(root/'manifest.json');mh=sha(root/'manifest.json');records=[];mips=[];constructions=[];errors=[]
 for cid,c in m['cases'].items():
  cp=root/'cases'/cid/'completion.json'
  if not cp.exists():continue
  try:
   v=read(cp)
   if v['manifest_sha256']!=mh or v['case_id']!=cid or v['kind']!=c['kind'] or v['status']!='finished' or not v['usable']:raise ValueError('completion identity/state')
   if sha(v['result_path'])!=v['result_sha256']:raise ValueError('result hash')
   result=read(v['result_path'])
   if c['kind']=='pool_construction':
    if result['artifact_kind']!='parent_mapped_partition_pool' or result['optimization_run'] is not False or result['certified_rc_optimal'] is not False:raise ValueError('construction scope')
    if sha(v['journal_path'])!=v['journal_sha256'] or sha(v['construction_path'])!=v['construction_sha256']:raise ValueError('construction artifact hash')
    compact={'case_id':cid,'partition':c['partition'],'artifact_kind':result['artifact_kind'],'optimization_run':False,'certified':False,'columns':result['final']['pool_columns'],'component_audits':result['pool_construction']['component_audits'],'mandatory_route_count':result['pool_construction']['mandatory_route_count'],'mandatory_covers_every_parent_trip':result['pool_construction']['mandatory_covers_every_parent_trip'],'parent_graph_constructed':False};constructions.append(compact);phase='pool_construction'
   else:
    if result.get('source_result_sha256')!=v['source_status_sha256'] or result.get('source_journal_sha256')!=v['source_journal_sha256'] or not result.get('physical_replay_validated'):raise ValueError('MIP source/physical gate')
    compact={'case_id':cid,'source_partitions':c['source_partitions'],'is_validation':bool(c.get('is_validation')),'buses':result.get('buses'),'fleet_proven':result.get('fleet_proven'),'status_name':result.get('status_name'),'stage1':(result.get('two_stage') or {}).get('stage1_status_name'),'stage2':(result.get('two_stage') or {}).get('stage2_status_name'),'physical_replay_validated':True,'source_status_sha256':v['source_status_sha256'],'source_journal_sha256':v['source_journal_sha256'],'best_known_source_upper_bound':v['best_known_source_upper_bound'],'finite_pool_only':True,'cg_certificate':False,'full_model_lp_bound':False};mips.append(compact);phase='mip'
   records.append({'phase':phase,'case_id':cid,'path':v['result_path'],'sha256':v['result_sha256'],'result':compact})
  except Exception as e:errors.append({'case_id':cid,'error':repr(e)})
 out={'schema':'evsp-decomposition-pool-union-collection-v1','root':str(root),'manifest_sha256':mh,'cg':[],'mip':mips,'construction':constructions,'records':records,'errors':errors,'workflow':{'manifest':m,'jobs':read(root/'jobs.json') if (root/'jobs.json').exists() else {},'attempt_progress':{}}}
 print(json.dumps(out,indent=2,allow_nan=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
