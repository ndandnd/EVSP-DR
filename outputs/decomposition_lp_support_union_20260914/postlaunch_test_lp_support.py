"""Regression for support-only retention and forced-union cap overflow."""
import csv,hashlib,json,tempfile
from pathlib import Path
import pool_logic
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):Path(p).write_text(json.dumps(v)+'\n')
fields=['Identifier','From1','Start1','End1','To1','Distance1','Usage kWh','count_trip_id','Ordered_Trip_ID']
def csvout(p,rows):
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
with tempfile.TemporaryDirectory() as d:
 root=Path(d);parent_rows=[];sources=[]
 for component in range(4):
  child_rows=[]
  for local in range(3):
   real=component*3+local;row={'Identifier':'Regular','From1':'A','Start1':f'{5+real}:00','End1':f'{5+real}:10','To1':'B','Distance1':'1','Usage kWh':'2','count_trip_id':str(real),'Ordered_Trip_ID':str(100+real)};parent_rows.append(row);cr=dict(row);cr['count_trip_id']=str(local);child_rows.append(cr)
  child=root/f'c{component}.csv';csvout(child,child_rows)
  records=[]
  for trips in ([0],[1],[2],[0,1],[0,2],[1,2]):records.append({'trips':trips,'route_nodes':['D',*trips,'D'],'cost':100000.0,'master_cost_semantics':'expanded_grid_cost','expanded_grid_cost':100000.0,'charging_stops':{}})
  journal=root/f'c{component}.jsonl';journal.write_text(''.join(json.dumps(x)+'\n' for x in records))
  positives=[{'trips':x,'value':.5,'cost':100000.0} for x in ([0,1],[0,2],[1,2])]
  status={'csv':child.name,'trip_ids':[0,1,2],'g_kwh':240,'charge_kw':240,'min_soc_frac':0,'soc_step':2.5,'block_min':5,'time_model':'event','master_sense':'cover','prices_csv':'hourly_prices_flat.csv','columns_journal':str(journal),'certified_rc_optimal':True,'stop_reason':'certified','final_lp':{'positive_routes':positives,'objective':150000.0,'route_weight':1.5,'artificial_total':0.0},'provenance':{'git_commit':pool_logic.SOURCE_COMMIT,'instance_sha256':sha(child),'prices_sha256':'p','reference_sha256':'r','deadhead_sha256':'d'}};sp=root/f'c{component}.status';put(sp,status)
  mip={'source_result_sha256':sha(sp),'source_journal_sha256':sha(journal),'selected_routes':records[:3],'physical_replay_validated':True,'physical_pool_audit':{'rejected_columns':0,'deterministically_repaired':0}};mp=root/f'c{component}.mip';put(mp,mip)
  sources.append({'case':f'c{component}','component':component,'input_path':str(child),'input_sha256':sha(child),'status_path':str(sp),'status_sha256':sha(sp),'journal_path':str(journal),'journal_sha256':sha(journal),'mip_path':str(mp),'mip_sha256':sha(mp),'certified':True,'source_stop_reason':'certified'})
 parent=root/'parent.csv';csvout(parent,parent_rows)
 spec={'partition':1,'parent_csv':'parent.csv','parent_input_path':str(parent),'parent_input_sha256':sha(parent),'data_dir':str(root),'model_hashes':{'prices_sha256':'p','reference_sha256':'r','deadhead_sha256':'d'},'per_component_cap':1,'sources':sources};out=root/'pool.json';result=pool_logic.construct_partition(spec,out)
 assert result['columns']==24
 assert all(x['integer_witness']==3 and x['lp_support']==3 and x['forced_union']==6 and x['selected']==6 and x['forced_union_exceeded_cap'] for x in result['component_audits'])
 assert result['mandatory_route_count']==12 and result['mandatory_covers_every_parent_trip']
 assert all(x['lp_validation']['remapped_minimum_coverage']>=1 for x in result['source_audits'])
 print(json.dumps({'status':'passed','selected_columns':24,'integer_warm_routes':12,'support_only_routes_retained':12,'forced_union_over_cap_exercised':True}))
