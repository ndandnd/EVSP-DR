import csv,hashlib,json,tempfile,unittest
from pathlib import Path
import pool_logic

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):Path(p).write_text(json.dumps(v)+'\n')
FIELDS=['Identifier','From1','Start1','End1','To1','Distance1','Usage kWh','count_trip_id','Ordered_Trip_ID']
def csvfile(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)

class CampaignTest(unittest.TestCase):
 def fixture(self,root):
  rows=[]
  for i in range(4):rows.append({'Identifier':'Regular','From1':'A','Start1':f'{5+i}:00','End1':f'{5+i}:10','To1':'B','Distance1':'1','Usage kWh':'2','count_trip_id':str(i),'Ordered_Trip_ID':str(100+i)})
  parent=root/'parent.csv';csvfile(parent,rows);sources=[]
  for i,row in enumerate(rows):
   child=root/f'c{i}.csv';r=dict(row);r['count_trip_id']='0';csvfile(child,[r])
   record={'trips':[0],'route_nodes':['PARX_0',0,'PARX_0'],'cost':100000.0,'master_cost_semantics':'expanded_grid_cost','expanded_grid_cost':100000.0,'charging_stops':{},'continuous_realization':{'schema':'x','trip_sequence_sha256':'old','route_nodes_sha256':'old','mapping_sha256':'old'},'physical_realization':{'realization_mapping_sha256':'old'}}
   journal=root/f'c{i}.jsonl';journal.write_text(json.dumps(record)+'\n')
   status={'csv':child.name,'trip_ids':[0],'g_kwh':240,'charge_kw':240,'min_soc_frac':0,'soc_step':2.5,'block_min':5,'time_model':'event','master_sense':'cover','prices_csv':'hourly_prices_flat.csv','columns_journal':str(journal),'certified_rc_optimal':True,'stop_reason':'certified','provenance':{'git_commit':pool_logic.SOURCE_COMMIT,'instance_sha256':sha(child),'prices_sha256':'p','reference_sha256':'r','deadhead_sha256':'d'}}
   sp=root/f'c{i}.status';put(sp,status)
   mip={'source_result_sha256':sha(sp),'source_journal_sha256':sha(journal),'selected_routes':[record],'physical_replay_validated':True,'physical_pool_audit':{'rejected_columns':0,'deterministically_repaired':0}}
   mp=root/f'c{i}.mip';put(mp,mip)
   sources.append({'case':f'c{i}','component':i,'input_path':str(child),'input_sha256':sha(child),'status_path':str(sp),'status_sha256':sha(sp),'journal_path':str(journal),'journal_sha256':sha(journal),'mip_path':str(mp),'mip_sha256':sha(mp),'certified':True,'source_stop_reason':'certified'})
  return {'partition':1,'parent_csv':'parent.csv','parent_input_path':str(parent),'parent_input_sha256':sha(parent),'data_dir':str(root),'model_hashes':{'prices_sha256':'p','reference_sha256':'r','deadhead_sha256':'d'},'per_component_cap':1,'sources':sources}
 def test_remap_full_payload_and_union(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);spec=self.fixture(root);out=root/'pool.json';c=pool_logic.construct_partition(spec,out)
   self.assertEqual(c['columns'],4);self.assertTrue(c['mandatory_covers_every_parent_trip']);self.assertFalse(c['parent_graph_constructed'])
   records=[json.loads(x) for x in Path(str(out)+'.columns.jsonl').read_text().splitlines()]
   self.assertEqual([r['trips'][0] for r in records],[0,1,2,3]);self.assertTrue(all(r['continuous_realization']['mapping_sha256']!='old' for r in records))
   ps=read= json.loads(out.read_text());us={'partitions':['p01','p01'],'sources':[{'pool_path':str(out),'pool_sha256':sha(out),'journal_path':ps['columns_journal'],'journal_sha256':sha(ps['columns_journal'])}]*2};u=root/'union.json';pool_logic.union_pools(us,u)
   self.assertEqual(json.loads(u.read_text())['final']['pool_columns'],4)
 def test_child_attribute_mismatch_fails(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);spec=self.fixture(root);p=Path(spec['sources'][0]['input_path']);rows=list(csv.DictReader(p.open()));rows[0]['Usage kWh']='3';csvfile(p,rows);spec['sources'][0]['input_sha256']=sha(p)
   status=json.loads(Path(spec['sources'][0]['status_path']).read_text());status['provenance']['instance_sha256']=sha(p);put(spec['sources'][0]['status_path'],status);spec['sources'][0]['status_sha256']=sha(spec['sources'][0]['status_path']);m=json.loads(Path(spec['sources'][0]['mip_path']).read_text());m['source_result_sha256']=spec['sources'][0]['status_sha256'];put(spec['sources'][0]['mip_path'],m);spec['sources'][0]['mip_sha256']=sha(spec['sources'][0]['mip_path'])
   with self.assertRaisesRegex(ValueError,'physical trip attribute mismatch'):pool_logic.construct_partition(spec,root/'bad.json')

if __name__=='__main__':unittest.main()
