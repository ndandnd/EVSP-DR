from pathlib import Path
import json,subprocess,hashlib,shutil
Q=Path('/home/nc437/ladder-lite/queue_recovery_20260912');B=Path('/home/nc437/ladder-lite/graph_recovery_20260912');O=Path('/home/nc437/ladder-lite/overnight_extension_20260912');C=B/'code';pin=(Q/'graph_execution_pin.txt').read_text().strip();B.mkdir(exist_ok=True);(B/'logs').mkdir(exist_ok=True);(B/'cache').mkdir(exist_ok=True)
if not C.exists():subprocess.run(['git','clone','--shared','--no-checkout','/home/nc437/ladder-lite/full_pool_recovery_20260912/code',str(C)],capture_output=True,check=True)
subprocess.run(['git','-C',str(C),'fetch',str(Q/'evsp-queue-graph.bundle'),'HEAD'],capture_output=True,check=True);subprocess.run(['git','-C',str(C),'checkout','--detach',pin],capture_output=True,check=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((O/'manifest.json').read_text());cases={}
for cid,c in old['cases'].items():
 if not(cid.startswith('d') or cid.startswith('join')):continue
 src=O/'code/data'/c['csv'];dst=C/'data'/c['csv'];dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():shutil.copy2(src,dst)
 assert sha(dst)==c['input_sha256']
 if cid=='d00_g3' or cid.startswith('join'):cases[cid]=dict(c)
for name in ['hourly_prices_flat.csv','Ref_dict.csv','par_ref_dhd.csv']:
 src=O/'code/data'/name;dst=C/'data'/name
 if not dst.exists():shutil.copy2(src,dst)
 assert sha(src)==sha(dst)
v={'schema':'evsp-shared-graph-recovery-v1','execution_commit':pin,'mip_execution_commit':'871d057e1067411f09581e37d78f7c1ca43f68bb','original_campaign':str(O),'cases':cases,'cache_seconds':43200,'cache_builds':{'d00_g3':{'csv':cases['d00_g3']['csv'],'cache':str(B/'cache/d00_g3.pkl'),'mem':'32G'},'parent32':{'csv':cases['join00']['csv'],'cache':str(B/'cache/parent32.pkl'),'mem':'128G'}},'scope':'Extended-budget recovery. New graph build budget up to12h is recorded separately from CG:4h for d00_g3,2h for each parent. Each parent starts from the four completed component integer solutions, replaying all distinct selected sequences, then searches for cross-group routes. No GIRO route seed, no shared capacity, same 240/240 event2.5/5 covering flat physics. This selected-incumbent initialization differs from the original512-sequence parent treatment and is a separately labeled experiment.','resources':'All default_partition, exclude scaglione-compute-01. Independent cache builds both launched; same parent cache reused by all10decompositions. MIPs3600s stage1<=1800 stage2 fleet<=validated incumbent.','borrowed_object_store':'/home/nc437/ladder-lite/full_pool_recovery_20260912/code/.git/objects'}
(B/'manifest.json').write_text(json.dumps(v,indent=2)+'\n');print('Prepared two caches and eleven CG cases')
