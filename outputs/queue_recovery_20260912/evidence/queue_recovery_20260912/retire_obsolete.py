from pathlib import Path
import json,subprocess,datetime
S='/usr/local/slurm/slurm-25.05.5/bin/';Q=Path('/home/nc437/ladder-lite/queue_recovery_20260912');F=Path('/home/nc437/ladder-lite/full_pool_recovery_20260912');G=Path('/home/nc437/ladder-lite/graph_recovery_20260912')
f=json.loads((F/'case_jobs.json').read_text());g=json.loads((G/'case_jobs.json').read_text());mapping={}
for old,cid,stage in [('810296','w1_k08','cg'),('810319','w1_k09','cg'),('810320','w1_k10','cg'),('810952','w1_k07','mip'),('810953','w1_k07','mip'),('810954','w1_k08','mip'),('810955','w1_k08','mip'),('810956','w1_k09','mip'),('810957','w1_k09','mip'),('810958','w1_k10','mip'),('810959','w1_k10','mip'),('810975','w2_k09','mip'),('810994','w4_k10','mip'),('810995','w4_k10','mip')]:mapping[old]={'replacement':f[cid][stage],'case_id':cid,'reason':'Retire stalled original full-pool submission; indexed full-pool recovery now scheduled. Original failure retained; bounded512 experiment remains separate.'}
for old,cid,stage in [('949749','join00','cg'),('949750','join00','mip'),('949752','join01','mip'),('949754','join02','mip'),('949756','join03','mip'),('949757','join04','cg'),('949758','join04','mip'),('949759','join05','cg'),('949760','join05','mip'),('949761','join06','cg'),('949763','join06','mip'),('949764','join07','cg'),('949765','join07','mip'),('949767','join08','mip'),('949769','join09','mip'),('949624_3','d00_g3','mip')]:mapping[old]={'replacement':g[cid][stage],'case_id':cid,'reason':'Retire dead prerequisite chain; shared-cache recovery scheduled with separately recorded extended cache budget and selected component incumbent initialization.'}
records=[]
for old,v in mapping.items():
 st=subprocess.check_output([S+'scontrol','show','job',old,'-o'],text=True);new=subprocess.check_output([S+'scontrol','show','job',v['replacement'],'-o'],text=True)
 assert 'JobState=PENDING' in st and 'JobHeldUser' not in st and '537227' not in old
 assert any(x in st for x in ['overnight_extension_20260912','nested_warm_multichain_p1246_k2_10_20260910_ecb60c1'])
 assert any(x in new for x in ['full_pool_recovery_20260912','graph_recovery_20260912'])
 r=subprocess.run([S+'scancel',old],capture_output=True,text=True);records.append({'old_job_id':old,**v,'original_scontrol':st,'replacement_scontrol':new,'returncode':r.returncode,'stderr':r.stderr})
 (Q/'retired_queue_entries.json').write_text(json.dumps({'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records},indent=2)+'\n')
print('Retired',len(records),'obsolete pending entries; source files unchanged')
