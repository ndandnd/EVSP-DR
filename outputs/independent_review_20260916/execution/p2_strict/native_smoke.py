import subprocess,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent;code=root/'code';out=root/'smoke/native';out.mkdir(parents=True,exist_ok=False)
manifest=json.load(open(root/'manifest.json'));commit=manifest['execution_commit']
common=[sys.executable,str(code/'src/run_capacity_speed_event_cg.py'),'--mode','cg','--arm','parx60','--prices',str(code/'data/hourly_prices_flat.csv'),'--reference-data-dir',str(code/'data'),'--battery-kwh','236.44','--reserve-kwh','35.466','--cg-wall-s','30','--threads','1','--max-station-wait-min','1560','--expected-commit',commit,'--require-clean']
commands=[]
for n,prefix in [(1,'parent'),(2,'child')]:
 cmd=common+['--instance',str(root/f'smoke/trips{n}.csv'),'--out',str(out/f'{prefix}.json'),'--pool-out',str(out/f'{prefix}.pool.jsonl')]
 if n==2:cmd+=['--inherit-status',str(out/'parent.json'),'--inherit-pool',str(out/'parent.pool.jsonl'),'--inherit-instance',str(root/'smoke/trips1.csv')]
 commands.append(cmd)
 with open(out/f'{prefix}.log','w') as f:subprocess.run(cmd,cwd=code,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=150)
ss=[json.load(open(out/f'{n}.json')) for n in ['parent','child']]
assert all(s['certified_rc_optimal'] for s in ss) and ss[1]['inheritance']['inherited_columns']==1
(out/'verification.json').write_text(json.dumps({'commit':commit,'both_certified':True,'child_inherited_columns':1,'commands':commands,'provenance':ss[1]['provenance']},indent=2))
