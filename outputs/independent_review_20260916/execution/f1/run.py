from pathlib import Path
import csv,json,hashlib,subprocess,shlex
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
f2=json.loads((P.parent/'f2/selected_trip_sets.json').read_text())
base={r['case_id']:r for r in csv.DictReader((ROOT/'outputs/overnight_next_20260914/status_20260916T194843Z/all_chain_extension_results.csv').open())}
req=[]
for r in f2['results']:
 cid=r['case_id'].replace('_longmip','');b=base[cid]
 inp=ROOT/'outputs/chain_extension_20260913/inputs'/f'{cid}.csv';mapping={x['count_trip_id']:int(x['Ordered_Trip_ID']) for x in csv.DictReader(inp.open())}
 req.append(dict(cohort=r['cohort'],case_id=cid,path=r['path'],sha256=r['sha256'],input_sha256=r['input_sha256'],cg_path=b['cg_path'],target=int(b['target_buses']),ordered_ids=mapping))
(P/'requests.json').write_text(json.dumps(req))
cmd='OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/nc437/evsp_env/bin/python -c '+shlex.quote((P/'replay_remote.py').read_text())
with (P/'replay_results.json').open('w') as out,(P/'replay_progress.log').open('w') as log:
 subprocess.run(['ssh','-S','/Users/nadan/.ssh/evsp-unicorn.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','nc437@unicorn-login-01.coecis.cornell.edu',cmd],input=json.dumps(req),text=True,stdout=out,stderr=log,check=True)
