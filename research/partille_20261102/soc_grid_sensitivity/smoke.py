"""Exercise the true capacity CLI on both SOC grids on the same four-row pilot fixture."""
import argparse,csv,hashlib,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--receipt',required=True,type=Path);a=p.parse_args(); repo=a.repo.resolve(); base=repo/'data';out=a.out.resolve();out.mkdir(exist_ok=False);data=out/'data';data.mkdir()
for name in ('Ref_dict.csv','par_ref_dhd.csv','hourly_prices_flat.csv'): (data/name).symlink_to(base/name)
original=base/'scale_ladder/instances/nested_probability_k2_15_20260908/Practice_Custom_DutyUnion_k08_p02_20260908.csv'
with original.open() as f, (data/'smoke.csv').open('w') as g:
    table=csv.DictReader(f);w=csv.DictWriter(g,fieldnames=table.fieldnames);w.writeheader(); w.writerows(list(table)[:4])
smoke={}
for step in (2.5,5.):
    results=out/f'grid{step}';args=[sys.executable,'-u','run_capacity_terminal_cg.py','--data-dir',str(data),'--csv','smoke.csv','--prices',str(data/'hourly_prices_flat.csv'),'--target','0','--fleet-cap','12','--cg-seconds','90','--dive-seconds','8','--mip-seconds','12','--charge-kw','240','--battery-kwh','240','--start-fee','5','--non-strict-tariff','--station-kw','PARX=60','--soc-step',str(step),'--out',str(results)]
    with (out/f'log_{step}').open('w') as log:subprocess.run(args,cwd=repo/'src',stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',SLURM_CPUS_PER_TASK='1'))
    result=json.loads((results/'summary.json').read_text());selected=json.loads((results/'selected_routes.json').read_text());
    if result['physics']['soc_step']!=step or not result['cg_pricing_certified'] or not selected: raise ValueError('incomplete smoke')
    smoke[str(step)]={k:result.get(k) for k in ['packed_arcs','capacity_alternatives','capacity_window_keys','columns','cg_pricing_certified','selection_state','selected_plug_violations']}
    smoke[str(step)]['physics']=result['physics'];smoke[str(step)]['selected_records']=len(selected)
if smoke['5.0']['packed_arcs']>=smoke['2.5']['packed_arcs']:raise ValueError('same graph size or coarse graph larger')
a.receipt.write_text(json.dumps(smoke,indent=2)+'\n')
