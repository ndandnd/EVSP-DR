#!/usr/bin/env python3
"""Native cache and fee smoke; produces no research comparison result."""
import argparse,copy,importlib.util,json,os,subprocess
from pathlib import Path
root=Path('/home/nc437/ladder-lite/zero_charge_start_fee_20260913')
spec=importlib.util.spec_from_file_location('fee_campaign',root/'tooling/campaign.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
m=c.read(root/'manifest.json');v=root/'validation';v.mkdir(exist_ok=True)
original=c.read(root/'manifest.json');large=next(x for x in m['pairs'] if x['id']=='w3_k15_fee0')
out=v/'large_cache_fee0';out.mkdir(exist_ok=False)
cmd=c.cg_command(m,large,'fee0',out)+['--event-network-cache-only']
out_index=cmd.index('--out');del cmd[out_index:out_index+2]
wrapper="""import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1]); destination=Path(sys.argv[2]); sys.argv=[sys.argv[0]]+sys.argv[3:]
import exact_pricer_expanded as module
original=module.run_cg
def capture(args):
 result=original(args)
 with destination.open('x') as stream: json.dump(result,stream); stream.flush(); __import__('os').fsync(stream.fileno())
 return result
module.run_cg=capture
raise SystemExit(module.main())
"""
cmd=[m['python'],'-u','-c',wrapper,str(Path(m['code'])/'src'),str(out/'cg.json')]+cmd[3:]
run=c.run_process(cmd,m['code'],out/'execution',1200,c.clean_environment(),{'cpus':8,'mem':'96G'})
assert run['returncode']==0 and not run['watchdog_triggered'],run
cache=c.read(out/'cg.json');audit=cache['network_metrics']['cache_charge_start_reprice']
assert cache['cache_hit'] and audit['current']==0 and audit['repriced_arcs']>0,cache['network_metrics']
src=m['inputs'][large['case_id']]
assert c.digest(src['cache'])==src['cache_sha256'] and c.digest(src['cache']+'.manifest.json')==src['cache_manifest_sha256']
m['root']=str(v);m['common'].update(cg_seconds=300,mip_seconds=45,stage1_seconds=15,inherit_time_limit_s=120)
for arm in m['arms'].values():arm['inherit_max_columns']=20
m['pairs']=[dict(m['pairs'][0],id='native_fee_smoke',order=['fee5','fee0'])]
c.atomic_write(v/'manifest.json',m,exclusive=True)
rc=c.worker(argparse.Namespace(root=v,index=0));assert rc==0,rc
status=next((v/'cases/native_fee_smoke').glob('*/pair_status.json'));s=c.read(status)
checks=[]
for fee in [5,0]:
 arm='fee'+str(fee);p=status.parent/arm;cg=c.read(p/'cg.json');mip=c.read(p/'mip/result.json');metrics=c.charging_metrics(mip,float(fee))
 assert cg['charge_start_cost']==fee,cg.get('charge_start_cost')
 assert mip['physics']['charge_start_cost']==fee,mip.get('physics')
 assert metrics.get('cost_components_reconcile') is True,metrics
 checks.append({'arm':arm,'cg_status_sha256':c.digest(p/'cg.json'),'mip_status_sha256':c.digest(p/'mip/result.json'),'metrics':metrics})
c.atomic_write(v/'PASS.json',{'validated_utc':c.now(),'commit':original['cg_commit'],'large_cache_audit':audit,'source_caches_unchanged':True,'checks':checks},exclusive=True)
print('NATIVE_FEE_SMOKE_PASS',flush=True)
