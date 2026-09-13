#!/usr/bin/env python3
"""Repeat only MIP after the native test exposed a logging-helper mismatch."""
import argparse,importlib.util,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--code',required=True,type=Path);p.add_argument('--commit',required=True);a=p.parse_args()
root=Path('/home/nc437/ladder-lite/zero_charge_start_fee_20260913');v=root/'validation'
spec=importlib.util.spec_from_file_location('fee_campaign',root/'tooling/campaign.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
manifest=c.read(root/'manifest.json');c.code_check(a.code,a.commit)
assert manifest['cg_commit']==a.commit and manifest['mip_commit']==a.commit
previous='666cd839ab3923071b1a571ef349b70f61a82fa9'
# Reuse the completed CG smoke only if all pricing/realization code is unchanged.
for rel in ['src/exact_pricer_expanded.py','src/event_pricer_network.py','src/expanded_path_realization.py','src/run_exact_pool_mip.py']:
 assert subprocess.check_output(['git','-C',str(a.code),'rev-parse',previous+':'+rel],text=True)==subprocess.check_output(['git','-C',str(a.code),'rev-parse',a.commit+':'+rel],text=True),rel
old_attempt=v/'cases/native_fee_smoke/110952_r0'
old=c.read(old_attempt/'pair_status.json');assert all(x.get('cg_execution',{}).get('returncode')==0 for x in old['arms'].values()),old['status']
checks=[];newroot=v/'mip_retry'/os.environ['SLURM_JOB_ID'];newroot.mkdir(parents=True,exist_ok=False)
for fee in [5,0]:
 arm='fee'+str(fee);source=old_attempt/arm;dest=newroot/arm;dest.mkdir()
 cg=c.read(source/'cg.json');assert cg['charge_start_cost']==fee and cg['final']['artificials']==0 and cg['final']['iter']>0
 journal=Path(cg['columns_journal']);env=c.clean_environment({'EVSP_EXPECTED_COMMIT':a.commit,'EVSP_REQUIRE_DETACHED':'1','EVSP_MIP_EXPECTED_RESULT_SHA256':c.digest(source/'cg.json'),'EVSP_MIP_EXPECTED_JOURNAL_SHA256':c.digest(journal)})
 m=json.loads(json.dumps(manifest));m['common'].update(mip_seconds=45,stage1_seconds=15)
 command=c.mip_command(m,dest,arm);i=command.index('--result');command[i+1]=str(source/'cg.json')
 execution=c.run_process(command,a.code,dest/'mip',645,env,{'cpus':8,'mem':'96G'})
 assert execution['returncode']==0 and not execution['watchdog_triggered'],execution
 result=c.read(dest/'mip/result.json');assert result['physics']['charge_start_cost']==fee
 assert result['physical_replay_validated'] is True
 metrics=c.charging_metrics(result,fee);assert metrics.get('cost_components_reconcile') is True,metrics
 checks.append({'arm':arm,'cg_status_path':str(source/'cg.json'),'cg_status_sha256':c.digest(source/'cg.json'),'mip_status_path':str(dest/'mip/result.json'),'mip_status_sha256':c.digest(dest/'mip/result.json'),'metrics':metrics})
cache=c.read(v/'large_cache_fee0/cg.json');audit=cache['network_metrics']['cache_charge_start_reprice'];assert cache['cache_hit'] and audit['current']==0 and audit['repriced_arcs']>0
s=manifest['inputs']['w3_k15'];assert c.digest(s['cache'])==s['cache_sha256'] and c.digest(s['cache']+'.manifest.json')==s['cache_manifest_sha256']
c.atomic_write(v/'PASS.json',{'validated_utc':c.now(),'commit':a.commit,'cg_smoke_commit':previous,'cg_pricing_code_unchanged':True,'large_cache_audit':audit,'source_caches_unchanged':True,'checks':checks,'prior_attempt':'110952 retained: MIP logging helper failed before optimization; CG outputs reused'},exclusive=True)
print('NATIVE_FEE_MIP_RETRY_PASS',flush=True)
