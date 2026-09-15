#!/usr/bin/env python3
"""Cluster-native parser, license, physics-propagation, and result-binding gate."""
from __future__ import annotations
import argparse,datetime as dt,json,os,subprocess
from pathlib import Path
import campaign

def replace_option(command:list[str],name:str,value:str)->None:
 i=command.index(name); command[i+1]=value

def run(command,*,cwd,env,stdout,stderr,timeout):
 with stdout.open('x') as out, stderr.open('x') as err:
  cp=subprocess.run(command,cwd=cwd,env=env,stdout=out,stderr=err,timeout=timeout)
 if cp.returncode: raise RuntimeError(f"native command failed {cp.returncode}: {command}")

def check_cg(root:Path,manifest:dict,case:dict,code:Path)->dict:
 out=root/'native_smoke'/os.environ.get('SLURM_JOB_ID','local')/case['case_id']; out.mkdir(parents=True,exist_ok=False)
 commands=campaign.build_commands(manifest,code,out,case,python='/home/nc437/evsp_env/bin/python')
 cg=list(commands['cg']); replace_option(cg,'--max-iters','0'); replace_option(cg,'--cg-wall-s','300')
 run(cg,cwd=code,env=campaign.execution_environment(),stdout=out/'cg.stdout.log',stderr=out/'cg.stderr.log',timeout=420)
 status=json.loads((out/'cg.json').read_text()); physics=status['physics']; prov=status['provenance']
 expected_capacity=case['arm'] in {'capacity','combined'}
 expected_parx=60.0 if case['arm'] in {'parx60','combined'} else 240.0
 assert status['arm']==case['arm'] and status['capacity_selector']=='prefix-memo'
 assert physics['battery_kwh']==236.44 and physics['initial_soc_kwh']==236.44
 assert physics['reserve_kwh']==35.466 and physics['terminal_soc_constraint']=='reserve_only'
 assert physics['parx_kw']==expected_parx and physics['non_parx_kw']==240.0
 assert physics['capacity_enforced'] is expected_capacity
 assert physics['charger_counts']==(manifest['common_model']['charger_counts'] if expected_capacity else {})
 assert prov['git_commit']==manifest['code']['commit'] and prov['git_tracked_dirty'] is False
 inputs=manifest['source_inputs']; assert prov['instance_sha256']==inputs[case['instance']]['sha256']
 assert prov['prices_sha256']==inputs['flat']['sha256']
 assert prov['reference_sha256']==inputs['reference_dictionary']['sha256']
 assert prov['deadhead_sha256']==inputs['deadhead_reference']['sha256']
 assert status['pool_sha256']==campaign.sha256_file(out/'pool.jsonl')
 return {'case_id':case['case_id'],'output_dir':str(out),'cg_path':str(out/'cg.json'),
   'cg_sha256':campaign.sha256_file(out/'cg.json'),'pool_path':str(out/'pool.jsonl'),
   'pool_sha256':status['pool_sha256'],'physics':physics,'provenance':prov,
   'checkpoint':status['checkpoint'],'commands':{'cg':cg,'mip':commands['mip']}}

def main(root:Path):
 manifest_path=root/'manifest.json'; manifest=campaign.load_manifest(manifest_path); code=root/'code'
 campaign.validate_tooling(manifest,root)
 validation=campaign.validate_checkout(code,campaign.validate_manifest(manifest,code))
 if not validation['valid']: raise RuntimeError(json.dumps(validation,sort_keys=True))
 parsed=campaign.validate_driver_commands(manifest,code)
 if not parsed['valid'] or parsed['parsed_command_count']!=20: raise RuntimeError(json.dumps(parsed,sort_keys=True))
 cases={c['case_id']:c for c in manifest['cases']}
 smoke=[]
 for cid in ['k1_13408_flat_236p44r15_baseline','k1_13408_flat_236p44r15_combined']:
  smoke.append(check_cg(root,manifest,cases[cid],code))
 # Exercise the dedicated finite-pool MIP and bind it to the combined CG artifact.
 combined=smoke[1]; cmd=list(combined['commands']['mip']); replace_option(cmd,'--mip-wall-s','30')
 out=Path(combined['output_dir'])
 run(cmd,cwd=code,env=campaign.execution_environment(),stdout=out/'mip.stdout.log',stderr=out/'mip.stderr.log',timeout=150)
 mip=json.loads((out/'mip.json').read_text())
 assert mip['cg_status_sha256']==combined['cg_sha256']
 assert mip['pool_sha256']==combined['pool_sha256']
 assert mip['pool_acceptance']['usable'] is True and mip['capacity_enforced_in_mip'] is True
 assert mip['physical_station_capacity_audit']['valid'] is True
 assert mip['duplicate_service_audit']['all_trips_covered'] is True
 assert mip['duplicate_service_audit']['extra_trip_assignments'] == 0
 result={'schema':'evsp-dr-reserve-feasibility-native-validation-v1','status':'passed',
  'recorded_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'job_id':os.environ.get('SLURM_JOB_ID'),
  'manifest_sha256':campaign.sha256_file(manifest_path),'code_commit':manifest['code']['commit'],
  'parsed_command_count':parsed['parsed_command_count'],'smoke':smoke,
  'combined_mip':{'path':str(out/'mip.json'),'sha256':campaign.sha256_file(out/'mip.json'),
   'cg_status_sha256':mip['cg_status_sha256'],'pool_sha256':mip['pool_sha256'],
   'pool_acceptance':mip['pool_acceptance'],'result':mip['result'],
   'physical_station_capacity_audit':mip['physical_station_capacity_audit'],
   'duplicate_service_audit':mip['duplicate_service_audit']},
  'scope':'Validation-only max-iters-zero singleton pools and a 30-second finite-pool MIP; no research result or CG certificate.'}
 campaign.atomic_json(root/'native_validation.json',result); print(json.dumps(result,sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root.resolve())
