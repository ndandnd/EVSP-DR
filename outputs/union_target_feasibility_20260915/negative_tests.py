"""Run actual native fail-closed paths without optimization or source mutation."""
from pathlib import Path
import copy,importlib.util,os,sys
import common as w
from target_solver import identity_gate,patched_source,model_digest
ROOT=Path(__file__).resolve().parent
m=w.read(ROOT/'manifest.json');c=m['cases']['validation_feasible'];native=Path(c['source_code'])/'src/run_exact_pool_mip.py'
sys.path.insert(0,str(native.parent));spec=importlib.util.spec_from_file_location('frozen_native',native);n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
rows=[]
def rejected(name,fn,contains):
 try:fn()
 except (ValueError,SystemExit) as e:
  assert contains in str(e),(name,str(e));rows.append({'test':name,'passed':True,'error':str(e)});return
 raise AssertionError(name+' did not reject')
private=ROOT/'negative_fixtures';private.mkdir(exist_ok=True)
status=w.read(c['source_result']);changed=private/'changed_instance.csv';changed.write_bytes(Path(c['input_path']).read_bytes()+b'\n');bad=copy.deepcopy(status);bad['csv']=str(changed)
rejected('changed_input_native_physical_gate',lambda:n.prepare_strict_partition_pool(bad,[],data_dir=c['data_dir'],reference_data_dir=c['data_dir']),'input hash mismatch')
os.environ.update(EVSP_EXPECTED_COMMIT=c['execution_commit'],EVSP_REQUIRE_DETACHED='1',EVSP_MIP_EXPECTED_RESULT_SHA256=c['source_result_sha256'],EVSP_MIP_EXPECTED_JOURNAL_SHA256='0'*64)
rejected('changed_journal_binding_native_main',lambda:n.main(['--result',c['source_result'],'--cover']),'staged journal does not match')
rejected('changed_native_source',lambda:patched_source(native.read_text()+'\n'),'native code SHA changed')
pa=w.read(c['source_production_result'])['physical_pool_audit'];v={'physical_pool_audit':pa,'source_result_sha256':c['source_result_sha256'],'source_journal_sha256':c['source_journal_sha256'],'model_sha256':model_digest()}
identity_gate(v,c)
bad=copy.deepcopy(v);bad['physical_pool_audit']['base_pool_ordered_sha256']='0'*64
rejected('changed_ordered_pool_identity',lambda:identity_gate(bad,c),'native pool identity changed')
bad=copy.deepcopy(v);bad['model_sha256']='0'*64
rejected('changed_model_identity',lambda:identity_gate(bad,c),'model identity changed')
bad=copy.deepcopy(v);bad['physical_pool_audit']['added_giro_route_count']=1
rejected('pool_augmentation',lambda:identity_gate(bad,c),'augmentation forbidden')
report={'status':'passed','manifest_sha256':w.sha(ROOT/'manifest.json'),'test_script_sha256':w.sha(__file__),'utc':w.now(),'checks':rows,'optimization_runs':0};w.save(ROOT/'negative_validation.json',report);print(report)
