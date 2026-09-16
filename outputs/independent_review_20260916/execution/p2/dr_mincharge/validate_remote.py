from pathlib import Path
import json,sys,subprocess
from process_support import read,save,sha,require_hash,check_code,now
b=Path(__file__).resolve().parent;m=read(b/'manifest.json');check_code(m['code_path'],m['execution_commit'])
allinputs={p:h for c in m['cases'].values() for p,h in c['input_hashes'].items()}
for p,h in allinputs.items():require_hash(p,h)
for pair in m['pairs']:
 c,f=m['cases'][pair['cg_case']],m['cases'][pair['fixed_case']]
 assert c['input_hashes']==f['input_hashes'] and c['terminal_target_kwh']==f['terminal_target_kwh']
 assert c['k']==f['k']==15
m['tooling_sha256']={name:sha(b/name) for name in ['worker.py','worker.sub','process_support.py','submit.py','validate_remote.py']};m['policy_sha256']=sha(b.parent/'SCAGLIONE_RESOURCE_POLICY.md');save(b/'manifest.json',m)
D=Path('/share/scaglione/nc437/evsp-dr')/b.name;D.mkdir(exist_ok=True);(D/'cases').mkdir(exist_ok=True)
if not (b/'cases').exists():(b/'cases').symlink_to(D/'cases')
(b/'logs').mkdir(exist_ok=True)
p=subprocess.run(['/home/nc437/evsp_env/bin/python','-m','unittest','discover','-s','tests','-p','test_minimum_charge_pricer.py','-v'],cwd=m['code_path'],capture_output=True,text=True)
(b/'native_tests.log').write_text(p.stdout+p.stderr);assert p.returncode==0
save(b/'validation.json',dict(status='passed',utc=now(),manifest_sha256=sha(b/'manifest.json'),cases=len(m['cases']),pairs=len(m['pairs']),unique_inputs_verified=len(allinputs),matched_pair_inputs_and_target=True,native_tests_returncode=p.returncode,native_tests_sha256=sha(b/'native_tests.log')))
print(json.dumps(dict(status='passed_prepared_not_submitted',cases=len(m['cases']),manifest_sha256=sha(b/'manifest.json'))))
