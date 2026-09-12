from pathlib import Path
from unittest import mock
import json,sys
O=Path('/home/nc437/ladder-lite/graph_recovery_20260912');B=Path('/home/nc437/ladder-lite/graph_recovery_retry2_20260912')
sys.path.insert(0,str(O/'code/src'));import exact_pricer_expanded as exact
checks=[]
for cid in ['d00_g3','parent32']:
 starts=list((O/'cases'/cid).glob('cache_*_start.json'));assert len(starts)==1
 argv=json.loads(starts[0].read_text())['argv'][2:];ix=argv.index('--out');del argv[ix:ix+2]
 with mock.patch.object(exact,'run_cg',return_value={}) as run:
  assert exact.main(argv)==0;run.assert_called_once();a=run.call_args.args[0];assert a.event_network_cache_only and a.out is None
 checks.append({'case_id':cid,'parser_passed':True,'argv':argv})
(B/'cache_cli_validation.json').write_text(json.dumps(checks,indent=2)+'\n');print('Both complete cache commands pass actual parser; constructor mocked for this launch check')
