#!/usr/bin/env python3
"""Read-only-source command checks; fake campaign is never submitted."""
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('campaign', HERE/'campaign.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

def parser_options(source):
    tree = ast.parse(Path(source).read_text())
    return {arg.value for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr=='add_argument'
            for arg in node.args if isinstance(arg, ast.Constant)
            and isinstance(arg.value,str) and arg.value.startswith('--')}

def main():
    baseline = HERE.parents[1]
    capacity = Path(sys.argv[1])
    source = Path(sys.argv[2])
    ev = json.loads(source.read_text())
    baseline_options = parser_options(baseline/'src/exact_pricer_expanded.py')
    capacity_options = parser_options(capacity/'src/run_capacity_speed_event_cg.py')
    cases=[]
    for i,name in enumerate(['d00_g0','d00_g1','w1_k08','w4_k11','w6_k12','w1_k08']):
        warm=i>=2
        cases.append(dict(id=name+str(i),kind='warm' if warm else 'fresh',instance='/input.csv',prices='/prices.csv',
                          source_args=ev[name]['args'],parent={'descriptor':'/parent.json'},cpus=8 if warm else 2,
                          mem='96G' if warm else '32G',slurm_time='08:00:00' if warm else '04:30:00',
                          arm_seconds=7200,preparation_seconds=10800 if warm else 0,code=str(baseline)))
    for i in range(3):
        cases.append(dict(id='capacity'+str(i),kind='capacity',instance='/input.csv',prices='/prices.csv',
                          arm='capacity' if i<2 else 'combined',cpus=1,mem='24G',slurm_time='06:30:00',
                          arm_seconds=10800,preparation_seconds=0,code=str(capacity),commit='pin'))
    with tempfile.TemporaryDirectory(prefix='.local-check-',dir=HERE) as td:
        root=Path(td)
        for case in cases:
            for mode in ['reference','optimized']:
                command=c.command(case,mode,root/mode,root/'network.pkl' if case['kind']=='warm' else None)
                known=capacity_options if case['kind']=='capacity' else baseline_options
                assert all(word in known for word in command if word.startswith('--')), command
            if case['kind']=='warm':
                command=c.baseline_command(case,root/'prepare',root/'network.pkl',True)
                assert '--event-network-cache-only' in command
                assert '--inherit-event-pool-from' not in command
                assert all(word in baseline_options for word in command if word.startswith('--'))
        c.write(root/'manifest.json',dict(cases=cases,python=sys.executable))
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            c.launch(SimpleNamespace(root=root,submit=False))
        dry=json.loads(output.getvalue())
        assert len(dry['commands'])==9
        for cmd in dry['commands']:
            assert '--exclude=scaglione-compute-01' in cmd
            assert '--partition=default_partition' in cmd
            assert '--no-requeue' in cmd
        assert not (root/'jobs.json').exists()
        env=os.environ.copy()
        env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0')
        ok=c.run_process([sys.executable,'-c','print("ok")'],baseline,root/'ok',10,env)
        assert ok['returncode']==0 and not ok['watchdog_triggered']
        timed=c.run_process([sys.executable,'-c','import time;time.sleep(5)'],baseline,root/'timeout',.05,env)
        assert timed['watchdog_triggered'] and timed['returncode']!=0
        # Collector observes preparation and first arm even before pair_status exists.
        attempt=root/'cases'/cases[2]['id']/'123_r0'
        c.write(attempt/'allocation.json',dict(case=cases[2],host='test-host',started_utc='test-start',manifest_sha256='manifest-pin'))
        c.write(attempt/'prepare/execution.json',dict(started_utc='prep-start',command=['python','solver','--event-network-cache-only']))
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):c.collect(SimpleNamespace(root=root))
        observed=json.loads(stream.getvalue())
        assert len(observed['attempts'])==1 and len(observed['rows'])==2
        assert observed['attempts'][0]['preparation']['started_utc']=='prep-start'
        assert all(r['process_state']=='not_started' for r in observed['rows'])
        assert len(observed['pending_cases'])==8
        c.write(attempt/'reference/execution.json',dict(started_utc='arm-start',command=['python','solver','--wall-limit-s','7200']))
        c.write(attempt/'reference/cg.json',dict(certified_rc_optimal=False,stop_reason='wall_limit',
                final={'lp_obj':123,'route_weight':2,'min_rc':-.2},iterations=4))
        cap_attempt=root/'cases'/cases[6]['id']/'456_r0'
        c.write(cap_attempt/'allocation.json',dict(case=cases[6]))
        c.write(cap_attempt/'optimized/cg.json',dict(schema='capacity-test',certified_rc_optimal=True,
                terminal_exact_min_reduced_cost=0,final={'objective':77,'route_weight':1,'artificial_total':0},
                pool=str(cap_attempt/'optimized/pool.jsonl'),iterations=[{'pricing_s':3,'lp_solve_s':.1,'nonzero_capacity_duals':1}]))
        (cap_attempt/'optimized/pool.jsonl').write_text('{}\n')
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):c.collect(SimpleNamespace(root=root))
        observed=json.loads(stream.getvalue())
        live=next(r for r in observed['rows'] if r['case']==cases[2]['id'] and r['mode']=='reference')
        assert live['process_state']=='started_no_completion_record'
        assert live['flags']['--wall-limit-s']=='7200' and live['certified_rc_optimal'] is False
        cap=next(r for r in observed['rows'] if r['case']==cases[6]['id'] and r['mode']=='optimized')
        assert cap['certified_rc_optimal'] is True and cap['terminal_exact_min_reduced_cost']==0
        assert cap['artifacts']['pool']['sha256'] and cap['final']['objective']==77
        assert cap['pricing_seconds_completed_iterations']==3
    print('PASS: command flags, dry launch, watchdog, preparation/live-first-arm/pending collection, both certificate schemas and pool hashes')

if __name__=='__main__':main()
