"""One restart-isolated, wall-clocked sparse-prefix chain in one Slurm allocation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
TOTAL=dict(cg=10800,heuristic=5400,mip=3600)
PIN='3a0ab6a8ab50e4af7d5a7be7688d5ffa1fbbf78c'
TIER={8:(5,8),10:(5,8,10)}


def read(p):
    return json.loads(Path(p).read_text())


def write(p,value):
    p=Path(p)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(p)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(c,msg):
    if not c:
        raise ValueError(msg)


def limits(n, i, spent):
    # Progressively larger nominal shares, with no rollover of unused shares.
    denom=n*(n+1)//2
    return {p:max(0,min(int(total*(i+1)/denom), math.floor(total-spent[p]))) for p,total in TOTAL.items()}


def audit_stage_input(row,data,shared):
    files=dict(instance=data/row['csv'],locations=data/'Ref_dict.csv',deadheads=data/'par_ref_dhd.csv',prices=data/'hourly_prices_flat.csv')
    hashes={k:sha(v) for k,v in files.items()}
    require(hashes==dict(shared,instance=row['instance_sha256']),'input hash differs from preregistered case')
    return hashes


def main():
    opts=argparse.ArgumentParser()
    opts.add_argument('--k',type=int,choices=TIER,required=True)
    opts.add_argument('--chain',type=int,choices=range(1,7),required=True)
    a=opts.parse_args()
    start=time.monotonic()
    root=Path(os.environ['PREFIX_ROOT'])
    solver=root/'solver'
    env=os.environ.copy()
    tier=TIER[a.k]
    jid=env['SLURM_JOB_ID']
    restart=int(env.get('SLURM_RESTART_COUNT','0'))
    dest=root/'results'/f'c{a.chain}_to_k{a.k:02}'/f'job{jid}_r{restart}'
    dest.mkdir(parents=True,exist_ok=False)
    trace=dict(stage='startup',status='running',chain=a.chain,target_k=a.k,policy='sparse-prefix-selected-sequences-simple-dive',
               solver_base_commit=PIN,phase_budget_s=TOTAL,allocation_s=72000,cpus=int(env['SLURM_CPUS_PER_TASK']),
               attempt=dest.name,job_family=env['SLURM_ARRAY_JOB_ID'],stages=[],started_epoch=time.time())
    write(dest/'receipt.json',trace)
    try:
        # A Slurm restart cannot create an uncharged second end-to-end budget.
        require(not restart,'interrupted allocation: preserve prior attempt; no free restart')
        require(trace['cpus']==8,'requires eight-CPU allocation')
        meta=read(root/'deployment.json')
        commit=subprocess.check_output(['git','-C',str(solver),'rev-parse','HEAD'],text=True).strip()
        require(commit==meta['patched_commit'] and not subprocess.check_output(
            ['git','-C',str(solver),'status','--porcelain','--untracked-files=no']), 'solver source/pin changed')
        require(all(sha(HERE/n)==h for n,h in meta['files_sha256'].items()),'campaign files differ')
        require(sha(solver/'src/prefix_initializer.py')==meta['initializer_sha256'] and
                sha(solver/'src/run_capacity_terminal_cg.py')==meta['solver_source_sha256'],'patched solver bytes differ')
        require(sha(root/'validate_selection.py')==meta['validator_sha256'],'independent validator differs')
        trace.update(patched_commit=commit,deployment_sha256=sha(root/'deployment.json'))
        design=read(HERE/'paired_design.json')
        cases=dict(design['cases'],**read(HERE/'prefix_k05.json'))
        data=Path(env['CAPACITY_DATA'])
        prices=data/'hourly_prices_flat.csv'
        spent={p:0. for p in TOTAL}
        seal=None
        def timed(cmd,log,child_env):
            remaining=72000-(time.monotonic()-start)
            if remaining<=0: raise TimeoutError('end-to-end wall budget exhausted')
            with log.open('w') as stream:
                return subprocess.run(cmd,cwd=solver/'src',env=child_env,stdout=stream,
                        stderr=subprocess.STDOUT,timeout=remaining).returncode
        for i,k in enumerate(tier):
            stage_at=time.monotonic()
            stage=dest/f'k{k:02}'
            stage.mkdir()
            row=cases[f'c{a.chain}_k{k:02}']
            hashes=audit_stage_input(row,data,design['shared_hashes'])
            cap=limits(len(tier),i,spent)
            require(all(v>0 for v in cap.values()),'phase budget exhausted by predecessors')
            entry=dict(k=k,instance_sha256=hashes['instance'],input_sha256=hashes,allocated_phase_s=cap,
                       ancestor_wall_s=stage_at-start,init='fresh' if seal is None else 'receiver-regenerated sequences plus fresh singletons',
                       predecessor_sha256=sha(seal) if seal else None)
            trace['stages'].append(entry)
            trace['stage']=f'k{k:02}_solve'
            argv=[sys.executable,'-u',str(solver/'src/run_capacity_terminal_cg.py'),'--data-dir',str(data),
                   '--csv',row['csv'],'--prices',str(prices),'--target','0','--fleet-cap',str(row['time_only_floor']+4),
                   '--cg-seconds',str(cap['cg']),'--dive-seconds',str(cap['heuristic']),'--mip-seconds',str(cap['mip']),
                   '--charge-kw','240','--battery-kwh','240','--start-fee','5','--non-strict-tariff',
                   '--station-kw','PARX=60','--out',str(stage/'solve')]
            if seal: argv+=['--validated-prefix',str(seal)]
            entry['argv']=argv
            write(dest/'receipt.json',trace)
            t=time.monotonic()
            exit_code=timed(argv,stage/'solver.log',env)
            entry.update(exit_code=exit_code,solver_wall_s=time.monotonic()-t)
            sp=stage/'solve/summary.json'
            require(sp.is_file(),'no native solver summary')
            summ=read(sp)
            entry.update(summary_sha256=sha(sp),graph_s=summ['graph_seconds'],cg_certificate=summ.get('cg_pricing_certified'),
                         phase_wall_s=summ['sparse_timing'],selection_state=summ['selection_state'],
                         finite_pool_fleet=summ.get('fleet_stage'),finite_pool_charging=summ.get('charging_stage'))
            for p in spent: spent[p]+=float(summ['sparse_timing'][p])
            trace['actual_phase_seconds']=dict(spent)
            trace['phase_limits_within_total']={p:spent[p]<=TOTAL[p] for p in TOTAL}
            if exit_code==3 and summ['selection_state']=='no_selection':
                trace['status']='target_no_selection' if k==a.k else 'predecessor_no_selection'
                return
            require(exit_code==0,'solver failed')
            selected=stage/'solve/selected_routes.json'
            t=time.monotonic()
            check=stage/'verification.json'
            var=dict(env,PYTHONPATH=str(solver/'src'),PYTHONDONTWRITEBYTECODE='1')
            v=timed([sys.executable,str(root/'validate_selection.py'),'--summary',str(sp),'--selected',str(selected),
                     '--out',str(check),'--arm','baseline','--start-fee','5'],stage/'validation.log',var)
            val=read(check)
            require(val['summary_sha256']==sha(sp) and val['selected_sha256']==sha(selected)
                    and val.get('instance_sha256')==row['instance_sha256'] and val.get('cover_valid') is True
                    and val.get('grid_plug_overload_minutes')==0 and val.get('trip_count')==row['n_trips']
                    and (v==0 and val.get('validated') is True or v==3 and val.get('reason')=='cover is feasible but has duplicate service'),
                    'unauthenticated or infeasible predecessor/target selection')
            entry.update(validation_s=time.monotonic()-t,validator_exit=v,verification_sha256=sha(check),selected_sha256=sha(selected),
                         exact_once=val['exact_once'],duplicate_service=val['duplicate_service'],fleet=val['fleet'],
                         grid_plug_overload_minutes=val['grid_plug_overload_minutes'],objective_nonfleet=val['objective_nonfleet'])
            if i+1<len(tier):
                nxt=cases[f'c{a.chain}_k{tier[i+1]:02}']
                seal=stage/'prefix_seal.json'
                write(seal,dict(base_solver_commit=PIN,solver_source_sha256=meta['solver_source_sha256'],
                      initializer_sha256=meta['initializer_sha256'],summary_sha256=sha(sp),selected_sha256=sha(selected),
                      validation_sha256=sha(check),parent_csv=row['csv'],receiver_csv=nxt['csv'],
                      source_instance_sha256=row['instance_sha256'],receiver_instance_sha256=nxt['instance_sha256']))
            entry['stage_wall_s']=time.monotonic()-stage_at
            write(dest/'receipt.json',trace)
        trace['status']='completed_exact_once' if trace['stages'][-1]['exact_once'] else 'completed_duplicate_cover'
    except BaseException as exc:
        trace.update(status='execution_error',reason=str(exc))
        raise
    finally:
        trace.update(ended_epoch=time.time(),end_to_end_s=time.monotonic()-start)
        write(dest/'receipt.json',trace)


if __name__=='__main__':
    main()
