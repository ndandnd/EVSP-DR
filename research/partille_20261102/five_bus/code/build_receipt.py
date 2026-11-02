"""Publish aggregate evidence and hashes only. Raw paths and charging traces stay private."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    ap.add_argument('private', type=Path)
    ap.add_argument('release', type=Path)
    args = ap.parse_args()
    base = args.root.resolve()
    private = args.private.resolve()
    release = args.release.resolve()
    src = base/'outputs/research_management_20260921/charging_fee_factorial/native'
    sys.path.insert(0,str(src/'bundle/outputs/week_20260921/cleanup_physics'))
    import joint_sequence_charging as j
    runs = {}
    grouped = [('literal_single_hour',private),('literal_cross_hour',private/'multihour'),
               ('clock_capped_cross_hour',private/'clock_capped/multihour')]
    for name, parent in grouped:
        runs[name] = {}
        for arm in ('original','saved_joint_fee0'):
            d=parent/arm
            data=read(d/(arm+'.json'))
            runs[name][arm]=dict(status=data['status'],objective=data['objective'],fixed_path_solver_bound=data['bound'],
                wall_seconds=data['wall_seconds'],incumbents=data['solutions'],files_sha256={
                f.name:sha(f) for f in sorted(d.iterdir()) if f.is_file() and f.suffix in {'.json','.log','.lp'}})
    records = read(private/'clock_capped/recorded/witness.json')
    pinned = read(private/'clock_capped/recorded/run.json')
    drop=[c['reported_kwh']-c['kwh'] for r in records for c in r['charges']]
    reductions=[pinned['original_terminal_targets'][k]-v for k,v in pinned['applied_terminal_targets'].items()]
    # Give each duty all its physically available time at every station, with
    # no charger competition and zero idle consumption. This is an upper bound.
    # In the one-hour treatment, no session can last more than the longest
    # intersection of its opportunity with an individual tariff hour.
    paths=read(private/'original/paths.json')
    margins={False:[], True:[]}
    for rr,orig in zip(paths,records,strict=True):
        for single in margins:
            soc=j.C-rr['actions'][0]['deadhead_kwh']
            for act in rr['actions'][1:]:
                soc-=j.TR[act['from_trip']]['energy']
                if act['kind']=='direct':
                    soc-=act['deadhead_kwh']
                    continue
                lo,hi=act['arrival_min'],act['latest_departure_min']
                duration=hi-lo
                if single:
                    duration=max((min(hi,60*(h+1))-max(lo,60*h) for h in range(math.floor(lo/60),math.ceil(hi/60))),default=0)
                soc-=act['inbound_kwh']
                soc=j.charge_soc_after_minutes(j.pr,act['station'],soc,duration)-act['outbound_kwh']
            margins[single].append(soc-pinned['original_terminal_targets'][orig['duty']])
    assert min(margins[False]) < -8e-6 and min(margins[True]) < -1
    first=read(private/'original/run.json')
    here=Path(__file__).resolve().parent
    output=dict(historical_solver_commit=first['frozen_solver_commit'],
        data_checkout_git_head=subprocess.check_output(['git','-C',str(base),'rev-parse','HEAD'],text=True).strip(),
        executed_code_sha256={p.name:sha(p) for p in sorted(here.glob('*.py'))},
        frozen_model_and_input_sha256=first['input_hashes'],
        strict_single_hour_wrapper_sha256=first['wrapper_sha256'],
        strict_cross_hour_wrapper_sha256=read(private/'multihour/original/run.json')['wrapper_sha256'],
        executed_clock_capped_wrapper_sha256=pinned['wrapper_sha256'],
        solver_version=first['solver_version'], resources=dict(host='local',threads_per_model=2,seconds_per_model=600,
             seed=0,relative_mip_gap=.0001,scheduler_jobs=[],external_dependencies=[]),
        audited_transfer_evidence=first['directed_connector'],runs=runs,nominal_replay_tolerance_kwh=1e-5,
        tighter_solver_feasibility_tolerance_across_hours=1e-8,
        relaxed_single_bus_terminal_upper_bound_minus_exact_nominal_floor_kwh=dict(
             one_hour_minimum=min(margins[True]),cross_hour_minimum=min(margins[False])),
        reconstructed_baseline=dict(charges_capped_gt_1e_minus_8_kwh=sum(v>1e-8 for v in drop),
             sum_capping_kwh=sum(drop),max_one_charge_capping_kwh=max(drop),
             min_one_duty_floor_reduction_kwh=min(reductions),max_one_duty_floor_reduction_kwh=max(reductions),
             per_duty_floor_reductions_sorted_kwh=sorted(reductions),sum_floor_reduction_kwh=sum(reductions)),
        aggregate_audit_sha256={p.name:sha(p) for p in sorted(release.glob('audit_*.json'))},
        scope='Five buses; fixed route assignment and mandatory recovered station paths; no pricing certificate or global duty optimum.')
    (release/'execution_and_evidence.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output['relaxed_single_bus_terminal_upper_bound_minus_exact_nominal_floor_kwh']))


if __name__ == '__main__':
    main()
