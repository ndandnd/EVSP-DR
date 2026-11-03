"""Audit trip-sequence membership and original consecutive links. Emit only aggregate results.

Run with Python 3 and two directories: earlier held-out campaign, later rematch.
The checker only reads source evidence. Trip identifiers never appear in its output.
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def one(paths):
    found = list(paths)
    check(len(found) == 1, 'expected a unique native attempt')
    return found[0]


def run(earlier, later):
    freeze_path = later / 'FREEZE.json'
    freeze = read(freeze_path)
    registration = read(earlier / 'submission_state.json')
    check(registration['submission']['job_ids'] == ['915057'], 'wrong previous job')
    controls = {}
    for control in sorted((earlier / 'results').glob('*/existing_root_control/job915057_*_r0/solve/receipt.json')):
        ctrl = read(control)
        case = ctrl['case']
        wrapper = read(control.parent.parent / 'wrapper_receipt.json')
        check(str(wrapper['array_job_id']) == '915057' and wrapper['status'] == 'terminal'
              and wrapper['exit_code'] == 0 and wrapper['scientific_receipt_sha256'] == sha(control)
              and wrapper['case'] == case, 'control wrapper or receipt mismatch')
        check(ctrl['arm'] == 'unchanged_root_exact_mip' and ctrl['binaries'] == ctrl['root_columns']
              and ctrl['solutions'] == 0 and case not in controls, 'wrong whole-root control')
        controls[case] = (control, ctrl)
    check(set(controls) == {f'c{c}_k08' for c in [1,3,4,5,6]}, 'incomplete original panel')
    targeted = sorted(c for c, (_, r) in controls.items() if r['solver_status'] == 3)
    check(targeted == freeze['task_map'] == ['c1_k08','c4_k08','c5_k08'], 'target subset differs from infeasible controls')
    check(all((rc['solver_status'],rc['outcome']) in [(3,'infeasible_in_original_root_pool'),(9,'time_limit_no_incumbent')]
              for _,rc in controls.values()), 'unexpected control result')
    result = []
    for idx,case in enumerate(targeted):
        old_receipt, ctrl = controls[case]
        rel = 'manifests/'+case+'.json'
        old_m = read(earlier / rel)
        new_m = read(later / rel)
        check(sha(earlier/rel) == ctrl['manifest_sha256'] and sha(later/rel) == freeze['artifact_sha256'][rel],
              'manifest digest mismatch')
        for key in ('case','input_sha256','pool','root_columns','root_prefix_sha256','summary_sha256','solver_commit','target_fleet'):
            check(new_m[key] == old_m[key], 'campaigns use different '+key)
        out = one((later/'results'/case).glob(f'job918045_{idx}_r0'))
        wrapper = read(out/'wrapper.private.json')
        check(wrapper['status']=='terminal' and wrapper['exit_code']==0 and str(wrapper['array_job_id'])=='918045'
              and wrapper['case']==case and wrapper['freeze_sha256']==sha(freeze_path),'rematch wrapper mismatch')
        receipt = out/'solve'/'receipt.json'
        rematch = read(receipt)
        source_summary = Path(new_m['summary'])
        source = read(source_summary)
        check(sha(source_summary)==new_m['summary_sha256']
              and source['columns_sha256']==new_m['root_prefix_sha256']
              and source['columns']==new_m['root_columns'], 'source root differs from registered prefix')
        for field in ('root_prefix_sha256','root_columns','input_sha256','solver_commit'):
            check(rematch[field] == ctrl[field] == new_m[field], 'receipts differ on '+field)
        for name,path in new_m['inputs'].items():
            check(sha(Path(path))==new_m['input_sha256'][name], 'physical input digest mismatch')
        check(rematch['source_hashes']['manifest']==sha(later/rel) and rematch['validated'] is True
              and rematch['exact_once'] is True and rematch['fleet']==8 and rematch['grid_plug_overload_minutes']==0,
              'missing validated eight-bus receipt')
        winner = one(a for a in rematch['attempts'] if a['outcome']=='validated_exploratory_candidate')
        candidate = out/'solve'/f"candidate_{winner['candidate']:02d}"
        selected = candidate/'joint/routes.json'
        check(sha(selected)==rematch['selected_sha256'], 'winner file digest mismatch')
        replay = read(candidate/'joint/replay.private.json')
        check(replay['validated'] and replay['exact_once'] and replay['fleet']==8 and replay['grid_plug_overload_minutes']==0,
              'winner does not pass its recorded physical replay')
        sequences = [tuple(r['trips']) for r in read(selected)]
        check(len(sequences)==8 and Counter(sequences)==Counter(tuple(r) for r in read(candidate/'paths.private.json')),
              'winning MIP/retimed passenger sequences differ')
        instance_path = Path(new_m['inputs']['instance'])
        with instance_path.open() as f:
            trips = sum(1 for _ in csv.DictReader(f))
        check(Counter(t for seq in sequences for t in seq)==Counter(range(trips)), 'winner fails exactly-once service')
        observed, edges = set(), set()
        root_digest = hashlib.sha256()
        with Path(new_m['pool']).open('rb') as f:
            for _ in range(new_m['root_columns']):
                line=f.readline()
                check(bool(line),'prefix ended too early')
                root_digest.update(line)
                sequence=tuple(json.loads(line)['trips'])
                observed.add(sequence)
                edges.update(zip(sequence,sequence[1:]))
        check(root_digest.hexdigest()==new_m['root_prefix_sha256'],'prefix digest mismatch')
        wanted = {edge for seq in sequences for edge in zip(seq,seq[1:])}
        check(len(wanted)==trips-8, 'unexpected duplicated adjacent links')
        result.append(dict(case=case, control_job=old_receipt.parent.parent.name, rematch_job=out.name,
                           instance_sha256=sha(instance_path),root_columns=new_m['root_columns'],
                           root_prefix_sha256=root_digest.hexdigest(),root_distinct_trip_sequences=len(observed),
                           target_fleet=8,replayed_exact_once_trips=trips,
                           winning_trip_sequences_absent=sum(s not in observed for s in sequences),
                           winning_trip_sequences_present=sum(s in observed for s in sequences),
                           winning_adjacent_trip_links=len(wanted),winning_adjacent_links_in_root=len(wanted & edges),
                           winning_adjacent_links_absent=len(wanted-edges),root_distinct_adjacent_links=len(edges),
                           native_root_summary_sha256=sha(source_summary),original_control_status='GUROBI_INFEASIBLE',
                           original_control_receipt_sha256=sha(old_receipt),rematch_receipt_sha256=sha(receipt),
                           winning_selection_sha256=sha(selected),winning_replay_sha256=sha(candidate/'joint/replay.private.json')))
    return dict(schema='rematch-passenger-sequence-membership-v1', original_control_array=915057,
                rematch_array=918045,original_panel_control_status={c:rc['outcome'] for c,(_,rc) in controls.items()},
                frozen_case_subset_equals_infeasible_controls=True,rematch_freeze_sha256=sha(freeze_path),cases=result,
                definition='Membership compares complete ordered passenger-trip sequences. A link joins consecutive passenger trips within one original root column; charging-site choices do not define either identity.')


if __name__=='__main__':
    print(json.dumps(run(Path(sys.argv[1]),Path(sys.argv[2])),indent=2))
