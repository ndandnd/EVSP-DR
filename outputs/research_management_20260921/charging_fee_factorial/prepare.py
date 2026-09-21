"""Freeze the existing fixed-path charging model for a controlled fee factorial."""
from pathlib import Path
import hashlib
import itertools
import json
import shutil

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
OLD = ROOT / 'outputs/week_20260921/cleanup_physics'
B = P / 'bundle'
sources = {}


def copy(src, rel):
    dst = B / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    sources[rel] = {'original_path': str(src), 'sha256': hashlib.sha256(dst.read_bytes()).hexdigest()}


base = 'outputs/week_20260921/cleanup_physics/'
for name in ['matched_fixed_windows.py', 'joint_sequence_charging.py', 'saved_sequence_replay.json', 'inputs/peak08_h26.csv']:
    copy(OLD / name, base + name)
for name in ['original.json', 'k05.csv']:
    rel = 'outputs/meeting_20260917/route_explainer/inputs/' + name
    copy(ROOT / rel, rel)
copy(ROOT / 'data/Par_VehicleDetails_Updated.csv', 'data/Par_VehicleDetails_Updated.csv')
rel = '.codex-work/review-strict-chain-20260916/src/giro_partille_physics.py'
copy(ROOT / rel, rel)
for peak in [8, 12, 18]:
    src = OLD / 'inputs/peak08_h26.csv' if peak == 8 else ROOT / f'.codex-work/zero-fee-terminal-cg/data/tariff_response/peak{peak:02}_h26.csv'
    copy(src, f'tariffs/peak{peak:02}.csv')

# Mathematical model unchanged. Only solver budget, thread count and seed are parameters.
path = B / (base + 'joint_sequence_charging.py')
text = path.read_text()
assert 'def solve(arm,fee):' in text
assert 'm.Params.Threads=4;m.Params.TimeLimit=120;' in text
text = text.replace('def solve(arm,fee):', 'def solve(arm,fee,seconds=600,threads=2,seed=0):')
text = text.replace('m.Params.Threads=4;m.Params.TimeLimit=120;', 'm.Params.Threads=threads;m.Params.TimeLimit=seconds;m.Params.Seed=seed;')
text = text.replace("m.dispose();print(arm,result['status'],result['objective'],result.get('capacity_peaks'))", "m.dispose();print(arm,result['status'],result['objective'],result.get('capacity_peaks'));return result")
path.write_text(text)
sources[base + 'joint_sequence_charging.py']['executed_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()

cases = [dict(case_id=f'{arm}_peak{peak:02}_fee{fee}', fee_pair_id=f'{arm}_peak{peak:02}',arm=arm, peak=peak, fee=fee, seconds=600, threads=2, seed=0)
         for arm, peak, fee in itertools.product(['original', 'saved_joint_fee0', 'saved_joint_fee5'], [8, 12, 18], [0, 5])]
manifest = dict(question='Within an identical frozen trip assignment and fixed station path, how does a start fee change charging?',
                source_model='outputs/week_20260921/cleanup_physics/joint_sequence_charging.py',
                cases=cases, sources=sources,
                arm_labels={'original':'Original trip assignment with recovered station paths',
                            'saved_joint_fee0':'Saved fee0-derived trip assignment with recovered station paths',
                            'saved_joint_fee5':'Saved fee5-derived trip assignment with recovered station paths'},
                model_family='Fixed trip-sequence charging MIP; no column generation',
                initialization='No MIP starts; Gurobi default initialization, Seed0 in every case',
                objective='Purchased energy at synthetic tariff plus start_fee times charging starts',
                master_sense='Not a covering master; all62 fixed-assignment trips served exactly once',
                source_modification='Only TimeLimit, Threads and Seed parameterization and returning the existing result; mathematical model unchanged',
                physics=dict(group='18E1', battery_kwh=236.44, initial_kwh=236.44, reserve_fraction=.15, depot_kw=60,
                             opportunity='documented nonlinear taper', min_charge_minutes=3, idle_kw=.1,
                             station_capacity='one at2190L and4808 within five-bus cohort',
                             terminal='each saved bus matched to original duty by maximum overlap; identical within fee pair'),
                scope='fixed sequences and station paths; one optional charge per gap confined to one tariff hour; not CG or unrestricted charging/routing optimum',
                limitations=['static reference deadhead', 'original recorded path need not belong to this restricted charging model', 'no background35buses, FIFO, platform blocking or crew constraints'],
                resources=dict(partition='default_partition', cpus=2, memory='8G', allocation='00:20:00', independent_tasks=18, concurrency=18, requeue=True, exclude='scaglione-compute-01'))
(P / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Frozen {len(cases)} cases and {len(sources)} source files')
