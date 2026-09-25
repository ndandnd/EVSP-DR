"""Rebuild 48-cell final-fleet plot from the source-pinned cluster snapshot."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
import numpy as np

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / 'cluster' / 'snapshot.json'
snapshot = json.loads(SOURCE.read_text())
rows = []
for chain in range(1, 7):
    for target in range(33, 41):
        case = f'w{chain}_k{target}'
        artifacts = snapshot['full40_artifacts'][case]
        mip_record = artifacts['mip_result.json']
        cg_record = artifacts['cg.json']
        mip, cg = mip_record['content'], cg_record['content']
        # Top-level buses is the published final selected incumbent, not a stage cap.
        buses = mip['buses']
        assert isinstance(buses, int) and mip['incumbent_found']
        rows.append(dict(case=case, chain=chain, target=target,
                         final_buses=buses, buses_above_target=buses-target,
                         stage1_buses=mip['two_stage']['stage1_buses'],
                         fleet_bound=mip['fleet_bound'],
                         fleet_proven=mip['fleet_proven'],
                         final_status=mip['status_name'],
                         final_incumbent_source=mip['incumbent_source'],
                         cg_pricing_certified=cg['certified_rc_optimal'],
                         cg_stop_reason=cg['stop_reason'],
                         individual_route_replay=mip['physical_replay_validated'],
                         duplicate_removal_validated=mip['duplicate_trip_removal_validated'],
                         shared_capacity_validated=mip['cross_route_charger_capacity_validated'],
                         source_mip_path=mip_record['resolved_path'],
                         source_mip_sha256=mip_record['sha256'],
                         source_cg_path=cg_record['resolved_path'],
                         source_cg_sha256=cg_record['sha256']))
assert len(rows) == 48 and len({r['case'] for r in rows}) == 48
assert all(not r['cg_pricing_certified'] for r in rows)
assert all(not r['fleet_proven'] for r in rows)
assert all(r['buses_above_target'] > 0 for r in rows)
AUDITED_CSV = OUT.parent / 'cluster' / 'full40_cases.csv'
with AUDITED_CSV.open() as stream:
    audited = {(int(r['chain']), int(r['k'])):r for r in csv.DictReader(stream)}
assert len(audited) == 48
for row in rows:
    comparison = audited[row['chain'], row['target']]
    assert row['final_buses'] == int(comparison['fleet_incumbent'])
    assert row['source_mip_sha256'] == comparison['mip_sha256']
    assert abs(row['fleet_bound'] - float(comparison['finite_pool_fleet_only_bound'])) < 1e-9

with (OUT / 'full40_fleet_results.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

actual = np.array([r['final_buses'] for r in rows]).reshape(6, 8)
extra = np.array([r['buses_above_target'] for r in rows]).reshape(6, 8)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(10.8, 5.5), layout='constrained')
cmap = LinearSegmentedColormap.from_list('extra_buses', ['#f6f1e9', '#f4bf81', '#c7582b'], N=8)
norm = BoundaryNorm(np.arange(.5, 9.5, 1), cmap.N)
im = ax.imshow(extra, cmap=cmap, norm=norm, aspect='auto')
ax.set_xticks(range(8), labels=range(33, 41))
ax.set_yticks(range(6), labels=[f'C{i}' for i in range(1, 7)])
ax.set_xlabel('Target k (buses)', labelpad=12)
ax.set_ylabel('Chain', labelpad=12)
ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
ax.grid(which='minor', color='white', linewidth=2)
ax.tick_params(which='minor', bottom=False, left=False)
ax.tick_params(which='major', length=0, pad=9)
for spine in ax.spines.values():
    spine.set_visible(False)
for i in range(6):
    for j in range(8):
        ax.text(j, i, str(actual[i, j]), ha='center', va='center',
                fontsize=16, color='white' if extra[i, j] >= 7 else '#202833')
bar = fig.colorbar(im, ax=ax, ticks=range(1, 9), fraction=.04, pad=.04)
bar.set_label('Buses above target k', labelpad=12)
bar.outline.set_visible(False)
for extension in ['png', 'pdf', 'svg']:
    fig.savefig(OUT / f'full40_fleet_heatmap.{extension}', dpi=220, bbox_inches='tight', facecolor='white')
plt.close(fig)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

provenance = dict(schema='evsp-dr-full40-figure-v1', captured_utc=snapshot['captured_utc'],
                  source_snapshot=str(SOURCE), source_snapshot_sha256=sha(SOURCE),
                  extraction_field='full40_artifacts[case][mip_result.json].content.buses',
                  annotation='final published integer fleet', color='final buses minus target k',
                  cells=48, target_hits=0, cg_certificates=0, finite_pool_fleet_proofs=0,
                  stage1_final_discrepancies=[r['case'] for r in rows if r['stage1_buses'] != r['final_buses']],
                  independent_audited_csv_comparison=dict(source=str(AUDITED_CSV), sha256=sha(AUDITED_CSV),
                    matching_final_fleets=48, matching_mip_hashes=48, matching_pool_bounds=48),
                  source_artifact_hashes_in='full40_fleet_results.csv',
                  outputs={p.name:sha(p) for p in OUT.glob('full40_*') if p.is_file()},
                  builder_sha256=sha(Path(__file__)))
(OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
print(json.dumps({'cells':48,'final_k40':actual[:,-1].tolist(),'extra_range':[int(extra.min()),int(extra.max())]}))
