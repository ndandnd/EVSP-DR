# EVSP-DR post-fix overnight campaign

This campaign answers four open engineering questions without repeating the
completed 94-task repool, 200-task tariff, 50-task realism, or 27-task grid
campaigns:

1. Does the corrected restricted master solve the latest terminal persisted
   pool from each large run without the former self-created row violation?
2. How does integer fleet/cost quality change when the MIP receives columns
   generated after 1, 3, 6, 10, or 15 hours?
3. Does continuing exact CG without the two-hour stall rule from 6 to 72 total
   hours materially improve six stratified k=30/k=40 controls?
4. At a fixed six-hour CG pool, how much fleet quality is gained by giving the
   final MIP 15 minutes, one hour, three hours, or six hours?

## Launch on Unicorn

After pulling the campaign commit, submit one lightweight job from the login
node:

```bash
set -euo pipefail
cd ~/EVSP-DR
git switch peel-and-price
git pull --ff-only origin peel-and-price
export EVSP_DR_ROOT=$HOME/EVSP-DR
bash src/launch_overnight_correctness.sh
```

The wrapper pins the current commit and only hands the prep script to Slurm.
The `OCprep` job runs on
`default_partition`; it performs all journal validation, SHA256 work, input
archiving, and downstream `sbatch` calls on its allocated compute node. The
heavy compute-side helper is `prepare_overnight_correctness.sh`; it refuses
login-node execution. If invoking `sbatch` manually, pass the commit explicitly:
`sbatch src/submit_overnight_correctness_prep.sub "$(git rev-parse HEAD)"`.

The prep job is semantically idempotent: an output is skipped only when its
JSON is complete, its source status+journal hashes match, and its runtime
commit matches the prep commit. It archives and hashes both
`data/duty_unions_big` and every selected source status/journal pair. That
input directory is ignored by Git and must not be regenerated in place.

Depending on which immutable snapshots have landed, it submits up to:

- 50 `LA...` raw-master audits of the latest terminal pool per run on
  `default_partition` (one CPU, 24 GB, two hours, requeue enabled);
- 30 `MC...` fleet-first snapshot MIPs on `scaglione` (five CG ages, one-hour
  MIP budget), plus up to 18 `MB...` MIP-budget cells on the fixed six-hour
  pools (15 minutes, three hours, and six hours; the one-hour cell is shared
  with the first curve). All use eight CPUs, 32 GB, and explicit no-requeue;
- six `CC...` no-stall exact-CG controls on `default_partition` (one CPU,
  48 GB, source pool at six hours, stop at 72 total hours, requeue enabled,
  with a two-hour Slurm margin for graceful serialization).

The six controls are k30-r5/peak12, k30-r3/SEK, k40-r1/peak12,
k40-r2/SEK, k40-r3/peak08, and k40-r4/peak18. They are copied from immutable
six-hour snapshots into isolated journals, so they cannot race with the old
`cg-bigtar` jobs.

For a given control, every timed MIP pool must receive the same **completely**
re-realized GIRO partition seed. A cell whose full seed cannot be re-realized
fails explicitly; a partial seed is never silently admitted. Thus successful
cells measure the incremental value of exact-CG columns beyond a strong common
feasible seed; they are not a test of unaided pool integrality.

The `m60`, `m180`, and later labels are nominal publication thresholds. The
actual cumulative CG wall time recorded in each snapshot/result is the timing
variable used in analysis, because a snapshot is emitted after an iteration
and may land later than its nominal threshold.

Do not pull or switch the shared checkout after submission while jobs are
pending. Each worker verifies the launch commit and refuses to run on checkout
drift, preventing a pending campaign from silently mixing code versions.

## Monitor

```bash
cd ~/EVSP-DR
squeue --me -o '%.14i %.42j %.2t %.10M %R'
python3 src/summarize_overnight_correctness.py
```

Job prefixes are meaningful in `squeue`:

- `LA30r5p12`: LP audit, k30 replicate 5, peak at 12;
- `MC40r2sekh10`: MIP curve, k40 replicate 2, SEK tariff, ten-hour pool;
- `MB40r2sek06h`: six-hour MIP budget on the fixed six-hour CG pool;
- `CC40r1p12`: no-stall CG continuation, k40 replicate 1, peak at 12.

Each prefix is followed by short status-hash, journal-hash, and commit suffixes.
Those suffixes prevent an older active job from being mistaken for the same
cell at a newer pool or code version.

The submission manifest, campaign source archives, both hashes for every
selected source pool, job IDs, allocation limits, and output paths are under
`src/results/campaign_manifests/`. Runtime logs are ignored by Git but included
by the collector.

For each no-stall control, prep also freezes the old canonical status, journal,
and iteration trajectory when that run is already terminal and valid. If it is
missing or live, the manifest records that fact and the comparison is
forward-only from the immutable six-hour snapshot; it must not be described as
a paired comparison with the old stopping rule.

## Collect after completion

```bash
cd ~/EVSP-DR
sbatch src/submit_overnight_correctness_collect.sub
```

The `OCcollect` job performs tar compression and SHA256 on
`default_partition`; `collect_overnight_correctness.sh` refuses to do that work
on the login node. When the batch job completes, copy the resulting `.tar.gz`
and `.sha256` from the Unicorn home directory to the authenticated Mac and
publish them as a GitHub release. Unicorn still has no GitHub write
credentials, so do not put a personal token there.

## Interpretation rules

- A successful raw-master audit means the *raw* HiGHS primal meets the recorded
  tolerance. It does not certify pricing optimality.
- `fleet_proven=true` in a snapshot MIP means stage 1 proved its integer fleet
  count over that frozen augmented pool. Stage 2 runs only after that proof and
  only with remaining time. `optimal_scope` distinguishes fleet-only from full
  lexicographic optimality. None is a global route-space proof.
- The no-stall controls are incumbent trajectories unless reduced-cost pricing
  actually certifies them. Compare them to an old stopping trajectory only
  where the prep manifest confirms that terminal baseline was archived.
- These tariff pools are presently algorithm/stopping evidence. Do not report
  demand-response savings from them until terminal energy, charge-start cost,
  station power, and shared charger-capacity treatment are normalized.
