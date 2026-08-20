# Controlled k30 pool-MIP 2x2 (dry-run preparation only)

## Scope and stop boundary

This prepares one four-arm comparison on the same immutable task-22, 1,440
minute snapshot.  It does **not** submit Slurm jobs.  Do not add `--submit`
until the branch tip, prepared start, four dry-run plans, and hypotheses below
have been reviewed.

The source snapshot is not pricing-certified.  Any later MIP result is a
statement about this frozen pool plus explicitly injected routes, not a global
route-space certificate.

## Experimental factors

| Arm | Objective | Pool/start treatment | Expected Slurm name prefix |
|---|---|---|---|
| A | single-stage stored route cost | deterministic pool-greedy | `MPA` |
| B | two-stage fleet then cost | deterministic pool-greedy | `MPB` |
| C | single-stage stored route cost | merge validated partition columns and use them as the explicit start | `MPC` |
| D | two-stage fleet then cost | merge validated partition columns and use them as the explicit start | `MPD` |

All arms use 1,800 solver seconds, eight worker threads, MIP gap `0.0001`,
partition `scaglione`, no requeue, identical snapshot bytes, and one reviewed
detached Git commit.

## Hypotheses

1. **Objective effect:** B improves the fleet bound and/or integer incumbent
   relative to A because fleet count is optimized directly.
2. **Integral-backbone treatment:** C improves the first/final incumbent
   relative to A if the raw pool lacks a useful exact partition or contains
   one that Gurobi fails to assemble.  This is not automatically a pure
   warm-start comparison because missing supplied incidences are merged into
   C/D's feasible column set.
3. **Combined effect:** D performs best if both limitations matter.
4. If C/D load the validated start but do not materially improve on A/B, the
   principal limitation is downstream branch-and-bound/formulation behavior,
   not absence of a starting partition.

## Decision rules after an explicitly authorized smoke

- Treat an arm as invalid if its expected/observed Git commits differ, tracked
  code is dirty, its input hashes differ from the manifest, or its start is
  rejected/not observed.
- Compare B-A for the objective effect.  Compare C-A and D-B for the combined
  partition-column augmentation plus explicit-start treatment; D-C remains
  the objective effect under the same augmented pool/start.
- Use `mip_start.pool_columns_added/replaced/reused` to classify the treatment.
  Only when `added=0` and `replaced=0` do C-A and D-B hold the feasible
  incidence set fixed closely enough to interpret mainly as a start effect.
- Primary metrics: accepted start bus count/objective, first incumbent, final
  buses, stage-1 fleet bound, root relaxation, nodes, simplex iterations, and
  wall time.  Do not compare charging cost unless fleet counts match.
- A `TIME_LIMIT` result is an incumbent, not a proof.  `fleet_proven=true`
  applies only to the frozen augmented pool.

## Prepare one reviewed detached checkout

Run these only when no job is using the proposed new worktree path.  Fetching
does not switch or pull the pinned legacy checkout.

```bash
set -euo pipefail

LEGACY_ROOT="$HOME/EVSP-DR"
git -C "$LEGACY_ROOT" fetch origin cursor/recovery-audit
REVIEWED_COMMIT=$(git -C "$LEGACY_ROOT" rev-parse FETCH_HEAD)
MIP_ROOT="$HOME/EVSP-DR-mip-2x2-${REVIEWED_COMMIT:0:12}"

test "$(git -C "$LEGACY_ROOT" rev-parse HEAD)" = \
  f4e31c372bfb9e440617ed61193e227552dec49d
test ! -e "$MIP_ROOT"

git -C "$LEGACY_ROOT" worktree add --detach "$MIP_ROOT" "$REVIEWED_COMMIT"
test -z "$(git -C "$MIP_ROOT" branch --show-current)"
test -z "$(git -C "$MIP_ROOT" status --porcelain --untracked-files=no)"

mkdir -p "$MIP_ROOT/data/duty_unions_big"
rsync -a --ignore-existing \
  "$LEGACY_ROOT/data/duty_unions_big/Practice_Custom_DutyUnion_k30_r2.csv" \
  "$MIP_ROOT/data/duty_unions_big/"
rsync -a --ignore-existing \
  "$LEGACY_ROOT/data/hourly_prices_single_peak_18.csv" \
  "$MIP_ROOT/data/"
```

Before continuing, compare `REVIEWED_COMMIT` to the reviewed branch SHA.  Never
infer review merely from the branch name.

## Prepare and validate the exact-partition start

Use an interactive compute allocation rather than a Unicorn login node.  These
commands generate/re-realize route metadata but do not launch Gurobi or Slurm
MIP jobs.

```bash
set -euo pipefail

SNAP="$HOME/EVSP-DR-legacy-recovery-bab7bfe/src/results/legacy_recovery/job867334/cf31513f44007/task22/Practice_Custom_DutyUnion_k30_r2_peak18.m1440.snapshot.json"
START_DIR="$MIP_ROOT/src/results/controlled_mip_inputs/rec22_m1440"
RAW_START="$START_DIR/Practice_Custom_DutyUnion_k30_r2_giro_seed.json"
VALIDATED_START="$START_DIR/Practice_Custom_DutyUnion_k30_r2_peak18_rrz.json"

test -f "$SNAP"
test -f "$SNAP.columns.jsonl"
mkdir -p "$START_DIR"

cd "$MIP_ROOT"
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate /home/nc437/evsp_env

python -u src/make_giro_seed_routes.py \
  --instance duty_unions_big/Practice_Custom_DutyUnion_k30_r2.csv \
  --out "$RAW_START"

python -u src/rerealize_routes.py \
  --routes "$RAW_START" \
  --physics-from "$SNAP" \
  --instance duty_unions_big/Practice_Custom_DutyUnion_k30_r2.csv \
  --prices hourly_prices_single_peak_18.csv \
  --out "$VALIDATED_START"

sha256sum "$SNAP" "$SNAP.columns.jsonl" "$VALIDATED_START"
```

`rerealize_routes.py` must exit zero.  The launcher then independently requires
every route to pass physical replay and all supplied routes together to cover
each snapshot trip exactly once.  A partial GIRO/MATCHING file must fail rather
than fall back to singleton routes.

## Exact four dry-run commands

These commands intentionally omit `--submit`.  Each performs source preflight
and prints its immutable Slurm plan without creating a campaign or job.

```bash
cd "$MIP_ROOT"

python -u src/cluster_campaign.py mip \
  --result "$SNAP" --minutes 30 --mip-gap 0.0001 \
  --campaign rec22_m1440_A_single_greedy

python -u src/cluster_campaign.py mip \
  --result "$SNAP" --minutes 30 --mip-gap 0.0001 --two-stage \
  --campaign rec22_m1440_B_two_greedy

python -u src/cluster_campaign.py mip \
  --result "$SNAP" --minutes 30 --mip-gap 0.0001 \
  --initial-partition-routes "$VALIDATED_START" \
  --campaign rec22_m1440_C_single_validated

python -u src/cluster_campaign.py mip \
  --result "$SNAP" --minutes 30 --mip-gap 0.0001 --two-stage \
  --initial-partition-routes "$VALIDATED_START" \
  --campaign rec22_m1440_D_two_validated
```

Review the four printed plans for:

- distinct `MPA`/`MPB`/`MPC`/`MPD` names, each at most 15 characters;
- the same `EVSP_EXPECTED_COMMIT`, source snapshot hash, journal hash, 1,800
  seconds, eight worker threads, and explicit requested gap `0.0001`;
- campaign-specific staged-result hashes (expected to differ because each
  staged status embeds its own absolute journal path);
- no `--submit`, no `sbatch` execution, and no output overwrite;
- C/D reporting the same validated start hash and bus count.

## Expected artifacts after later authorization

For campaign name `NAME`:

```text
src/results/cluster_campaigns/NAME/submission.json
src/results/cluster_campaigns/NAME/<snapshot>_partition_30m.json
src/results/cluster_campaigns/NAME/input/<staged worker/snapshot/journal/start>
src/logs/cluster_campaigns/NAME/<semantic-name>_<job-id>.out
src/logs/cluster_campaigns/NAME/<semantic-name>_<job-id>.err
```

Each manifest must record expected/observed commit, detached/clean state,
explicit `requested_mip_gap`, staged hashes, objective mode, start treatment,
reviewed worker/runner hashes, and exact worker command.  Each result must independently record
observed/expected commit, requested versus achieved gap, and MIP-start
acceptance evidence.

## Explicitly out of scope

- No lossy JSONL recovery.
- No cancellation/requeue of running campaigns.
- No destructive cleanup or worktree reuse.
- No branch merge.
- No scientific claim from dry runs or from a time-limited smoke alone.
