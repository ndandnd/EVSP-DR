# Cluster cleanup handoff (Unicorn `/home/nc437`)

**For a future agent.** This is a delegated housekeeping task. It is **not** on
the scientific critical path — do it when the operator is not mid-campaign, and
never at the cost of a running job.

Read `CLUSTER_OPERATING_RULES_20260821.md` first.

## Situation

The `scaglione-nfs-01` sysadmins asked all users to reduce `/home` usage.
`nc437` was **355 GB** (second largest of 13 users). The filesystem itself is at
30% of 130 TB, so this is courtesy, not an emergency.

**Progress so far: 383 GB → 192 GB.** Three actions, all reversible or verified:

| action | freed |
|---|---:|
| deleted 466 snapshot-duplicate column journals | **113 GB** |
| gzipped 1,349 April-2026 route dumps in `~/demandresponse/src/results` | **79 GB** |
| staged 61 ghost artifacts into `~/attic/` | 14 GB moved, not yet freed |

## What remains

| path | size | plan |
|---|---:|---|
| `~/ladder-lite` | ~147 GB | almost entirely **primary column journals**. gzip at **18.6x** → ~8 GB. |
| `~/EVSP-DR` | ~26 GB | contains the `ll_20260820c` campaign; journals here are cited evidence — compress, never delete |
| `~/attic` | 14 GB | staged ghosts; see the caution below |
| `~/demandresponse` | 5 GB | done |

Endpoint if fully executed: **~35–45 GB**, with no result lost.

## The one rule that matters

**Gate journal compression on chain completion.** `src/run_exact_pool_mip.py`
and `src/target_pool_feasibility.py` read the raw journal path out of the status
JSON and will **not** open a `.gz`. So compress a journal only when its
downstream analysis has finished:

    C=$HOME/ladder-lite/chain
    while IFS= read -r j; do
      b=$(basename "$j" .json.columns.jsonl)
      [ -s "$C/tf_$b.json" ] || continue          # analysis done?
      gzip -q "$j"
    done < <(find $HOME/ladder-lite -name '*.json.columns.jsonl' -mmin +60)

`-mmin +60` keeps it away from in-flight writes.

## Verified-safe deletion pattern

Snapshot journals are **byte-prefixes** of their primary, because a column
journal is append-only. Verify before deleting, on real pairs:

    n=$(stat -c %s "$snapshot"); head -c "$n" "$primary" | cmp -s - "$snapshot"

466 of 466 passed. Keep the snapshot **status JSONs** — they are small and hold
the trajectory record. Only the `.columns.jsonl` copies are redundant.

## Never touch

- `~/evsp_env` — the Python 3.12 interpreter every cluster job uses. Deleting or
  moving this kills every running job instantly.
- `~/ladder-lite/repo`, `repo-wp`, `repo-n6`, `repo-g`, `repo-af` — running jobs
  read source and data from these. ~50 MB each; not worth touching.
- `~/gurobi1102`, `~/egg` (a separate project).
- Anything under `~/ladder-lite/*/` whose cell still appears in `squeue`.

## Caution on `~/attic`

It holds two trees that are **cited evidence** in the project record:

- `EVSP-DR-k40fx-eb85ca0` — source for the "no k40 regression" finding.
- `EVSP-DR-scale-ladder-7937c22fef77` — holds the `B0001` activation traceback
  and `gate_250838` logs.

Give the attic a week of nothing breaking before deleting any of it. If those two
should persist, tar them into `~/evsp_archives/` first — they are small.

## Underlying design bug worth fixing, not just cleaning

**The CG snapshot mechanism writes a full copy of the column journal at every
mark.** With marks at 30/60/120/240/480/720/1440 that is up to seven copies per
cell. It produced **133 GB of provably redundant data** — over a third of the
quota. A snapshot should record a journal **offset** (line or byte count) plus
its status JSON, not duplicate the pool.

Raise this as a finding in `records/inbox/` with a provisional local label; the
curator assigns the canonical ID (see `records/inbox/README.md`).

## Order of work

1. Drain the chain queue, then run the gated journal compression above.
2. Compress `~/EVSP-DR` journals the same way (cited evidence — compress only).
3. After a week clean, `tar` the attic into `~/evsp_archives/` and remove it.
4. File the snapshot-duplication design finding.
