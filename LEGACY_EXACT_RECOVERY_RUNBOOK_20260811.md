# Legacy exact-CG recovery runbook (2026-08-11)

## Scope

This recovers array `867334` tasks 22, 24, and 32. Slurm preempted them while
the legacy `f4e31c3` writer was appending a column. Requeue restarted the jobs,
but the old reader crashed on the malformed last JSONL record.

The recovery is deliberately isolated:

- `~/EVSP-DR` stays pinned at `f4e31c3` while its remaining array tasks run.
- Recovery runs use a separate checkout at the current `peel-and-price` commit.
- Legacy status, journal, iteration log, and available Slurm logs are archived
  byte-for-byte.
- Only copied working files are repaired.
- Instance and price hashes must be reconstructed from independent hashed
  result statuses. Missing or conflicting witnesses fail closed.
- Each recovery output has an exclusive process lock, durable per-iteration
  writes, and restart-safe identity status.

Recovered pools mix legacy-generated columns with current-code continuation
columns. They are valid pool/recovery evidence, but their combined wall-time
curves are **not** clean single-version performance benchmarks.

## One-time isolated checkout

Do not switch or pull the shared `~/EVSP-DR` checkout. It is both the pinned
legacy source and the home of still-running `f4e31c3` jobs. Create a dedicated
recovery worktree without changing that checkout:

```bash
SOURCE_ROOT="$HOME/EVSP-DR"
REC_ROOT="$HOME/EVSP-DR-legacy-recovery"

git -C "$SOURCE_ROOT" fetch origin peel-and-price
git -C "$SOURCE_ROOT" worktree add --detach \
  "$REC_ROOT" origin/peel-and-price
git -C "$REC_ROOT" status --short --branch
```

If `REC_ROOT` already exists, inspect it rather than deleting or overwriting
it. Never switch a worktree while one of its jobs is active.

An ignored-data directory is not populated automatically in a new worktree.
If the launcher reports missing inputs, copy only the required bytes from the
pinned checkout without overwriting anything already present:

```bash
mkdir -p "$REC_ROOT/data/duty_unions_big"

for relative in \
  duty_unions_big/Practice_Custom_DutyUnion_k30_r2.csv \
  duty_unions_big/Practice_Custom_DutyUnion_k30_r4.csv \
  hourly_prices_single_peak_18.csv \
  hourly_prices_transdev_sek.csv; do
  rsync -a --ignore-existing \
    "$HOME/EVSP-DR/data/$relative" \
    "$REC_ROOT/data/$relative"
done
```

## Preview, then submit

```bash
bash "$REC_ROOT/src/launch_legacy_bigtariff_recovery.sh" \
  --source-root "$HOME/EVSP-DR" \
  --array-job 867334 \
  --tasks 22,24,32

bash "$REC_ROOT/src/launch_legacy_bigtariff_recovery.sh" \
  --source-root "$HOME/EVSP-DR" \
  --array-job 867334 \
  --tasks 22,24,32 \
  --submit
```

Semantic job names are:

- `R22-30r2-p18-...`
- `R24-30r4-p18-...`
- `R32-30r2-sek-...`

Task 24 may report `WAIT_NO_WITNESS` until a hashed k30-r4 terminal status is
available. That job exits cleanly but leaves `WAIT_NO_WITNESS.txt` in its task
directory; Slurm does not retry it automatically. Rerun the same launcher
after a sibling terminal status exists. Completed migrations and active jobs
are idempotently validated, skipped, or locked.

The compute job's first pass is genuinely read-only: it copies the damaged
journal and iteration log to temporary storage, exercises the repair there,
validates every column and elapsed-time row, and only then applies the same
repair to the isolated destination. The original task-22/24/32 bytes are not
available on the Mac, so the first Unicorn submissions are also the required
Python 3.12 check against the real damaged tails. A refusal preserves all
source data and should be diagnosed rather than bypassed.

## Outputs

For task `N`, outputs are under:

```text
src/results/legacy_recovery/job867334/c<continuation-commit>/taskN/
```

Each cell contains:

- the current resume/result JSON;
- its current append-only column journal and iteration CSV;
- `*.migration_attestation.json`;
- `*.legacy_raw/` with the byte-identical source files, repair quarantine,
  and raw manifest;
- immutable continuation snapshots as elapsed-time marks are crossed.

Logs are under:

```text
src/cluster_logs/legacy_recovery/job867334/
```

## Monitoring

```bash
squeue --me -o '%.14i %.32j %.2t %.10M %R'

find "$REC_ROOT/src/results/legacy_recovery/job867334" \
  -name '*.migration_attestation.json' -print

grep -R -E '\[MIGRATE\]|\[RECOVERY\]|\[EXACT\] it|DONE|WAIT_' \
  "$REC_ROOT/src/cluster_logs/legacy_recovery/job867334" | tail -100
```

Never manually truncate the source journals and never point `--resume` at the
legacy files. A provenance or corruption refusal is evidence to preserve and
diagnose, not a reason to disable validation.
