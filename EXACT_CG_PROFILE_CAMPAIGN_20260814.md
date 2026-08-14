# Five-pool exact-CG profiling campaign

This package profiles one historical pool plus CA/CS/PA/PS.  The launcher is
dry-run by default and requires five explicit immutable snapshot paths.  It
does not contain or source a strict-mode block in the caller's SSH shell;
strict mode lives only inside the submitted worker script.

The reviewed profiler core is pinned to:

```text
702491e2b9fa548b75a8b140ba5a4213c06df24f
```

The packaging checkout must be detached, tracked-clean, descended from that
commit, and byte-identical to it for profiler/pricer/telemetry/master files.

## Detached checkout and environment

Run each command individually.  Do not switch a worktree used by active jobs.

```bash
git -C "$HOME/EVSP-DR" fetch origin cursor/exact-cg-performance-audit-2969
PROFILE_COMMIT=$(git -C "$HOME/EVSP-DR" rev-parse FETCH_HEAD)
PROFILE_ROOT="$HOME/EVSP-DR-profile-${PROFILE_COMMIT:0:12}"
git -C "$HOME/EVSP-DR" worktree add --detach "$PROFILE_ROOT" "$PROFILE_COMMIT"
git -C "$PROFILE_ROOT" status --short --branch
```

Activate the intended environment before invoking the launcher.  The launcher
and worker independently require Python 3.12 and import NumPy, pandas, and
SciPy from the exact interpreter recorded in the campaign.

```bash
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate /home/nc437/evsp_env
python -V
```

## Dry-run five explicit pools

Set these five paths to immutable `*.snapshot.json` artifacts.  The launcher
discovers each matching journal and generated instance/tariff bytes relative
to that snapshot's checkout, checks provenance, and prints five complete
`sbatch` commands without creating a campaign or submitting.

```bash
HIST=/absolute/path/to/historical-flat-k40-r2.snapshot.json
CA=/absolute/path/to/ca-m360.snapshot.json
CS=/absolute/path/to/cs-m360.snapshot.json
PA=/absolute/path/to/pa-m360.snapshot.json
PS=/absolute/path/to/ps-m360.snapshot.json

cd "$PROFILE_ROOT"
python -u src/launch_exact_cg_profile_campaign.py \
  --historical "$HIST" \
  --ca "$CA" --cs "$CS" --pa "$PA" --ps "$PS" \
  --campaign k40_master_factorial_m360_profile \
  --python "$(command -v python)" \
  --solve-limit-s 1800 \
  --repeat 3 \
  --mem-gb 64 \
  --job-hours 24
```

Review must confirm:

- five distinct outputs and names `PFhist-*`, `PFca-*`, `PFcs-*`, `PFpa-*`,
  `PFps-*`;
- one CPU, 64 GB, BLAS/OpenMP threads fixed to one;
- prefixes 1k/5k/10k/25k/50k and all three HiGHS methods, three repetitions;
- source and staged hashes for status, journal, instance, and tariff;
- expected packaging commit and pinned profiler-core commit;
- `--no-requeue`, no phase telemetry, and no `--submit`.

Only after explicit authorization, rerun the same command with `--submit`.
Every retry must use a fresh `--campaign`; outputs and reservations are
intentionally non-resumable/no-clobber.

## Read-only monitoring and summary

```bash
CAMPAIGN_ROOT="$PROFILE_ROOT/src/results/exact_cg_profiles/k40_master_factorial_m360_profile"

python -u src/monitor_exact_cg_profile_campaign.py \
  --campaign-root "$CAMPAIGN_ROOT" --format tsv

python -u src/summarize_exact_cg_profiles.py \
  --campaign-root "$CAMPAIGN_ROOT" --format tsv

python -u src/summarize_exact_cg_profiles.py \
  --campaign-root "$CAMPAIGN_ROOT" --format json
```

The summary has one row per pool/prefix/method with individual-repetition
failure counts plus median/min/max total time, median backend time, objective,
route weight, artificials, row/bound residuals, and peak RSS.

## Archive after completion

```bash
python -u src/archive_exact_cg_profile_campaign.py \
  --campaign-root "$CAMPAIGN_ROOT" \
  --out "$HOME/archives/k40_master_factorial_m360_profile.tar.gz"
```

The helper refuses existing archive artifacts, verifies source files remain
unchanged while archiving, and writes a sidecar manifest containing expected
commit, profiler-core commit, campaign manifest hash, every file checksum, and
archive checksum.

## Explicit limits

- No requeue/resume: retry under a new campaign name.
- No telemetry on the 22--24-hour live CA/CS/PA/PS CG jobs.
- No pool, pricing, master, MIP, or stopping-rule changes.
- No scientific claim follows from profiler timings alone.
