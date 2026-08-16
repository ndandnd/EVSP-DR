# Completed k40 factorial analysis, archive, and strict-MIP screen

This package is dry-run/read-only by default.  No command below submits the
12-cell screen.  The two completed campaigns are:

```text
~/EVSP-DR-k40fx-eb85ca0/src/results/k40_factorial/k40fx_20260814T140232Z_eb85ca0c
~/EVSP-DR-k40fx-eb85ca0/src/results/k40_factorial/k40fx_20260814T191933Z_eb85ca0c
```

Visible `k40r1` stems are accepted only as the documented naming error; all
statuses must carry intended k40-r2 instance SHA
`3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd`.
Generation provenance is pinned to full commit
`eb85ca0cc439956939ba6bf9c42958808d89aadd`, 947 trips, the deterministic
seed-20260803 manifest entry, flat-tariff bytes, and actual source-data bytes.

## Read-only factorial summary

```bash
python -u src/summarize_k40_factorial.py \
  --campaign-dir "$HOME/EVSP-DR-k40fx-eb85ca0/src/results/k40_factorial/k40fx_20260814T140232Z_eb85ca0c" \
  --campaign-dir "$HOME/EVSP-DR-k40fx-eb85ca0/src/results/k40_factorial/k40fx_20260814T191933Z_eb85ca0c" \
  --historical /absolute/path/to/historical-k40-r2-final.json \
  --out-prefix /new/output/path/k40_factorial_summary
```

JSON/CSV/Markdown are published together under
`k40_factorial_summary.bundle/` by one atomic directory rename; a failed
publication exposes no partial bundle and a lock prevents in-place retries.
Route weight,
artificials, objective, and minimum reduced cost remain separate.  Deltas from
historical `39.252026205592166` use artificial-free m1320 rows and recorded
`wall_s`, never nominal filename age.

## Compute-node archive

First save canonical pipe-delimited accounting for all ten prep/arm jobs as an
immutable input:

```bash
sacct -X -P -j JOB_IDS \
  --format=JobIDRaw,JobName,State,Elapsed,ExitCode,MaxRSS \
  > /absolute/path/to/sacct-accounting.txt
```

The helper requires every root job to be `COMPLETED` with exit `0:0`, matching
stdout/stderr and completion markers. It refuses
login nodes, existing output, changed sources, symlinks, or missing trajectories
and logs.  It embeds SHA-256 for every campaign/historical/log/accounting member
and verifies the completed archive before atomic publication.

```bash
sbatch \
  --export=HOME,PATH,USER,EVSP_DR_ROOT="$PROFILE_ROOT",EVSP_EXPECTED_COMMIT="$REVIEWED_COMMIT",EVSP_PROFILE_PYTHON="$(command -v python)" \
  src/submit_k40_factorial_archive.sub \
  "$HOME/EVSP-DR-k40fx-eb85ca0/src/results/k40_factorial/k40fx_20260814T140232Z_eb85ca0c" \
  "$HOME/EVSP-DR-k40fx-eb85ca0/src/results/k40_factorial/k40fx_20260814T191933Z_eb85ca0c" \
  /absolute/path/to/historical-k40-r2-final.json \
  /absolute/path/to/sacct-accounting.txt \
  /new/output/path/k40_factorial_evidence.tar.gz
```

This command is documentation only; it was not executed by Cursor.

## Dry-run strict-MIP screen

Prepare one re-realized GIRO route file containing exactly 40 routes, every
factorial trip exactly once, zero `infeasible` entries, flat tariff, and
300/300/0 physics.  This command uses one immutable source snapshot, rejects
partial re-realization, then runs the exact-MIP loader in `--validate-only`
mode to replay every route physically.  It does not launch Gurobi optimization:

```bash
python -u src/prepare_k40_factorial_giro_start.py \
  --snapshot "$HOME/EVSP-DR-k40fx-eb85ca0/src/results/k40_factorial/k40fx_20260814T140232Z_eb85ca0c/k40r1_flat_CA.m360.snapshot.json" \
  --out /new/output/path/k40_r2_flat_giro_40_rerealized.json \
  --python "$(command -v python)"
```

The visible `k40r1` stem in that example is only the known naming error; the
preparer requires the intended k40-r2 content hash.  The launcher structurally
validates the published output again and every worker replays all routes before
Gurobi.

```bash
python -u src/launch_k40_factorial_mip_screen.py \
  --replicate "R1=$HOME/EVSP-DR-k40fx-eb85ca0/src/results/k40_factorial/k40fx_20260814T140232Z_eb85ca0c" \
  --replicate "R2=$HOME/EVSP-DR-k40fx-eb85ca0/src/results/k40_factorial/k40fx_20260814T191933Z_eb85ca0c" \
  --validated-start /absolute/path/to/re-realized-40-duty-giro.json \
  --mode screen \
  --campaign k40_factorial_mip_screen_30m \
  --python "$(command -v python)" \
  --plan-out /new/output/path/k40_factorial_mip_screen_30m.plan.json
```

The plan contains exactly 12 cells: two replicates × CA/CS × m360/m720/m1440.
Names encode a campaign nonce plus replicate/treatment/age/budget and remain
under 15 characters.
All cells use strict binary partitioning, validated 40-duty start, two-stage
fleet-first objective, 1,800 seconds, eight threads, Scaglione, no artificials,
and no requeue.

Review the canonical plan and preserve its exact hash:

```bash
PLAN_SHA=$(sha256sum /new/output/path/k40_factorial_mip_screen_30m.plan.json | awk '{print $1}')
echo "$PLAN_SHA"
```

Submission is impossible without rerunning the identical command with both
`--approved-plan-sha256 "$PLAN_SHA"` and `--submit`.  No submission was made in
this task.

## User-selected two-hour escalation dry run

```bash
python -u src/launch_k40_factorial_mip_screen.py \
  --replicate "R1=/campaign/one" --replicate "R2=/campaign/two" \
  --validated-start /absolute/path/to/re-realized-40-duty-giro.json \
  --mode escalation \
  --cell R1:CA:M360 \
  --cell R2:CS:M1440 \
  --campaign k40_factorial_mip_escalation_2h \
  --python "$(command -v python)" \
  --plan-out /new/output/path/k40_factorial_mip_escalation_2h.plan.json
```

Only explicitly selected primary cells are included.  Review/reconcile the
30-minute screen before preparing an escalation approval hash.

## Read-only MIP monitoring and reconciliation

```bash
python -u src/monitor_k40_factorial_mip_screen.py \
  --campaign-root /path/to/campaign --format tsv

python -u src/reconcile_k40_factorial_mip_screen.py \
  --campaign-root /path/to/campaign
```

Use reconciliation before any retry if `sbatch` may have accepted a job before
its ID reached `campaign.json`.  A fresh campaign name and new approval plan
are mandatory for definitively failed new attempts. If reconciliation uniquely
recovers an accepted job, remaining planned 30-minute cells can be submitted
without duplicating recorded cells:

```bash
python -u src/launch_k40_factorial_mip_screen.py \
  --resume-campaign /path/to/campaign \
  --approved-plan-sha256 "$PLAN_SHA" \
  --submit
```

Without `--submit`, this resume command is read-only and only reports
recorded/pending/ambiguous jobs.

Only results with a hash-bound completion sidecar are valid. MIP results record
the 40 actual supplied start-column hashes and carry
`route_space_scope=finite_augmented_snapshot_pool_only`; they are never global
route-space optimality claims.
