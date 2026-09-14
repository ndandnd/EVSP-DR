# Strict-capacity parallel follow-up design

This directory contains a reviewed, non-submitting design for twelve
independent one-to-four-hour cells.  It uses the dedicated capacity-aware
event driver at `309d98d266ebaf6b7e99543a67f8f2be5736874a`, whose ancestry
contains the station-specific tariff correction and bounded pricing
deadlines/checkpoints.  It does not use the generic trip-set-deduplicating MIP.

The first four cells use duty 13408 because its flat combined reference was
previously certified.  They compare reference and prefix-memo selection under
flat and peak12 prices.  This creates a short certification/equivalence gate
and a matched tariff test without repeating the censored duty-13406 pair.

The remaining eight cells form a matched E1-short k2 factorial.  Four arms
separate shared charger counts and PARX 60 kW, first at the historical 240 kWh
and zero-reserve abstraction and then at 236.44 kWh with the confirmed 15%
reserve.  The latter is an energy-bound sensitivity inside the same
constant-rate event graph.  It does not implement the nonlinear 18E1 charging
curve and is not labeled complete GIRO vehicle physics.  No 65% terminal rule
is imposed.

Every cell runs CG and then the dedicated two-stage capacity-aware finite-pool
MIP.  CG pricing proof, saved-pool integer proof, and route/shared-capacity
validation are reported separately.  A pricing deadline preserves an atomic
pool but has no terminal reduced-cost certificate.  Capacity cells receive a
four-hour allocation because earlier E1-short k2 traces locate the censoring
at the first nonzero-capacity-dual pricing calls; baseline and PARX-only cells
receive two hours.  The known certified k1 flat controls receive one hour and
the variable-tariff controls two hours.

The tooling has `validate`, `worker`, and read-only `collect` commands.  It
cannot submit jobs.  `collect` emits
`evsp-dr-strict-capacity-parallel-collection-v1`, with CG certification,
finite-pool MIP proof, and physical audits in separate fields.
`collect.py` is the stdout adapter for the research-register collector.  Its
top-level `cg`, `mip`, and `records` arrays include only artifacts whose
matching worker stage returned zero.  Incomplete CG results and unproven MIP
results are explicitly provisional; `workflow.attempt_progress` retains all
attempts, including interrupted ones.

The worker sets `GRB_LICENSE_FILE` to
`/share/apps/software/gurobi/gurobi.lic`, removes inherited Python and dynamic
library overrides, and fixes solver/native thread variables at one.  Each
child process has an outer watchdog 120 seconds beyond its internal solver
budget, leaving at least 180 seconds of the allocation margin for final status
writes.  `worker_status.json` is atomically updated between stages and records
manifest, command, logs, outputs, return codes, and watchdog state.  Attempt
directories are unique and there is no automatic retry.
Before any authorized launch, deploy a detached clean checkout of the exact
driver commit, retain all source hashes from `manifest.json`, copy this
directory to a unique campaign root, and rerun:

```text
python3 campaign.py validate --code-root /path/to/detached-309d98d2
CAPACITY_CODE_ROOT=/path/to/detached-309d98d2 python3 -m unittest test_campaign.py
python3 collect.py --campaign-root /path/to/campaign \
  --out /path/to/new-collection-snapshot.json > collection.stdout.json
```

The submission owner should translate each case's `slurm_time_s` into a
matching Slurm limit on `default_partition`, request one CPU and 24 GiB, launch
all twelve independently (concurrency 12), set `--no-requeue`, and include
`--exclude=scaglione-compute-01`.  The prepared manifest has
`submission_authorized=false`; no submission or scheduler mutation is part of
this artifact.
