# `run_exact_pool_mip --progress-dir` interrupt/relaunch audit

Date: 2026-08-21. All execution was local.

## Answer

`--progress-dir` is **observational only**. It does not resume the Gurobi search
tree, bound, node queue, basis, cuts, pseudocosts, or node state. It also does
not restore an incumbent into a relaunched process.

The relaunched solve rebuilt its ordinary deterministic greedy MIP start from
the column pool, started a new Gurobi model at node zero, and re-explored the
same early branch-and-bound trajectory.

Therefore large pool MIPs do **not** become partition-independent merely by
enabling `--progress-dir`. A protected scaglione partition remains useful when
an immediate high-quality MIP start is required.

## Runtime experiment

The physical toy used:

- k03 selection 2, 15 kWh / 10 minute grid, 300/300 physics;
- 1,055 unique physical columns over 71 trips;
- Gurobi 13.0.2 restricted local license, one thread;
- presolve, cuts, heuristics, and symmetry disabled to expose tree work;
- two-stage fleet-first solve;
- identical deterministic pool and parameters on both launches.

The harness waited for the fleet-stage progress record, delayed 0.2 seconds,
then sent `SIGINT`. The callback converted it into a graceful Gurobi
interruption and finalized all observational files.

| observation | first process | independent relaunch |
|---|---:|---:|
| status | INTERRUPTED | INTERRUPTED |
| actual Gurobi node count | 20 | 19 |
| incumbent fleet | 35 | 35 |
| fleet bound | 6 | 6 |
| MIP-start fleet | 35 | 35 |
| source result SHA | `2b78e522…e3c2ff` | same |
| source journal SHA | `7d4bb69d…3dddc` | same |

The near-identical 20/19-node trajectory and identical bound/incumbent show
that the second process repeated the first search prefix. It did not continue
from node 20.

An attempted relaunch using the original progress directory failed before
model creation with:

`--progress-dir already exists; choose a new path`

The successful relaunch therefore required a new empty directory.

## Metadata inspection

Both `final.json` files state:

```json
{
  "schema": "evsp-dr-mip-convergence-v1",
  "observational_only": true,
  "gurobi_tree_restart_supported": false
}
```

The progress payload contains:

- selected route indices and a route-vector hash for observed incumbents;
- sampled incumbent, bound, gap, node count, and solution count;
- stage transitions and scheduled observation times;
- source hashes, solver version, parameters, and physical-pool audit;
- final status metadata.

It contains no serialized Gurobi model/tree state. `run_exact_pool_mip.py`
constructs `MIPProgressRecorder` only after requiring that the directory not
exist. No code path reads a progress checkpoint into the new model. The
separate recovery utility only publishes censored reporting artifacts and also
labels them observational.

The progress recorder throttles sampled statistics to two-second intervals, so
these sub-second runs' `latest_statistics.node_count` remained zero. The actual
20/19 counts above come from each final solver summary, not the throttled
observation.

## Incumbent nuance

The same 35-bus MIP start appeared on both launches because the runner
recomputed the same greedy partition from the unchanged pool. It was not loaded
from the first progress directory. A different pool or explicit initial
partition would determine the second launch's start independently of progress
metadata.

## Evidence bindings

Producing code commit:
`3d5e11fa4fd4eaafdef678117447d4bbe109dbfb`.

| artifact | SHA256 |
|---|---|
| toy CG status | `2b78e5224024a8cf234fae8c8a677b7cbe68a025af61f150ede0e8f12de3c2ff` |
| toy column journal | `7d4bb69d0e96354b0034116b97654e521bd82e8d2a31aed46c449fafb533dddc` |
| first interrupted result | `309259c3a56cdbf89f58120e0e13d558800845c9a86a8d058f956130de301271` |
| relaunched result | `d343467e2bd41fac0d5254514ef6b39e6166e3cba8eeeb70a9fa9bcab4111b2c` |
| first progress final | `1bc171cd7ab70a70607e89a61fd56175958015d7c28cf934b7c3bded98d7099b` |
| relaunch progress final | `40c519bbc704b50b8d90695a47128071bd2837537d2ecf5ba180a2a70963e62b` |
| first signal event | `f767968de785d6dae10341027395431e356b02fa052c686889d87dc730b65fc7` |
| relaunch signal event | `23e63087428dfcd2061b961262967ecdd9437057ad7ec2d2e4fc3fa042f83d5e` |

No cluster jobs were submitted.
