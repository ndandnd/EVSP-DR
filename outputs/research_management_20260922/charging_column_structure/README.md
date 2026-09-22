# Charging columns: compact intervals, capacity rows and an exact fixed-pool pilot

A route column already contains a complete charging schedule. Its starts and ends are **data**, not variables chosen again by the final route-pool MIP. Compact intervals reduce storage; making the MIP smaller requires an equivalent representation of the shared-capacity rows. On one real strict-physics k3 pool we verified a substantial exact reduction, without optimization or changing the original capacity semantics.

## Concrete fixed-pool result

Source: `outputs/meeting_20260910/giro_k23_capacity_duals/results/e1_short_k3/pool.jsonl`, 321 routes, 35 trips, SHA256 `714cec263633e3fa8ac178052b3be595097c1990769503afcac86e3d5d11a845`. Its native model is `run_giro_small_cg.py`: strict18E1 physics, fleet objective, and conservative one-minute plug occupancy. This is not the separate historical capacity-free baseline panel.

| Equivalent representation | Binary route variables | Continuous occupancy variables | Full model rows, including 35 trip rows | Full nonzeros |
|---|---:|---:|---:|---:|
| Original active one-minute capacity rows | 321 | 0 | 1,789 | 44,844 |
| Merge identical capacity rows | 321 | 0 | **303** | **10,232** |
| Endpoint differences + cumulative occupancy variables | 321 | 272 | 307 | **7,714** |

Capacity block B alone changes from **1,754 rows / 42,036 nonzeros** to **268 rows / 7,424 nonzeros** using identical supports and identical RHS. The endpoint form has 4,364 route coefficients and 542 auxiliary linking coefficients. Bounds on occupancy variables enforce capacity without additional explicit inequality rows; table counts follow that formulation. It has more variables, so fewer nonzeros do not guarantee faster solving.

All original minute rows were reconstructed exactly from compressed rows and from endpoint differences. An additional **1,000 rational route-selection checks** passed. No optimizer was called during those algebra checks. These initial checks establish matrix equivalence and size reduction. The subsequently completed solver pilot is reported below; it is too small for a general speed claim.

**Safest pilot:** original rows versus identical-row compression first; optionally compare the endpoint auxiliary form. `proposed_pilot.json` proposes the same pool/order/objective, cover sense, seed0, four threads, 300 MIP seconds plus a separate30-second LP diagnostic per variant, and identical starts. The runner was approved, deployed and submitted as **job729675**, now **COMPLETED, exit0:0**; see `pilot/deployment_manifest.json` and the submission receipt for current scheduler status. Preserve original-B validation of every chosen solution and compare construction, presolve, root LP, bound/incumbent, work/nodes and elapsed time. Native presolve may already remove much of the redundancy.

## The column and the combined master matrix

For route r, let `A[i,r]=1` if it serves trip i. For station s and time segment q, let `B[s,q,r]=1` if that vehicle occupies a charger during the segment. The route selection model is

```
min  Σ c[r] x[r]
     A x >= 1       (or =1 for exact partition)
     B x <= C
     x[r] ∈ {0,1}.
```

The full column is `[A[:,r]; B[:,r]]` plus its objective coefficient—not just its trip set. A route's charging decisions can affect c and B even when A is identical. With charger capacity, retaining only the cheapest column per trip set can delete an essential alternative schedule; use complete relevant row coefficients and objective semantics for any dominance rule.

In CG the same route variables are continuous. If trip duals are pi and capacity-row duals are gamma, the reduced cost is `c[r] − A[:,r]'pi − B[:,r]'gamma` (for the minimization model's `<=` rows, gamma is nonpositive). Capacity affects pricing through the charging schedule, not merely route feasibility after CG ends.

## Exact fixed-interval endpoint representation

Take the sorted union of charging endpoints at each station. With half-open occupancy `[start,end)`, every elementary segment between adjacent endpoints has constant route membership. One capacity row per distinct membership/RHS is enough. At a shared time, disconnections and connections are applied together; adjacent nonoverlapping visits do not conflict merely because they share an endpoint.

Let D have `+1` where a route's occupied interval begins and `−1` where it ends. A route with several visits contributes several endpoint pairs. First union overlapping/adjacent intervals of the **same route at the same station**, to match the implementation's set-membership coefficient, rather than double-counting one vehicle. With u denoting total occupancy immediately after an event,

```
u[s,k] − u[s,k−1] = Σ D[s,k,r] x[r],   with initial occupancy zero
0 <= u[s,k] <= C[s,k].
```

This is an exact extended formulation for fixed intervals; u can remain continuous even when x is binary. Eliminating u yields `B = cumulative_sum(D)` and the original occupancy inequalities. **Simply imposing `D x <= C` is wrong:** D measures changes, not the number of occupied chargers. If capacity itself changes, include capacity-change events; merge rows only when their RHS also matches.

Important distinction for our code: the current one-minute model reserves an entire minute whenever a session overlaps it. To retain **that exact model**, build endpoint intervals from the route's union of occupied minute bins, then compress. Replacing rounded bins by the original physical timestamps is generally a relaxation, not an equivalent optimization.

Example: visits `[0.1,0.4)` and `[0.6,0.9)` are physically disjoint but both touch minute bin `[0,1)`. With one charger, physical endpoint rows permit both; the conservative minute row forbids both. Our native compression deliberately preserves the latter. Coarse average occupancy or energy rows are another different model.

## Editable toy example

One charger, three trips, five route columns:

| Route | Served trips | Fixed charging intervals at S |
|---|---|---|
| r1 | 1 | [0,2) |
| r2 | 2 | [1,3) |
| r3 | 3 | [2,4) |
| r4 | 2 | [4,5) |
| r5 | 1,3 | [0,1), [3,5) |

The five B rows are:

```
segment   r1 r2 r3 r4 r5
[0,1)      1  0  0  0  1
[1,2)      1  1  0  0  0
[2,3)      0  1  1  0  0
[3,4)      0  0  1  0  1
[4,5)      0  0  0  1  1
```

For example, `{r1,r2,r3}` covers all trips but has two simultaneous users; `{r1,r3,r4}` covers all trips with no overlap, including the r1→r3 endpoint at time2. See editable `toy_routes.csv`, `toy_trip_A.csv`, `toy_capacity_B.csv` and `toy_endpoint_D.csv`. Exact-rational tests cover all32 binary selections, all243 half-step fractional selections, and2,000 randomized fractional cases across additional interval sets. The tiny toy actually has more D nonzeros than B; compaction benefits depend on the data.

## Why energy and endpoints do not describe every capacity question

A charger-count constraint concerns **simultaneous plugged vehicles**. Two buses each charging for30 minutes within an hour can use the same total energy whether they charge sequentially or simultaneously; an hourly energy budget cannot tell those schedules apart. Setup, full-battery connected waiting and other zero-power connection periods may still occupy a plug. Preserve the relevant connection interval, not only the interval of positive power.

Energy is `E = integral(P(t) dt /60)` when time is in minutes. Under a constant-power assumption, E and active duration determine average rate; under SOC taper they do not determine the entire power trace. The strict profile integrates the SOC-dependent curve, including battery limit, reserve, setup and idle consumption. Start/end plus energy alone cannot establish its rate feasibility.

Fixed plug-occupancy endpoints **are sufficient for charger count** if the whole connected interval is known. They are **not sufficient for instantaneous site kW** unless the power profile is specified. For piecewise-constant power, include every power-change breakpoint; for piecewise-linear power, include slope-change breakpoints and evaluate the resulting aggregate linear segment endpoints. Unknown SOC-dependent traces require physical reconstruction. A site-power block would have coefficients in kW rather than B's binary plug counts.

## What the inspected implementation actually does

Inspected worktree: `.codex-work/strict-packed-20260921`, HEAD `35770aae2c08e7d5a356cc3b673e67608e5b1036`; file hashes are in `provenance.json`. No local symbol named `NativeGurobiCapacityMaster` or `charge_slot` was found. The actual classes/helpers below implement the relevant native Gurobi paths.

| File and line | Verified behavior |
|---|---|
| [event_pricer_network.py:798](/Users/nadan/Documents/projects/demandresponse/.codex-work/strict-packed-20260921/src/event_pricer_network.py:798) | Saves `stations,cst,cet,kwh` for continuous and expanded-grid schedules; keeps timing but can reduce realized energy using continuous SOC. |
| [event_pricer_network.py:180](/Users/nadan/Documents/projects/demandresponse/.codex-work/strict-packed-20260921/src/event_pricer_network.py:180) | `conservative_capacity_rows` marks every overlapping station/grid interval; default one minute, half-open overlap with tolerance. Coefficients are0/1, not fractional overlap durations or energy. |
| [run_capacity_speed_event_cg.py:181](/Users/nadan/Documents/projects/demandresponse/.codex-work/strict-packed-20260921/src/run_capacity_speed_event_cg.py:181) | Unions a route's capacity rows using expanded-grid charging windows. |
| [run_capacity_speed_event_cg.py:245](/Users/nadan/Documents/projects/demandresponse/.codex-work/strict-packed-20260921/src/run_capacity_speed_event_cg.py:245) | `ExactCapacityMaster` creates every finite station/minute row before pricing; native `gp.Column` inserts trip and capacity coefficients; returns capacity duals. |
| [event_pricer_network.py:270](/Users/nadan/Documents/projects/demandresponse/.codex-work/strict-packed-20260921/src/event_pricer_network.py:270) | Capacity-aware pricing includes capacity-grid breakpoints and subtracts the duals of the selected occupied rows. |
| [run_capacity_speed_event_cg.py:658](/Users/nadan/Documents/projects/demandresponse/.codex-work/strict-packed-20260921/src/run_capacity_speed_event_cg.py:658) | Its final capacity MIP explicitly adds nonempty station/minute rows, then fleet/cost objectives. This event path is constant station power, not documented nonlinear taper. |
| [giro_partille_physics.py:122](/Users/nadan/Documents/projects/demandresponse/.codex-work/strict-packed-20260921/src/giro_partille_physics.py:122) | Strict single-bus replay integrates taper and uses60kW at PARX. The charging-window helper near line198 distinguishes setup, active power, connected duration and idle. |
| [giro_weighted_pricing.py:27](/Users/nadan/Documents/projects/demandresponse/.codex-work/strict-packed-20260921/src/giro_weighted_pricing.py:27) | Strict capacity occupancy is `setup_start_min` through `connection_end_min`, rounded to one-minute plug rows; PARX excluded from finite opportunity counts. |
| [run_giro_small_cg.py:98](/Users/nadan/Documents/projects/demandresponse/.codex-work/strict-packed-20260921/src/run_giro_small_cg.py:98) | The strict small-cohort LP and final MIP use the same active minute incidence, with binary/continuous variants and fleet objective. This is the321-route pilot's model. |
| [run_giro_small_cg.py:76](/Users/nadan/Documents/projects/demandresponse/.codex-work/strict-packed-20260921/src/run_giro_small_cg.py:76) | Its route key includes occupancy, not merely trips. |
| [run_exact_pool_mip.py:266](/Users/nadan/Documents/projects/demandresponse/.codex-work/strict-packed-20260921/src/run_exact_pool_mip.py:266) | Separate historical generic pool MIP deduplicates by trip set; line2982 explicitly reports cross-route charger capacity unvalidated. Its “strict” physical gate must not be mistaken for a capacity-aware master. |

## Gurobi implementation choices and CG caveat

Gurobi accepts SciPy sparse matrices through `addMConstr`. `PreSparsify` and presolve may reduce model nonzeros, so compare the actual presolved models; merely giving the same coefficients through CSR is not a stronger formulation. [Official matrix API](https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html), [official parameter reference](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html).

For a **fixed-pool MIP**, omitted essential capacity rows can instead be separated as lazy constraints: validate candidate integer solutions, enable `LazyConstraints`, and add violated original capacity inequalities. Gurobi's `cbCut` user cuts must not exclude integer solutions feasible under the original model; use the proper lazy mechanism for omitted defining constraints, not only user cuts. This is a later pilot, not the safest first compression test. [Official callback documentation](https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html).

For **column generation**, a fixed-pool row equivalence may disappear when a new route is added. A new interval can split a previously identical segment. Static row merging based only on the current pool is unsafe unless it is updated and the dual cost seen by every candidate route is correct. With full minute rows merged only over current columns, aggregate master duals generally do not define the costs of future columns that distinguish those minutes. Row generation requires capacity separation plus reoptimization and pricing with all active row duals; no final certificate may ignore missing violated rows or capacity-aware pricing. Endpoint representation during CG therefore needs an explicit evolving-row and dual mapping design. Start with fixed-pool compression.

## Files and reproduction

`interval_equivalence.py` and `audit_native_pool.py` run with standard Python only. `test_results.json` records toy checks; `native_compression_results.json` records the real matrix sizes. `native_grid_to_unique_rows.csv` maps every original capacity row to its exact representative; `native_compressed_rows.csv`, `native_rounded_intervals.csv`, `native_endpoint_difference.csv` and `native_route_columns.csv` are editable construction inputs. `proposed_pilot.json` specifies the solver comparison. `provenance.json` hashes sources and artifacts.

This work changes no production code, source pool, CG certificate, solver result, live Doc or Slide. The algebra checks made no optimizer calls. Subsequently approved job729675 is recorded in `pilot/submission_receipt.json`; its solver results are separate from those checks.


## Sparsity percentage versus work

The full matrices have **92.191116%**, **89.480070%** and **95.762726%** zero entries for original, merged-row and endpoint-auxiliary forms, respectively. Row merging reduces absolute nonzeros by77.18% even though the percentage of zeros falls, because the matrix dimensions shrink faster. Percentage sparsity alone does not predict solver work.

## Pinned solver runner and publication input

`pilot/run_pilot.py` and `pilot/worker.sub` are committed and deployed together; `pilot/execution_commit.txt` records the execution commit. All three formulations use Seed0, Threads4, MIP300s, LP diagnostic30s, and no explicit MIP start. The variant order is fixed (original, merged, endpoint); each model is newly constructed and the separately optimized LP is not supplied as a MIP start. Process/license/cache warmup and host variability remain possible confounders. One small case cannot establish a general or causal speed advantage. Actual presolve/root statistics are extracted from each log; LP and MIP wall times are separate from build and total elapsed time.

The job requests default_partition,4CPUs,4GiB,30minutes, excludes scaglione-compute-01 and preserves a distinct output directory for each requeue attempt. Current Unicorn policy was read and copied into `pilot/SCAGLIONE_RESOURCE_POLICY.read_20260922.md`. The runner pins the shared Gurobi license environment without recording license contents. Root approval was obtained before submission.

The publication input is `pilot/inputs/pool.jsonl.gz`, a deterministic gzip with mtime0 (194,067bytes). The uncompressed hash remains714cec263633e3fa8ac178052b3be595097c1990769503afcac86e3d5d11a845. Retain raw JSONL locally/remotely for execution; publish only the gzip. From the pilot directory, restore it with `python3 -c "import gzip,pathlib; p=pathlib.Path('inputs'); (p/'pool.jsonl').write_bytes(gzip.decompress((p/'pool.jsonl.gz').read_bytes()))"`. Verify SHA256 before execution; the runner also refuses a different pool hash.


## Completed native Gurobi pilot — job729675

The pinned worker completed on snavely-cpu-02 with scheduler stateCOMPLETED, exit0:0, sixseconds elapsed. Executioncommitf37c6ce4c6e0a7ccffd3719a633bacd40b9a351c, Gurobi12.0.3, Seed0, Threads4. All LP relaxations have objective3 within numerical tolerance; all MIPs prove fleet3 in one node and all selected routes pass the original full trip/capacity matrix. The three-model process took1.8683seconds including build, LP diagnostics, solves and artifact work. The300-second MIP limits were allowances, not consumed runtimes.

| Formulation | Build/update s | LP optimize wall s | MIP optimize wall s | Presolved rows / columns / nonzeros | Fleet incumbent / bound |
|---|---:|---:|---:|---|---|
| original_minute | 0.2072 | 0.0495 | 0.3392 | 82 / 290 / 5009 | 3 / 3 |
| identical_rows | 0.0283 | 0.0151 | 0.0587 | 108 / 292 / 5596 | 3 / 3 |
| endpoint_difference | 0.0424 | 0.0152 | 0.1002 | 131 / 403 / 6779 | 3 / 3 |

Original and merged models select zero-based columns199,274,315; endpoint auxiliaries select199,274,316. These are equally valid optima, not a disagreement in the model. Native presolve reduces the original model to fewer nonzeros than either alternative's presolved model, despite its larger input. This is why input percentage sparsity alone is insufficient. The observed timings favor row merging on this tiny case, but fixed order, first-model setup/cache effects and a single subsecond solve prevent a causal or general speedup claim.

Editable summary: `pilot/results.csv`. Full immutable collection: `pilot/collections/20260922T053756Z/attempts/729675_r0/`, including the manifest, per-variant resultJSON, Gurobi LP/MIP logs and `.mps.gz` models. The collection receipt hashes every downloaded artifact and includes Slurm evidence. `MaxMemUsed` is process/environment cumulative and must not be read as an isolated per-variant memory comparison; scheduler MaxRSS is coarse for this very short job.

The execution commits are retrievable from branch `codex/charging-column-structure-20260922` in the shared repository, pointing to `f37c6ce4c6e0a7ccffd3719a633bacd40b9a351c`. Commit `3c3cdd2aae16d66d7337058dd5048389cc4f9fcc` adds only `pilot/run_pilot.py` and `pilot/worker.sub`; commit `f37c6ce4c6e0a7ccffd3719a633bacd40b9a351c` changes only that worker. No bundle is needed. See [RESULTS.md](RESULTS.md) for the concise result and proof links.
