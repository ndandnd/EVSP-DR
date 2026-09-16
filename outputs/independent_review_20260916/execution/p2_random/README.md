# P2 item 12 / F5 — random trip-group continuation control

**Submitted.** Graph array **341352** has all 14 tasks eligible concurrently; 14 dependent CGs and 14 dependent MIPs complete the chain. This pilot tests whether warm continuation works when intermediate groups do not follow GIRO duties.

Use the same **364 trips** as original chain 1 at k15. Shuffle the trip IDs with fixed seed **20260916**, then cut the shuffled list into groups with exactly the original chain's per-duty trip counts. Solve cumulative groups 2–15 with full-pool inheritance. The final input has exactly the same physical trip attributes as the original C1 k15 instance.

**Stage number is not a bus target.** For example, stage 2 has 61 trips drawn from 12 GIRO duties. Only the final stage has the original 15-duty reference. No GIRO duty columns are inserted. The initial stage uses singleton columns; each later stage inherits all previous-stage columns and initializes newly added trips with singletons.

| Setting | Frozen treatment |
|---|---|
| CG execution | `a0e0bb7681c8451e3cbbbfa06aef390026d9af4b` |
| MIP execution | `871d057e1067411f09581e37d78f7c1ca43f68bb` |
| Battery / charging | 240 kWh / uniform 240 kW |
| Reserve / final SOC / station capacity | 0 / none / unlimited |
| Objective | 100,000 per bus + flat electricity cost + 5 per charging start |
| Master | Set covering |
| CG budget | Four hours per stage; all inherited columns, no inheritance cap |
| Pricing | Event representation, 2.5 kWh / 5 min, 30 columns per iteration, RC tolerance 0.0001 |
| Final MIP | One hour total; 30-minute fleet stage, then charging with fleet ≤ incumbent |
| Partition | `default_partition`, requeue enabled, exclude `scaglione-compute-01` |
| Parallelism | All 14 graph preparations may run simultaneously. CG stages have true previous-stage dependencies. Each MIP depends only on its own CG. |

`trip_group_assignment.csv` preserves the full randomization, including the original duty label for audit only. `inputs/manifest.json` records source hashes, group sizes, stage inputs, final-set equivalence and the exact generation rule. Trip count sequence is **61, 72, 87, 104, 150, 177, 194, 211, 225, 246, 263, 296, 310, 364**.

## Comparison and time accounting

Compare the final random-group pool and MIP with original C1 k15 warm and equal-budget fresh results. Report intermediate stages by trip count, not target buses. This is one order/one chain pilot, so it cannot establish general effectiveness or a statistically independent effect.

`collect.py` separates graph, CG-worker and MIP-worker elapsed time and sums every recorded attempt, including preemptions and failures. It also reports the solver's own `wall_s`, iterations, stopping reason, pricing certificate, pool fleet proof and physical checks separately. These totals are computation-time sums, **not parallel calendar duration**. Missing worker end times remain unknown until reconciled with Slurm accounting. Allocation CPU-hours and actual CPU usage are different quantities.

The old warm comparator has a different earlier history and potentially a different graph-preparation commit; retain its exact provenance. A strict causal performance comparison would require replaying a GIRO-grouped chain under the same new harness as a second arm. This pilot is a screening control, not proof that random grouping is better or worse.

## Deployment and validation

The campaign is already prepared at `/home/nc437/ladder-lite/random_trip_groups_c1_20260916/`. Its frozen launch manifest and validation receipt are saved locally as `manifest.json` and `validation.json`. Submission is complete; do not rerun preparation or submission. `prepare` clones the pinned CG execution tree and freezes data; it does not submit jobs. `submit` prints/hashes the cluster resource policy, requests all 14 independent graphs, and preserves the 13 true previous-stage CG dependencies. It refuses to duplicate an existing campaign.

The adapted worker retains identity checks, attempt directories, source journal hashes, validated cache hashes, checkpoint resume on preemption, two-stage MIP and preemption-study registration. MIP preemption restarts the tree, not a claimed saved branch-and-bound state. Existing historical held jobs are untouched.

Local checks completed: input nesting and final attribute equality; 14 stage hashes; first-stage CLI contains no parent-pool argument; later-stage CLI retains unlimited full-pool inheritance; campaign syntax compiled. Remote preparation and validation passed. All 14 remote input hashes, code pin, clean tracked source, tooling hashes and 13 parent edges were checked. A **2,101-variable Gurobi LP** solved optimally, excluding the size-limited-license problem for this environment. Scheduler accepted all 29 submission records / 42 tasks. All effective partitions, node exclusions and CG/MIP dependency edges were verified; see `submission_verification.json`.
