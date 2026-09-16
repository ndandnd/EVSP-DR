# Review §4 P1: fixed-pool MIP experiments

Submitted 27 jobs for F5 (item 7), F2/F4 (item 8), and F2/F5 (item 9); four existing seed-0 cells reused. All 27 were observed running with the required node exclusion/resources. No new column generation. Solver findings remain unresolved until endpoints arrive. See jobs.json and startup_verification.json.

| Review item | Frozen source | Fleet search allowance | Seeds | New jobs |
|---|---|---:|---|---:|
| 7, F5 | Six fresh k=15 pools from cumulative_budget_20260913, base arm | 3 hours | 0, 1, 2 | 18 |
| 8, F2/F4 | Original chain 5 k=31 pool | 12 hours | 0 | 1 |
| 9, F2/F5 | Original chains 1, 3, 4, 5 at k=32 | 3 hours | 0, 1, 2 | 8 new + 4 reused |

Historical one-hour runs allocated **30 minutes to the fleet-only stage**, then the remaining time to charging. These repeats allocate 3 hours to the fleet-only stage and 3.5 hours total (12 hours and 12.5 hours for item 8). Each second stage constrains fleet to be no greater than the first-stage incumbent and minimizes the original cost objective.

All new runs use default_partition, 8 CPUs, 24 GB, exclude scaglione-compute-01, and have private job/restart output directories. There is no arbitrary concurrency throttle. Slurm allocations are 4.5 hours or 13.5 hours, with 45 minutes of non-solver watchdog allowance. Existing jobs, held jobs, and other projects remain unchanged.

`manifest.json` binds source inputs, tariffs, source CG status, column journals, original ordered pools, original MIP-start records, comparison outputs, and execution commits. Every new endpoint must match the comparator's exact ordered-pool hash and deterministic greedy MIP-start record before publication. No previously improved incumbent is injected. Model remains 240 kWh, uniform 240 kW, no SOC floor or terminal requirement, unlimited shared charging capacity, covering, fleet cost 100000, charging-start fee 5, flat electricity tariff.

The isolated runner commit is `6830caa225856903d1157ef8587863c7ae21ad53`, based directly on `871d057e1067411f09581e37d78f7c1ca43f68bb`. Its only behavioral change is optional explicit Gurobi Seed. Omission preserves the solver default; Seed=0 explicitly equals that default. Patch and Git bundle accompany this record. Python compilation and native CLI parsing were checked; actual solver parameters are checked again at output.

Existing seed-0 results are retained only after matching input/journal hashes, ordered pool, MIP start, and 10800-second fleet allowance. Chain 1's existing job 336492 is reused while it runs; chains 3/4/5 have completed seed-0 endpoints. Chains 4/5 are retained for variance measurement despite now matching k=32. Their inclusion is explicit, not a claim that they still miss the target.

A same-pool incumbent is an upper bound on its integer optimum. A Gurobi bound/proof concerns that pool only. Several seeds that fail to find the target do not prove target infeasibility. Fleet matching does not imply an exactly-once trip partition or shared-capacity feasibility. These remain separate recorded flags.

`prepare.py` freezes and validates without submitting. `submit.py` is idempotent, records each accepted job immediately, and stops if sbatch acceptance is uncertain. `summarize.py` collects completed/reused cells without submitting or retrying jobs. Review's F5 headline trigger is at least one matching seed on at least four of the six fresh pools; report every seed regardless of result.
