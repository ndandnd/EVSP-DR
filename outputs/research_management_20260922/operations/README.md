# Scoped overnight operations — 22 September 2026, 05:21–05:23 UTC

SSH healthy; remote SCAGLIONE_RESOURCE_POLICY.md read before this check. No new allocation, restart, code-pin change, held-job release or stochastic-project mutation was performed by this operations task. Existing native salvage recoveries have meaningful final results; baseline and strict work continue.

## Queue and resource assessment

At05:21:46UTC,41 jobs run:39baseline graph tasks, the first baseline C3k33CG, and strict k19MIP668434. All44baseline graph allocations are accounted for:39RUNNING,5COMPLETED/0:0 (tasks16,23,38,39,40). The95pending solver rows are Dependency waits, with no DependencyNeverSatisfied reason observed. Held jobs are untouched. The full user queue has117pending display rows, representing149tasks after expanding the33-task historical held array; do not conflate display rows and array tasks.

The default partition isUP. At05:23UTC its aggregate CPU states report7,104idle CPU slots across225nodes, before eligibility, node exclusions, placement, competing work and account/QOS admission. This does not guarantee instantaneous capacity or suitable memory placement. The remote policy prescribes50or all-fewer independent cases and excludes scaglione-compute-01 for every CPU job. No observed policy or partition-level resource reason justifies throttling the coordinating task's proposed20–25independent8CPU/32Gfleet jobs below all cases. Those requests total160–200CPUs and640–800GiB if simultaneously admitted; Slurm decides placement.32G retains substantial headroom over the relevant observed saved-pool MIP peaks below7GiB. This check did not newly enumerate every account/QOS limit and does not claim none exist. No structure/parameter campaign was submitted here; another task owns it.

## Completed supplemental k15 salvage

Both jobs completed0:0 with unchanged pinned runner `c50e5f207869bac25507adfae90bb830a44039b7`. The native pre-gate and native final-MIP preparation agree on zero repaired/rejected columns and ordered pool identity. Source hashes match the prepared frozen augmented pools; no incumbent, node fixings, artificial values or dual solution was transferred from the failed dive.

| Case | Job | Accepted columns | Native gate s | Salvage buses / bound | Control buses / bound | Result |
|---|---:|---:|---:|---|---|---|
| C1k15 |728184|97,470|135.194|18 /15|19 /15|Open fleet gap; no target hit|
| C5k15 |728185|88,607|117.149|17 /15|16 /15|Open fleet gap; no target hit|

Both fleet and charging stages terminate at time limits. The final charging incumbents/bounds are1,028.712/672.943318(C1) and1,023.040/629.132425(C5). Reported overcovered trips are88/83; selected-route physical replay passes, while duplicate removal and shared capacity remain unvalidated. There is no global integer or continuous-cost proof. The fact that an enlarged C5pool returns17instead of control16under its remaining budget does not imply its pool optimum is worse; these are incumbent-search outcomes.

Scientific residual solver allowances were1,852/3,266s. Recorded solver-phase runtime is1,857.081954/3,268.530956s; failed-dive wall plus this phase totals7,204.943705/7,202.230225s, preserving small termination overruns. The extra pre-gate135.193932/117.149222s and final-MIP external overhead137.891457/117.154824s are separately recorded. These are supplementary recoveries, not replacements for the original failed paired treatments or a strict end-to-end7,200s claim. Scheduler elapsed is35m48s/58m39s; MaxRSS7,082,476/6,547,560KiB. No retry or memory increase is indicated.

Full copied logs/results, native gates and receipts live under `salvage/`; `proof_lines.json` indexes exact fleet and charging endpoints. `salvage_endpoints.csv` is the compact editable table.

## Strict k19: graph build exhausted the CG allowance

Parent global prefix19 means **331trips /11reference18E2duties**, not19buses. CG668433 completed at the scheduler but scientifically stopped incomplete/uncertified. Graph build took**16,292.771283s (271.546188min)** and total CG runtime**16,330.941709s (272.182362min)**. Graph construction was included in the nominal4hscientific allowance and overran it before any pricing iteration: `iterations=[]`.

Initial/final pool8,397 comprises8,343physically replayed parent routes plus54new singletons. Zero artificials and fractional route weight64yield weighted RMP6,400,391.890434. Neither quantity is a full-model bound or target11achievement. CGMaxRSS6,301,824KiB; larger memory is not the relevant bottleneck. Physics remains239.01kWh,15%reserve,PARX60kW/others240kW,covering,unlimited shared capacity, reserve-only terminal constraint. Published result hash matches CG_COMPLETE; declared pool hash matches the receipt. This collection did not independently scan/replay the whole strict pool.

Dependent MIP668434 was RUNNING at05:23UTC, already in charging search; no final result/COMPLETE artifact existed in this collection. Preserve its attempt and await its endpoint. Do not queue a larger strict prefix or silently increase scientific budgets from this observation. Backlog before any next strict CG: prepare and validate the graph separately, then reuse that completed graph with the approved identical physics, source/input identity and genuine parent-column checkpoint. Keep graph preparation wall time and the solver allowance separately reported; do not repeat a multi-hour graph build inside the CG wall cap. Register the revised setup separately from this original budget experiment. This is preparation/design authorization, not permission to autoqueue k20; preserve the running k19 MIP and current pins.

## Validation and next triggers

Run `python3 outputs/research_management_20260922/operations/audit.py`:38checks pass, covering copied hashes, immutable source identity, native gate counts/pool identity, scientific arguments, pinned code, recorded budget sums, full-log endpoint matches and strict initial-pool accounting. This is an artifact audit, not a new physical simulator run. `collection.json` retains remote paths/hashes, exact resource queries and timestamps; `verified_summary.json` contains independent scope fields. `initial_policy_queue.txt` preserves the policy and scoped scheduler evidence.

Next monitor should collect strict668434 when terminal and meaningful baseline CG/graph changes, and avoid rerunning completed salvage cases. The separately owned [matrix campaign](../mip_structure/README.md) now has five preparations and 25 dependent trials729439–729468; use its manifest, jobs and scoped collectors. The [capacity representation pilot](../charging_column_structure/README.md)729675 is already completed and collected, with finite-pool fleet3 for all three equivalent forms; no repeated polling is needed. The existing four-hour heartbeat was updated in place, preserving its identity, cadence and quiet unchanged checks. Current user-supplied no-Slides instructions take precedence over earlier dated policy records; update the existing Doc tabs only. [Heartbeat before/after and verification](heartbeat_update.md) records the integration. This subtask did not edit live artifacts or launch either new campaign.
