# Overnight chain extension and decomposition — 11–12 September 2026

Submitted 12 September UTC (11 September evening EDT). All 174 research tasks use default_partition and exclude scaglione-compute-01. Scheduler admission determines actual concurrency. No held historical or EVSPV2G jobs changed.

| Work | CG cases | Dependent MIPs | Scope |
|---|---:|---:|---|
| Six inherited-column chains through k=15 | 37 | 37 | Continue P1 from k7, P2 from k9, P3 from k11, P4 from k10, P5/P6 from k11 |
| Ten decompositions of one 32-duty / 750-trip parent | 40 | 40 | Four eight-duty groups per split; contiguous, balanced and eight seeded random partitions |
| Recombine each split, then run global CG | 10 | 10 | Preserve component solution and try cross-group routes |

CG array 949623 and MIP array 949624 use throttle50. Remaining IDs and exact dependencies are in jobs.json. There are 96 scheduler records representing174 tasks, excluding integration checks.

## Scientific settings

Covering in CG and MIP; 240kWh battery /240kW charging; event2.5kWh/5min; zero reserve; no return-SOC floor or shared-station capacity. Flat tariff. CG route cost100000 + electricity +5 per charging start. This is the baseline cohort, not the harsher-capacity pilot.

New bounded inheritance treatment: select at most512 predecessor sequences, preferring longer then lower cost per trip; replay in the child graph for at most900seconds on8workers. Completed validated replays survive the deadline. Parallel completion can affect the accepted subset. No inherited duals, basis or certificate. This deliberately differs from earlier unlimited full-pool inheritance, which exhausted hours during initialization. Keep variants separate. P2k10 full-pool experiment remains untouched.

Component/warm CG budget4hours; combined32-duty CG2hours. Solver MIP budget1hour: first minimize fleet for at most1800seconds, then fleet <= validated incumbent and minimize electricity plus start fees using remaining optimizer time. Slurm wall allowance2hours includes loading/validation. Requeue enabled with unique job/restart output directories; a preempted Gurobi search restarts, not resumes its tree.

## Proof boundaries

Known GIRO duty membership defines the decomposition; GIRO schedules are not injected. A component sum gives a feasible upper bound only after selected-route physical validation, under this baseline model without shared charger capacity. It is not a full32-duty optimum or a general trip-only decomposition result. Global CG uses mapped component sequences plus singleton seeds; fleet/certification must be read from actual outputs.

Execution commit `a29992196acb74d02b8c7891be4061718889999f`; MIP runner871d057e1067411f09581e37d78f7c1ca43f68bb. manifest.json records input hashes, settings and parent paths; jobs.json records resource requests and dependencies; submission_verification.json checks effective partition/exclusion/requeue.

## Validation and collection

Three bounded-inheritance tests passed (default behavior, selection cap, deadline). Integration check949572 stopped CG after network build because its test-only90second budget was too short; MIP launch detected missing identity environment fields. Corrected worker binds runner commit and source-status/journal hashes. MIP check949621 completed both stages and physical replay. Its singleton-only28-bus result is an infrastructure test, excluded from research tables.

Collector outputs/meeting_20260910/collect_remote.py includes this campaign, attempts, CG timing and decomposed solutions. All87MIP cases added to the preemption registry. Preserve all raw attempt outputs, including preempted attempts.
