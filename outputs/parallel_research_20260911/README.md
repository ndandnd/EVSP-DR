# Parallel research expansion

## Submitted

| Campaign | Cases | CG execution | Purpose |
|---|---:|---|---|
| Fresh set covering | 75 | 810454, array 0–74, concurrency 50 | Complete six chains k=2–15; reuse existing nine results |
| Inherited columns P1/P2/P4/P6 | 36 | Four independent k=2–10 pipelines | Complete warm-start evidence beyond P3/P5 |

The fresh campaign holds the original CG implementation, inputs, physics, tariff, pricing selection and eight-hour budget fixed; changes partitioning to covering. Prioritized k=5,8,10 on P2/P4/P6 first. Source input hashes and cached-network identities are checked before each case. See [submission and exact plan](cover75/submission.json).

Fresh jobs request 1 CPU each. Initial memory was 32 GiB; measured-use overrides reduce k≤5 to 8 GiB and k=6–10 to 16 GiB, retaining 32 GiB for k=11–15. The exact per-task scheduler changes are in cover75/resource_override.json. The prior largest observed k=15 CG peak was approximately 11 GiB. All Scaglione nodes are excluded from this fresh CG array to reserve their RAM for MIPs; compute-01 remains excluded from all CPU-only jobs. Default capacity observed before launch exceeded 7,000 idle CPUs. This is shared capacity, not an entitlement to fill every node.

Scaglione CPU nodes had approximately 13–30 GiB unallocated RAM despite many idle CPUs. Measured MIP peak RSS: P5 k5 0.46 GiB, P5 k6 1.58 GiB, P3 k10 4.11 GiB. Requesting 16 GiB for comparable small MIPs provides substantial headroom and enables admission; historical held jobs remain untouched.

## Reporting

New campaign roots are in the remote evidence collector. Preserve all raw statuses, logs, inputs and output hashes. Scheduler completion, CG pricing certificate, finite-pool MIP proof, physical replay and target attainment remain separate. A queued job has no numerical result. Downstream MIPs depend on their own case, not all CG cases finishing.

## Decisions from the new evidence

1. Paired covering versus partitioning across all six chains, with exact RMP objectives and stop reasons.
2. Fresh versus inherited columns on the same chain and k, including import time.
3. Fleet and charging results after validated two-stage MIP; separate time-limited gaps.

See [gap matrix](GAP_MATRIX.md) for existing controls and figure opportunities.

## First completed recovery

Terminal-energy MIP retry 810459 completed all three tariffs. The common aggregate return-energy floor is 280.7833253 kWh. All joint solutions use five buses, proved within the saved pool; the charging stage also reached finite-pool optimality.

| Peak | Fixed-duty grid objective | Joint grid objective | Reduction |
|---|---:|---:|---:|
| 08:00 | 279.899878962 | 261.5697635682 | 6.55% |
| 12:00 | 332.025524540 | 332.0255245396 | 0% |
| 18:00 | 232.249933738 | 217.5050887809 | 6.35% |

Objective includes electricity plus charging-start fees. Equal minimum return energy does not imply equal realized return energy. Source root: `/home/nc437/ladder-lite/terminal_energy_fair_mip_retry_5cdb813_20260910`, commit `5cdb8138c29faef9d5bf949175cb1e815a0b4220`. Comparison hashes in peak order: `c5bd813c9f2c8f727150be77a95b3f80db2990a83b0fee6d6b0944bbdf87335d`, `d2f8ddfce55c5dfcc1680ba0c2204fd9305711f8c454667d8b0a220028fbc008`, `dcff1c7807b643cfad4b0642d506bc30c98162a98ddbde4a43b2e567afb47037`. Original failures remain retained; no full-model optimality claim.

Fresh75 downstream is submitted: freeze array **810587** uses `aftercorr:810454`; small MIP array **810588** and large MIP array **810589** each use the corresponding freeze task. Small k≤10 requests 16 GiB; k≥11 requests 32 GiB. Both MIP arrays allow eight concurrent tasks, subject to scheduler admission. No all-CG-completion barrier.

All 36 warm-chain cases now have their own freeze and MIP jobs, listed in `nested_warm_multichain_p1246_k2_10_20260910_ecb60c1/freeze_mip_jobs.tsv` on Unicorn. Completed upstream cases are accepted only after scheduler COMPLETED plus saved result/journal verification; active predecessors retain their dependencies.

New figures: [fresh formulation matrix](figures/fresh_partition_vs_cover_fleet_matrix.png) and [fresh versus inherited chain 3](figures/fresh_vs_inherited_covering_chain3.png), with editable explanations and proof scopes in [figure provenance](figures/FIGURE_PROVENANCE.md).

## Capacity timeout recovery queued

Six fresh matched CG reruns: array **811181**, tasks **5,7,9,11,13,15**, eight-hour algorithm budget, nine-hour scheduler allocation, 1 CPU/24 GiB. Original code and arm physics are preserved; old driver saved no resumable pools. New root: `/home/nc437/ladder-lite/capacity_speed_pilot_20260910_timeout6_rerun_7d38ef`. Dependent two-stage MIP array **811182**, corresponding-task dependencies, 8 CPU/16 GiB, 1500-second total solver budget and 750-second first stage. GPU compute-01 excluded. These are budget-extension experiments, not checkpoint resumes. Original timeouts remain in the record.

At the 23:22 EDT snapshot, fresh array810454 had **50 running** and **25 pending** tasks. Sample log task3 reached Gurobi LP optimization with empty stderr. See cover75/startup_check.json. This validates startup only.

## Default-partition MIP migration

User-authorized overnight trial: fresh75 MIPs now use **812766/812767** on default_partition (combined concurrency50), one-hour solver budget with two-hour allocation and unchanged per-case freeze dependencies. Warm and capacity MIPs remain on Scaglione. Previous pending arrays were administratively cancelled before any solve. See [preemption study](../research_register/preemption_study/README.md) and [exact final migration audit](../post_meeting_20260910/warm_multichain_p1246/records/fresh75/default_mip_migration_a02.json).
