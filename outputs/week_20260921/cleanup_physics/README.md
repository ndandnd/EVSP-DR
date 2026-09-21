# 21 September — duplicate cleanup and corrected k5 charging

The old one-bus graphic used 240 kWh, constant 350 kW charging, zero reserve and no charger-count constraint. These are **not** the documented parameters for the five `134xx` duties. The replacement below reoptimizes charging on the saved trip sequences under the actual 18E1 battery and charging curve, with charger counts enforced across the five-bus cohort.

## Corrected figure and evidence

[Three-panel PNG](one_bus_k5_joint_matched.png) · [PDF](one_bus_k5_joint_matched.pdf) · [editable event CSV](one_bus_k5_joint_matched.csv) · [independent validation](joint_validation.json) · [metrics](joint_figure_metrics.json).

Panels display original GIRO duty 13414 and the bus sharing the most trips with it in each saved solution. Original: 12 trips / 9 charging starts; saved-CG fee 0: 13 trips / 9 starts; saved-pool fee 5: 12 trips / 6 starts. Other buses remain in the optimization and capacity audit. Trip IDs are source dataset IDs. Connections are schematic, not geographic trajectories. Panel identity, axes and legend are the only embedded text; the caption belongs in editable slide/document text.

**Matched parameters:** group 18E1; usable and initial battery 236.44 kWh; 15% minimum SOC (35.466 kWh); opportunity charging powers 371.5, 357, 342.5, 328, 313.5, 299, 284.5, 270, 150, 120 kW in successive 10% SOC bands; depot PARX 60 kW; at least three active charging minutes; zero setup; 0.1 kW idle draw; allowed opportunity sites 2190L and 4808, one charger each. PARX is configured at 60 kW although none of these resulting charging sessions uses it. No second application of battery efficiency is made. The 65% recharge target is not misread as a hard terminal minimum.

Each new bus is matched one-to-one to an original duty by maximum total trip overlap. Its terminal energy must be at least that original duty's replayed terminal energy. All ten floors bind: original and both new fleets end at **250.7183243524 kWh**. This replaces the old comparison's differing per-bus and aggregate energy accounting.

| Five-bus schedule | Trips served exactly once | Charging starts | Electricity | Start fees | Total charging objective |
|---|---:|---:|---:|---:|---:|
| Original GIRO, documented-curve repricing | 62 | 52 | 230.644788 | 0 or 260, depending on comparison | 230.644788 or 490.644788 |
| Recharged saved CG trip sequences, fee 0 | 62 | 42 | 158.706820 | 0 | 158.706820 |
| Recharged saved pool trip sequences, fee 5 | 62 | 30 | 183.714825 | 150 | 333.714815 |

Values are **synthetic tariff cost units**, not a verified operator invoice or asserted currency. The original curve-based repricing is a reconstructed power trace, not a measured trace. It integrates the documented maximum taper curve within recorded windows; original kWh and movement activities are retained. Previous uniform-power repricing differs slightly. The same `peak08_h26.csv` is used for both new arms.

The fee-0 MIP took 0.55 seconds and met its 0.01% gap tolerance (actual gap 0.00565%). The fee-5 MIP stopped at 120 seconds with a 0.6504% gap. Neither proves globally optimal routing or charging. Full readable Gurobi logs: [fee 0](saved_joint_fee0.gurobi.log), [fee 5](saved_joint_fee5.gurobi.log). Models: [fee 0 LP](saved_joint_fee0.lp), [fee 5 LP](saved_joint_fee5.lp). Complete five-bus witnesses: [fee 0](saved_joint_fee0.json), [fee 5](saved_joint_fee5.json).

**Scope matters.** These are new bounded charging optimizations of saved trip sequences, not new full column-generation runs. The fee-0 and fee-5 arms inherit different trip assignments and station paths, so the change from 42 to 30 starts is descriptive, **not a controlled fee-only causal effect**. The reconstructed station path is fixed, with one optional charging visit per gap, confined to a single tariff hour. Exact piecewise SOC-time constraints, physical SOC replay, chronology, trip ownership, terminal energy and five-bus charger counts pass. Counts peak at one at both opportunity sites, including in the original five-bus baseline. Background buses outside this cohort are absent. The new trajectories still use the static reference deadhead table; time-dependent deadheads, departure-platform blocking, FIFO at 4808 and crew rules are not certified. Thus battery/rate/reserve/group/count parameters are corrected, but full GIRO operational equivalence is still unfinished.

A first, individually feasible maximum-charging replay overlapped two chargers at some sites; [that failed-capacity diagnostic is preserved](saved_sequence_replay.json). The final MIP jointly rescheduled those visits and removed the conflicts. This illustrates why individual route replay is insufficient.

The more restrictive [fixed-original-window figure](one_bus_k5_matched.png) is retained as a diagnostic. With original movements and original charging windows, both fees retain all 52 starts (9 on duty 13414). Original opportunity windows already operate at their taper limit; fixed windows plus equal terminal energy provide essentially no freedom. A 0.001 kWh per-bus numerical tolerance is used in that secondary model; its sub-0.002 cost difference is not a reported saving. Earlier strict-equality numerical failures remain in its append-only logs. The exact executed source is archived under `executed/` with a matching manifest hash.

## Cost-aware duplicate removal: actual k5 result

The raw historical peak08 five-bus cover served source trip **165** twice. The bounded cleanup considered seven trip subsequences and 1,291 columns, assigned the service to output bus 2, and removed its occurrence from original bus 4 (indices in JSON are zero-based). It recomputed travel and charging instead of assuming the bus could wait at the deleted trip's destination.

| Measure | Before | After |
|---|---:|---:|
| Buses / passenger trips | 5 / 62 | 5 / 62 |
| Redundant traversals retained as empty driving | 1 | 0 |
| Electricity cost, continuous repricing | 127.269205 | 124.692049 |
| Charged energy, kWh | 1972.369986 | 1891.499985 |
| Trip traversal energy, kWh | 2653.629985 | 2611.129985 |
| Deadhead energy, kWh | 236.8 | 199.2 |
| Fleet terminal energy, kWh | 281.940001 | 281.170001 |

Consumption falls by **80.10 kWh** (42.50 redundant trip + 37.60 deadhead), while purchased energy falls by 80.87 kWh because return energy also falls by 0.77 kWh. The aggregate source floor remains satisfied; do not call all 80.87 kWh a consumption saving. There are still 44 charging starts. Electricity cost falls by 2.577156 tariff cost units. Fleet optimality within this repair pool is proved; the charging MIP ends after 120 seconds at a 0.0253% gap.

This cleanup retains its source physics: 240/350, zero reserve, unconstrained shared capacity. It is **not** a strict-GIRO feasibility claim. [Summary](cleanup_result/summary.json), [whole-schedule validation](cleanup_result/validation.json), [per-occurrence ledger](cleanup_result/assignment_ledger.json), [fleet log](cleanup_result/fleet.log), [charging log](cleanup_result/charging.log). Source routes and hashes are preserved. The engine is existing tested commit `663ece9c8d5d02e03087a300ecca0605a4ad9292`; no separate competing cleanup algorithm was introduced.

## Pipeline integration and reproducibility

New commit `24fbda81` on `codex/week-cleanup-20260921` adds opt-in final cleanup to `run_terminal_energy_cg.py`: `--postprocess-exact-once --cleanup-seconds 600 --cleanup-threads 4`. It writes a separate `dispatch_cleanup/` and `dispatch_cleanup_receipt.json`, preserving raw CG/MIP proofs. Downstream passenger service must use `dispatch.service_trips`; a validated fallback can retain empty driving. Unsupported physics is rejected before launch, so the constant-power backend cannot silently weaken a nonlinear GIRO model. The metadata now also records the actual configured start fee rather than hardcoding zero.

Tests: **35 passed plus 3 subtests**, including no-incumbent handling, unsupported-physics rejection, immutable source handling and duplicate-cleanup validation. [Test output](cleanup_tests.log). A real saved-source integration smoke test is recorded in `integration_smoke/`; it uses a two-second MIP budget only to exercise the complete new stage.

All runs here are local Gurobi 12.0.1, four threads, with no cluster job or dependency changes. [Artifact and input hashes](artifact_manifest.json). Primary rebuild scripts: `matched_fixed_windows.py`, `replay_saved_sequences.py`, `joint_sequence_charging.py`, `validate_and_plot_joint.py`. The two repaired charging arms depend on the individually replayed fixed sequences. Figures use standard scientific plotting; no generated artwork is substituted for numerical results.
