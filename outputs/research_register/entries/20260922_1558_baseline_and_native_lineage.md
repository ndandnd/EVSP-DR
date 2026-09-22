# Baseline extension and native lineage check — 22 September, 15:58 UTC

Entry point: [current report](../../research_management_20260922/monitor_20260922T155736Z/README.md).

## New baseline endpoints

The single scoped scheduler collection found 20 running jobs, 33/44 prepared graphs and 75 genuine solver dependencies. No new failure, broken dependency or additional preemption required recovery. Existing held historical jobs and EVSP–V2G work were untouched.

New k33 MIPs: C1 36/bound33, C2 36/bound32, C4 38/bound32. Earlier C3 remains 34/bound33. Both stages reach time limits in all four cases; target attainment and finite-pool fleet proof are false. Native selected-route replay passes baseline physics, while duplicate removal and shared capacity remain unvalidated.

New CG endpoints C1/C3/C4 k34 and C2/C6 k33 reach the four-hour wall limit without pricing certificates. Their fractional route weights are restricted-master values. The audit retains C2's 0.037537633 objective-reconstruction discrepancy and the original solver tolerance. It does not relabel the CG source clean or reconcile differing application/Slurm memory metrics by assumption.

[Manifest, 130 copied-source hashes, exact endpoints, full Gurobi logs, settings and resource accounting](../../research_management_20260922/monitor_20260922T155736Z/operations/README.md). Baseline settings, input identities, initialization and execution pins are unchanged; no new baseline job was submitted.

## New native validation job

**768638 completed 0:0**, 35 scheduler seconds. The authentic k17→k19 gate matches all 8,343 inherited records exactly after normalizing only checkpoint ID, and physically replays all 8,397 saved initial routes. The 54 new singletons also reproduce their saved grid/continuous schedules, costs and metadata. No graph, CG or MIP ran. Fresh singleton optimization equality and full graph parity remain separate checks.

Audit commit `34d39caff534c089fbffed397f5a42b223b590ad`; model commit `fedf421461f94727e6b1292a0e7789ab76ed8587`; explicitly compatible parent `35770aae2c08e7d5a356cc3b673e67608e5b1036`. The source branch is pushed and verified. Manifest SHA256 `9eb6585cf4e452cd8bad301eab6b92916ad5a3c9140f6d602fb33aa1459fa560` authenticates all input/source/physics identities. Its independent review includes full replay, three unit tests, three negative tests and 23 package/resource checks.

Resources: 1 CPU, 8 GiB, 30 minutes, default partition, no job dependency, requeue with unique attempt paths, reserved GPU node excluded. Gate time 29.074884 seconds; child peak 499,560 KiB. Remote root `/home/nc437/ladder-lite/strict_lineage_gate_20260922_34d39caf`. [Manifest, native logs, hashes, submission and validation receipts](../../research_management_20260922/strict_graph_reuse/production_gate/README.md).

The original strict physics remain 239.01 kWh, 15% reserve, PARX 60 kW/other chargers 240 kW, 2.5 kWh SOC and five-minute event discretization, 1,560-minute waiting bound, reserve-only terminal constraint and covering master without shared charger capacity. No new solver outcome or full-GIRO feasibility claim follows from the gate.

## Publication and next work

The current Doc's existing k33 table and strict-validation sentences, and weekly slides 42 and 10, were updated in place. Before/after exports and independent preservation/source checks are recorded in the [publication directory](../../research_management_20260922/monitor_20260922T155736Z/publication/). Existing figures, historical decks and unedited Doc tabs remain untouched.

Full-size graph validation **772820** was RUNNING on joachims-cpu-02 at 16:35:58 UTC, with 2 CPUs / 16 GiB / eight hours on the default partition, requeue and the reserved GPU node excluded. Wrapper `3bb32c1a84af73c97689dfc9f5ad43136cfcead9` is pushed; the model remains clean `fedf4214`. Manifest SHA256 `37c8419ad5da09d968efa5ea2471c5a282412f4f44b139d23a6517823d5e6e3a` and independent 27-package/five-unit/three-mock checks bind the graph-only test. One cold build and a separate reload process compare graph identity, two diagnostic dual vectors and the complete native initializer. Graph preparation is charged separately; old k19 and ancestor costs remain in accounting.

Remote root `/home/nc437/ladder-lite/strict_k19_graph_validation_20260922_3bb32c1a`; no scheduler dependency, but the worker authenticates the completed 768638 lineage proof before building. Only a sealed completed cold phase may resume at reload after preemption. Partial-build artifacts fail closed for review; they are never silently overwritten or rebuilt. [Deployment, resource verification and restart policy](../../research_management_20260922/strict_graph_reuse/production_gate/full_graph_proposal/README.md).

No new strict CG/MIP or k20 has been submitted. The four-hour meaningful-change heartbeat remains unchanged. The completed 25 MIP trials and capacity-compression pilot were not duplicated.
