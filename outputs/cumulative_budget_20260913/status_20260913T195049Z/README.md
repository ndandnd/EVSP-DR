# Research update — 13 September, 15:51 EDT

**New integer reach: chain3 matches17 buses.** The extension has four completed MIPs: C1k16, C2k16, C3k16 and C3k17. All match their targets, prove the fleet within their saved pools, and pass individual-route physical replay. Their second-stage charging searches reached the one-hour total MIP budget, so charging optimality is unproved. Seven extension CG endpoints have pricing certificates, including C3k18; a CG certificate alone is not an integer solution.

| Chain | Largest target with a verified integer match | Buses found | CG minutes at that k |
|---|---:|---:|---:|
| 1 | 16 | 16 | 47.7 |
| 2 | 16 | 16 | 44.3 |
| 3 | 17 | 17 | 32.0 |
| 4 | 15 | 15 | 64.4 |
| 5 | 15 | 15 | 52.3 |
| 6 | 15 | 15 | 64.0 |

The k15 rows retain the previous completed full-inheritance baseline; their sources remain in the register. CG minutes are this k’s import+CG time, excluding earlier k values, original graph construction and MIP. All these runs use baseline covering/240kWh/240kW, fee5, no shared station capacity and no terminal-SOC floor. Selected-route replay, duplicate-trip removal and shared-capacity validation remain separate.

## Cumulative-budget fresh controls

19 of24 fresh CG runs have pricing certificates; five continue. All24 matched warm-reference MIPs finished and match their target fleets with finite-pool proofs. Twelve fresh MIPs finished: six target matches, two proved larger pool optima, and four still-open fleet gaps. Seven other fresh MIPs are pending/running after certified CG; the remaining five await CG. No certified fresh endpoint should be restarted just to spend unused allowance.

| Case | Fresh CG minutes | Fresh buses | Fresh pool fleet bound | Fleet proved in fresh pool? | Warm buses | Meaning |
|---|---:|---:|---:|---|---:|---|
| C1, k=5 | 20.6 | 5 | 5 | yes | 5 | target matched |
| C1, k=8 | 51.1 | pending | pending | pending | 8 | MIP pending |
| C1, k=10 | 80.0 | pending | pending | pending | 10 | MIP pending |
| C1, k=15 | running | pending | pending | pending | 15 | CG running |
| C2, k=5 | 11.6 | 5 | 5 | yes | 5 | target matched |
| C2, k=8 | 18.0 | 9 | 8 | no | 8 | integer gap open |
| C2, k=10 | 45.9 | pending | pending | pending | 10 | MIP pending |
| C2, k=15 | running | pending | pending | pending | 15 | CG running |
| C3, k=5 | 5.1 | 5 | 5 | yes | 5 | target matched |
| C3, k=8 | 13.3 | 9 | 8 | no | 8 | integer gap open |
| C3, k=10 | 30.0 | 11 | 10 | no | 10 | integer gap open |
| C3, k=15 | 76.7 | pending | pending | pending | 15 | MIP pending |
| C4, k=5 | 11.3 | 5 | 5 | yes | 5 | target matched |
| C4, k=8 | 47.3 | pending | pending | pending | 8 | MIP pending |
| C4, k=10 | 59.4 | pending | pending | pending | 10 | MIP pending |
| C4, k=15 | running | pending | pending | pending | 15 | CG running |
| C5, k=5 | 8.0 | 6 | 6 | yes | 5 | pool needs more buses |
| C5, k=8 | 25.9 | 9 | 9 | yes | 8 | pool needs more buses |
| C5, k=10 | 33.5 | pending | pending | pending | 10 | MIP pending |
| C5, k=15 | running | pending | pending | pending | 15 | CG running |
| C6, k=5 | 6.1 | 5 | 5 | yes | 5 | target matched |
| C6, k=8 | 12.8 | 8 | 8 | yes | 8 | target matched |
| C6, k=10 | 27.2 | 11 | 10 | no | 10 | integer gap open |
| C6, k=15 | running | pending | pending | pending | 15 | CG running |

**Two different limitations:** C5k5 proves6 in its fresh pool versus warm5; C5k8 proves9 versus warm8. More MIP time on either unchanged pool cannot meet the target. At C5k8, the overall MIP status is TIME_LIMIT because charging optimization remained open; the fleet proof is complete. In contrast C2k8 and C3k8 have9buses/bound8, and C3k10 and C6k10 have11/bound10. These four ran the full one-hour MIP allowance without closing the fleet gap: the evidence does not prove that target-sized integer solutions are absent from those pools. All12 fresh pools came from pricing-certified CG.

Thus distinguish missing useful integer columns (proved in two cases) from incomplete integer search (four unresolved cases). The matched warm pools attain every target. Historical CG revisions/hardware varied, so the timing comparison remains retrospective. Pricing certificates apply to the tested graph and tolerance, not every continuous formulation.

## Monitoring recovery

The first collection timed out after300s while its process waited in filesystem RPC; SSH responded normally. Its own collector process was stopped. A diagnostic retry, with unchanged collection logic plus a timed stack dump, completed successfully; it showed time spent decoding phase telemetry. No broader storage outage or solver error is inferred. Live scheduler at15:49 showed57running allocations,124pending queue entries and no unsatisfiable dependencies; the held historical array remained held. The successful snapshot has no new execution errors or confirmed preemptions. Source results use the completed retry, not the failed collection.

[Machine-readable comparison](comparison.csv), [source collections](collection.json). Full snapshot: `outputs/post_meeting_20260910/monitor/20260913T195049Z.json`.
