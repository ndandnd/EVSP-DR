# EVSP–DR research journal — week of 21 September 2026

## 21 September, afternoon — from pilot evidence to controlled tests

The paper preview now reproduces five figures with268checks across70source files. Across24cases at targets5/8/10/15, both fresh and sequential methods certify their weighted LP endpoints; one-hour pool MIPs match6/24 versus24/24. Accumulated sequential work is included. This comparison establishes the LP/integer split, not a causal hardware-normalized runtime speedup.

Sixteen balanced k8 jobs test integer-directed pricing after fixing own-incumbent transfer and budget accounting;55tests pass. At16:24EDT the first four treatments (C3/C5,bothseeds) prove8 in14.1–18.8minutes; controls remain incomplete. Six paired k15 follow-ups are conditional on native validation. Eighteen charging cells hold assignment, paths, physics and terminal floors fixed while changing fee0/5 across three tariffs. These are fixed-path charging tests, not new CG. All18witnesses validate:17raw and one separately corrected numerical witness. The nine paired incumbents use8–22fewer starts atfee5; six searches retain gaps up to3.83%.

Strict packed recovery finishes2,453iterations in4h including115.4min graph construction, at3.934GiB peak. Its parent-prefix16 subgroup has9duties/256trips. Fractional route weight9 is uncertified; the final saved pool proves10integer buses. All pool routes individually replay, but the chosen covering solution duplicates57assignments and violates omitted capacity limits. This is computational progress with remaining mathematical and dispatch gaps. New successor/benchmark jobs test scaling; all44baseline full40 graph builds were running at16:08EDT. Six requeues lost84.6min, motivating isolated graph-checkpoint work.

[Figures and source tables](../research_management_20260921/paper_results/RESULTS_PREVIEW.md) · [Strict logs/validation](../research_management_20260921/operations/README.md) · [Controlled integer design](../research_management_20260921/integer_columns/README.md) · [Charging design/results](../research_management_20260921/charging_fee_factorial/README.md).

## 21 September — a bus day across five places

[Recorded GIRO duty 13309](complex_route_graphs/README.md) visits five areas, serves 22 passenger trips and charges four times at Heden, Jons väg and PARX. Its midday depot return at 10:42 leads to 105 minutes of charging (+105 kWh), then departure at 12:43. Morning/afternoon graphs share an approximate layout; PARX is displaced for clarity. The five-page companion preserves 27 inter-area movements, 28 visits and all 60 events. Recharge cells match the correct 239.01 kWh 18E2 basis. This is original-only, with no optimized counterpart asserted. [The updated figure tab preserves its previous text and eight images](complex_route_graphs/doc_verification/README.md); no Slides or solver settings changed.

## 21 September — eight jobs finished; the scientific outcomes differ

All eight jobs and six dependent capacity MIPs finished. Those MIPs prove **finite-pool** fleet optima; only one capacity CG supplies a pricing certificate.

| Job | Experiment | Verified result |
|---|---|---|
| 628428 | k1 capacity, shortcut off | 4 h pricing deadline; pool fleet 1, bound 1 |
| 628429 | k1 capacity, shortcut on | Pricing certified in 26.87 min; pool fleet 1, bound 1 |
| 628436 | k2 capacity, shortcut off | 4 h deadline; pool fleet 3, bound 3 |
| 628437 | k2 capacity, shortcut on | 4 h deadline; pool fleet 2, bound 2 |
| 628438 | k3 capacity, shortcut off | 4 h deadline; pool fleet 21, bound 21 |
| 628439 | k3 capacity, shortcut on | 4 h deadline; pool fleet 5, bound 5 |
| 628314 | Strict 18E2 k15 recovery | Graph 15,442 s; zero pricing iterations; subsequent pool MIP 37 |
| 628441 | C1 dive-incumbent transfer | Fleet 8 / bound 8; fixed-pool charging optimum 447.44 |

Capacity tests use 240 kWh / 240 kW, zero reserve and flat tariff. Capacity sweeps pass; k2-off/k3-on retain 1 / 6 extra assignments. Strict k15’s 37 reflects its impoverished 596-column pool and 18E2 subgroup, not full-model or whole-prefix requirements. [Source tables, settings and logs](capacity_strict/README.md).

## 21 September — why fresh integer solutions are worse

**Four fresh k8 pools require nine buses, although an eight-bus solution exists outside them.** C1/C3/C4/C5 each switch from proven 9 to proven 8 after adding eight sequential witness routes. C2 adds seven and reaches 8; its original 9 / bound 8 remains unresolved.

The LP spreads total route weight 8 across 80–101 positive columns, all fractional. Such a blend can balance coverage even when eight complete routes cannot. Of 40 witness routes, 39 trip sets are absent from fresh pools, 37 have positive reduced cost above 1e-4, and 34 originated as inherited routes. Ordinary pricing values improvement to the fractional objective; it does not directly reward a route that completes a particular integer fleet.

C1 makes this quantitative: the eight-bus witness costs 800479.960 versus LP 800383.688. Its 96.272 premium equals 72.142 in reduced costs plus 24.130 in dual-valued overcoverage. It is slightly dearer for the LP, yet avoids an entire extra integer bus. This identity was recomputed from the saved duals. This establishes the mechanism, not a unique obstruction or universal failure of other enrichment methods. [Detailed explanation and reproducible checks](evidence/README.md).

**Read the proof together:** [C1 full log](evidence/k8_witness/c1_k08/control/391804_r0/gurobi.log), lines 19–26 show unit bus costs; line 53 gives fractional 8; lines 155–156 report `Optimal solution found` and `Best objective 9, best bound 9, gap 0`. The later time limit concerns charging. [Eight added columns](evidence/k8_witness/c1_k08/augmented/391805_r0/gurobi.log), line 112, prove 8. Local logs: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/`. Unicorn C1 log: `/home/nc437/ladder-lite/review_witness_columns_20260917/results/c1_k08/control/391804_r0/gurobi.log`. [The manifest](evidence/source_manifest.json) maps all paths and hashes. These are numerical proofs within specified pools.

## 21 September — longer MIPs and integer-directed pricing

All twelve 12-hour fleet searches **and their charging stages are finished**. Plain fleets are 18/17/17/19/16/18; heuristic-focused fleets 18/17/16/17/16/17. Every fleet bound remains 15. The largest search explored 2,087,152 nodes. This favors investing in better columns; it does not prove any k15 pool lacks 15. [Final results and full log locations](evidence/k15_12h_summary.csv).

The original pilot attains 8 in 3/4 treatment MIPs versus 9 in controls. C1’s dive found 8 but its final MIP did not receive the incumbent. Follow-up 628441 transfers it successfully. This additional work preserves the original benchmark, including C1’s 3692 s end-to-end runtime. [Pilot and follow-up logs](evidence/README.md).

## 21 September — harder constraints and dispatch cleanup

The capacity shortcut advances k2 from 6 to 134 iterations and k3 from 2 to 51 within four hours. Strict k15 creates 260.7 million Python arcs; k16 times out building its graph at 211.9 GiB peak memory. The new packed no-capacity implementation passes 44 tests. **Benchmark 646674 now passes:** on the same 26-trip case/node/lattice, all five reduced costs match and all 15 routes replay. Packed construction is 2.90× faster, peak memory 14.1× lower, and mean pricing 530.8× faster. These are small-case measurements, not a scaling proof or CG certificate. K16 recovery 646675 now runs; MIP 646676 depends on it. [Verified benchmark and scope](capacity_strict/README.md).

Charging-aware duplicate removal on raw k5 removes one repeated service, retains five buses and 62 exactly-once trips, and lowers realized electricity cost 127.269205→124.692049. Consumed energy falls 80.10 kWh; charged energy falls 80.87 kWh, including 0.77 kWh less return energy while meeting its floor. Charging optimality remains open after 120 s. This uses historical 240 kWh / 350 kW, zero-reserve, no-capacity physics. Waiting requires checking location, subsequent travel and energy. [Accepted repair and checks](cleanup_physics/cleanup_result/summary.json).

## 21 September — replace the mismatched k5 picture

The old picture did not match GIRO fully. The [replacement picture](cleanup_physics/one_bus_k5_joint_matched.png) recharges saved trip sequences using actual 18E1 settings: 236.44 kWh, 15% reserve, opportunity taper 120–371.5 kW, PARX 60 kW, minimum 3-minute charges, zero setup and 0.1 kW idle. Both arms validate five buses, 62 exactly-once trips, matched per-bus terminal floors, and one-charger limits at 2190L/4808. The other 35 buses are not modeled.

Fee 0 gives electricity cost 158.707 and 42 starts, optimal within 0.01%; fee 5 gives electricity 183.715 plus 150 start fees, 30 starts, and 0.6504% gap after 120 s. **The arms inherit different trip sequences and station paths: 42→30 is descriptive, not a controlled fee-only effect.** Original GIRO has 52 starts and reconstructed electricity 230.645; these are synthetic tariff cost units, not verified currency. The pictured original/fee 0/fee 5 buses have 9/9/6 starts. This is **post-hoc charging optimization of saved sequences**, not fresh CG. Charging windows are restricted to one hour per gap; time-dependent deadhead, platform blocking, FIFO and crew constraints remain unvalidated. [Validation, logs and figure data](cleanup_physics/README.md).

## 21 September, follow-up — geographical context

The [new map and travel table](geography_map/README.md) place the pictured buses at Eketrägatan, Merkuriusgatan and the PARX depot address proxy; a second map adds other chargers as context. Original passenger service takes 53–63 minutes, distinct from the 22-minute empty-driving reference between the same endpoints. Fee0 finishes at Eketrägatan and has a 19-minute depot return; the original finishes at Merkuriusgatan and returns in 7 minutes. Matching charging physics still leaves travel approximations: original 2190→2190L is 1 minute/0.4 kWh versus model 0/0, and the morning depot departure is 8 minutes versus model 7. OSM stop positions and the operator's published address are explicitly sourced proxies; the exact passenger platform and ET_R layover are unresolved.

## 21 September, follow-up — continue all six baseline chains to k40

The [launch audit](chain_extension_40/launch_verification.json) verifies **140 production tasks**: 44 distinct input graphs, 48 CGs and 48 MIPs for six chains at k33–40. Graph array **661616** had five tasks running in the 18:41 UTC snapshot; the other 39 waited for resources, and CG/MIP tasks preserved their true dependencies. All six authenticated k32 parents have later 32-bus incumbents, although only C1/C2/C3/C6 prove 32 within their pools; C4/C5 retain bound 31. Each chain inherits its entire previous-k column journal. Baseline physics and four-hour CG/one-hour MIP budgets remain unchanged. Graph allocation grows to 37 hours/96 GiB based on measured k32 build time and memory. Full40 has three frozen variant classes with 948/947/946 trips, so the six chains are not identical input sets. Favorable scheduling suggests roughly two days to the full frontier, with earlier k33 results. [Settings, variants, resources and receipts](chain_extension_40/README.md).

By the [14:46:43 EDT snapshot](chain_extension_40/final_queue_snapshot.json), 14 new graph tasks run; 29 wait for resources and one scheduled requeue waits for its eligible time. Including the separate strict k16 CG, 15 jobs run in this scope. No new k33+ scientific endpoint is claimed.

## 21 September, later follow-up — spatial schedules and ladder order

The [five-pair gallery](spatial_schedule_graphs/README.md) adds fixed geographic multigraphs, complete visit itineraries and editable time/charge keys for all 15 saved schedules. Original duty 13414 and its fee5 counterpart share twelve passenger trips; every fee0/fee5 paired trip set differs. Trip numbers are prepared-input labels, not GIRO journey numbers. All 751 events and 124 charge windows were independently checked; no optimization was rerun.

The [source audit of ladder order](spatial_schedule_graphs/ladder_path_dependence.md) confirms inherited sequences are reoptimized on the child graph, without importing parent-only charging times. Within each identical final-input/model group, the complete weighted event-graph LP optimum is path independent; uncertified restricted masters and finite integer pools can still differ. The six k40 inputs form three exact groups. This is an implementation/theory conclusion, not a new k40 endpoint.

## Historical milestones retained

**16–17 September:** time-only group separation explains mixed-group fleet advantages, without establishing strict-energy feasibility. **18 September:** k8 witness proofs and open k15 bounds. **19–20 September:** replay/memory recovery and integer-directed pilot. **21 September:** completed charging, verified logs and C1 recovery supersede stale statuses. [Original evidence and failed attempts remain preserved](../research_register/README.md).
