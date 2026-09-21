# EVSP–DR research journal — week of 21 September 2026

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

## Historical milestones retained

**16–17 September:** time-only group separation explains mixed-group fleet advantages, without establishing strict-energy feasibility. **18 September:** k8 witness proofs and open k15 bounds. **19–20 September:** replay/memory recovery and integer-directed pilot. **21 September:** completed charging, verified logs and C1 recovery supersede stale statuses. [Original evidence and failed attempts remain preserved](../research_register/README.md).
