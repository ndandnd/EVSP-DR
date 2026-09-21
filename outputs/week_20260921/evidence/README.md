# Solver evidence — verified 21 September 2026

This directory contains **35 complete two-stage Gurobi logs**, result JSONs, receipts, and the saved duals used below. The 193 copied files match their Unicorn SHA-256 hashes. Start with [the compact solver table](solver_summary.csv) or [verbatim excerpts with original line numbers](LOG_EXCERPTS.md); use [source_manifest.json](source_manifest.json) for every exact local/remote location and hash. Run `python3 outputs/week_20260921/evidence/build_evidence.py` from the repository to reproduce the tables and assertions. No optimization was run for this audit.

## Read one nine-bus proof together

Open [C1 fresh control, full Gurobi log](k8_witness/c1_k08/control/391804_r0/gurobi.log). Its complete local location is:

`/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k8_witness/c1_k08/control/391804_r0/gurobi.log`

The unchanged source is:

`/home/nc437/ladder-lite/review_witness_columns_20260917/results/c1_k08/control/391804_r0/gurobi.log`

1. Lines **19–26**: 194 rows, 39,940 binary columns, and every objective coefficient is one. This first solve counts buses; it is not the charging objective.
2. Line **53**: `Root relaxation: objective 8.000000e+00`. Fractional weight eight is possible.
3. Lines **150–156** end that first solve:

```text
150: Explored 28694 nodes (10867636 simplex iterations) in 822.12 seconds (2773.96 work units)
155: Optimal solution found (tolerance 1.00e-04)
156: Best objective 9.000000000000e+00, best bound 9.000000000000e+00, gap 0.0000%
```

The incumbent proves that nine buses suffice; the matching best bound rules out eight **in this supplied finite pool**, within Gurobi's numerical tolerances. It does not say that the original scheduling instance requires nine. The augmented pool supplies a counterexample to that stronger claim.

4. At line **169** the second solve begins with 195 rows: a fleet cap has been added and the objective now concerns charging. The later `Time limit reached` at line **534** concerns this second solve and does not undo the fleet proof.
5. Open [the paired augmented log](k8_witness/c1_k08/augmented/391805_r0/gurobi.log): line **19** has 39,948 columns, exactly eight more; line **112** reports objective eight, bound eight, gap zero. The fleet solve took **92.38 seconds** (line 106).

## All five k=8 pairs

Numbers below are **fleet-stage** runtime from the full Gurobi log, excluding preparation and the subsequent charging stage.

| Case | Fresh columns | Fresh fleet / bound | Fleet seconds | Added records | Augmented fleet / bound | Fleet seconds |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 39,940 | **9 / 9** | 822.12 | 8 | **8 / 8** | 92.38 |
| C2 | 20,800 | 9 / 8 (time limit) | 1800.03 | 7 | **8 / 8** | 56.03 |
| C3 | 16,931 | **9 / 9** | 1757.78 | 8 | **8 / 8** | 5.17 |
| C4 | 39,668 | **9 / 9** | 506.37 | 8 | **8 / 8** | 41.37 |
| C5 | 26,942 | **9 / 9** | 151.16 | 8 | **8 / 8** | 52.86 |

The original and augmented journals were checked on Unicorn again: **each augmented journal starts with the byte-identical original journal**, followed by exactly the stated number of records. Every original/augmented source hash matches the historical prep analysis and the MIP input hashes; see [pool_identity_remote_audit.json](pool_identity_remote_audit.json). Instance/deadhead/reference/tariff hashes and recorded physics match between fresh and sequential witnesses. Both arms use covering, 240 kWh, 240 kW, zero reserve, flat tariff, conservative expanded-grid costs, no shared charging capacity. Runner commit `871d057e1067411f09581e37d78f7c1ca43f68bb`; seed 0; eight threads; total two-stage solver limit 3600 s; fleet-stage limit 1800 s. CG pricing certificates refer to the conservative expanded-grid weighted objective, not a continuous-cost or fleet-only full-model certificate.

Other fresh proof lines: **C3 line369** in `k8_witness/c3_k08/control/391812_r0/gurobi.log`; **C4 line141** in `k8_witness/c4_k08/control/391806_r0/gurobi.log`; **C5 line110** in `k8_witness/c5_k08/control/391808_r0/gurobi.log`. All paths are below the absolute local directory printed above. Exact links and full remote paths are in [the excerpts](LOG_EXCERPTS.md).

## Why the fresh integer solutions are worse

The missing property is **a combination of whole routes that covers every trip**. The LP can split buses across many overlapping routes: the five saved LP solutions have total route weight eight distributed over **99, 101, 80, 94, and 97 positive columns**, respectively. Every positive column in those solutions is fractional. A fractional blend can balance trip coverage even when no selection of eight complete routes can do so. The four nine-bus proofs demonstrate that obstruction for the complete saved fresh pools, not just their LP supports.

Ordinary LP pricing scores one route by its cost minus the sum of its trip dual credits. This score rewards improvement to the fractional objective. It does not directly value a route's ability to complete a particular integer fleet. **37 of 40 witness routes have positive reduced cost above 1e-4; 39 of their 40 trip sets are absent from the fresh pools.** The fresh pools already contain 381–670 multi-trip columns within 1e-4 of zero reduced cost. Adding the witness routes preserves the old LP optimum while changing the integer optimum. Thus the LP objective gives ordinary pricing no incentive to add these particular routes at the saved final duals.

There is an exact, checked numerical explanation for C1. Let `c` include the route penalty plus charging cost, `pi` be the final covering duals, and `x` the eight-route witness. Then

`c*x − LP = sum(selected reduced costs) + sum(pi_i * (coverage_i − 1))`.

For C1 this is **96.2720 = 72.1421 + 24.1299**. The LP costs 800,383.688 and the eight-bus witness costs 800,479.960. The witness is a slightly dearer fractional choice, even though it avoids the ninth bus in an integer solution. This identity is recomputed directly from all saved trip duals and witness routes, with error below 1e-6 for every case; see [mechanism_summary.csv](mechanism_summary.csv) and [witness_route_audit.csv](witness_route_audit.csv). The weighted objective and route weight are deliberately reported separately.

Why does sequential initialization help? **34/40 witness routes are explicitly tagged as inherited from an earlier pool and replayed in the child graph; the other six were generated along the warm trajectory.** Sequential runs therefore retain useful combinations that a fresh run's pricing trajectory misses, and initialization also changes later duals. This is stronger evidence than merely counting more columns. It does **not** identify a unique combinatorial obstruction or establish that all other enrichment methods fail. Positive reduced costs at one final dual do not prove a route could never have been generated earlier. Testing route-pair conflicts, fixed-route residual pricing, and controlled inheritance ablations would distinguish the remaining structural explanations.

## The 12-hour k=15 test is now fully finished

All 12 fleet searches hit their 43,200 s limit; **all 12 charging stages have also finished at their remaining budget**, with final solver status `TIME_LIMIT`. Final fleets are unchanged. All scheduler records are `COMPLETED/0:0`; scheduler completion is distinct from an optimization proof.

| Case | Plain buses | Heuristic buses | Fleet bound (both) | Plain fleet nodes | Heuristic fleet nodes |
|---|---:|---:|---:|---:|---:|
| C1 | 18 | 18 | 15 | 144,481 | 613,278 |
| C2 | 17 | 17 | 15 | 352,135 | 679,987 |
| C3 | 17 | 16 | 15 | 2,087,152 | 445,603 |
| C4 | 19 | 17 | 15 | 124,810 | 706,891 |
| C5 | 16 | 16 | 15 | 1,462,169 | 1,067,811 |
| C6 | 18 | 17 | 15 | 179,639 | 641,682 |

Plain and heuristic arms use the **same six hashed fresh pools** searched previously. Heuristic settings are `MIPFocus=1`, `NoRelHeurTime=1800`, `Heuristics=0.5`. Runner `6830caa225856903d1157ef8587863c7ae21ad53`; seed 0; eight threads; total solver limit 45,000 s; allocation 13h30, 24G, Scaglione CPUs excluding compute-01. Scientific settings match the k=8 baseline above. See [k15_manifest.json](k15_manifest.json) for all input hashes and settings, and [k15_12h_summary.csv](k15_12h_summary.csv) for completed charging costs, gaps, stage times, full log locations, and exact proof-line references.

Example full log: local `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k15_12h/c3_k15__plain/gurobi.log`; Unicorn `/home/nc437/ladder-lite/review_fresh_k15_longmip_20260917/results/c3_k15__plain/gurobi.log`. Lines **2892–2898** record 2,087,152 nodes, 43,200.07 s, time limit, incumbent 17 and bound 15. The final charging summary is at line **3125**.

**Conclusion:** this is substantial negative evidence for spending more search time on unchanged fresh pools, and supports prioritizing new columns. It **does not prove** that any k=15 pool lacks a 15-bus cover. The best result is 16, only one bus above the target. Earlier prose claiming that no search budget could help at k=15, that the k=15 pool integrality gap was established, or that no arm came within two buses is superseded by this precise statement. The k=8 missing-column proofs cannot be promoted to k=15 proofs.

## Integer-directed pilot and the completed C1 follow-up

The original pilot remains: controls A/B both end at nine buses in C1/C3/C4/C5; treatment final MIPs attain eight in **C3/C4/C5**, while C1's dive finds eight but its final MIP returns nine because the dive incumbent was not transferred. Recorded treatment elapsed times are 3692, 1207, 2372, and 773 s; C1 exceeds a strict one-hour end-to-end claim. Controls B receive the recorded graph-build allowance; this is not evidence of a rebuilt graph. No sequential or GIRO witness solution was supplied to treatment pricing.

Follow-up **628441** now succeeds: the eight routes already found by C1's dive were supplied as a validated start, with no new CG and no external witness. In [its full Gurobi log](c1_followup/628441_r0/gurobi.log), line **30** accepts the start at objective eight; lines **71–72** prove fleet eight after 11.43 solver seconds; lines **313–314** prove the charging objective **447.44** in this fixed augmented pool. Total recorded solver runtime is 1593.92 s; scheduler elapsed is 29m33s. This is **additional work on the existing pilot pool**, not a replacement for the original benchmark. Its eight start records are copied as [c1_dive_start.json](c1_followup/c1_dive_start.json). The generic schema field `added_giro_route_count=8` names a reused route-input mechanism; these records came from the dive, not GIRO. The start reports zero added/replaced pool columns and preserves eight existing equal-incidence entries.

All these final routes passed route-level physical replay. Duplicate removal is not validated for C1, including the follow-up, and cross-bus station capacity is not imposed. A cover and route replay are not a complete dispatch validation. The pilot's artificial-variable infeasibility labels and a full global branch-and-price certificate remain outside this evidence.

## Source and historical record

Original research notes remain at `outputs/independent_review_20260916/advisor_witness_columns_20260917/README.md`, `outputs/independent_review_20260916/advisor_fresh_k15_longmip_20260917/README.md`, and `outputs/research_execution_20260921/README.md`. This verified entry supersedes only the specific overstatements identified above and the stale “charging stages running / C1 pending” status. Preserve the historical runs, failed preflight attempts, and original benchmark result.
