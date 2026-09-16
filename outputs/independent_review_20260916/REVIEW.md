# EVSP–DR independent review — 16 Sep 2026

> **For Astra / GPT — read this file in full, then execute §4 in order.** Start with P0 items 3 (depot-rate grep), 2 (Lagrangian bound) and 4 (Jaccard test); they change headlines and cost no solver time. Do not resubmit or cancel cluster jobs without the user's say-so. Report each result against the finding it tests (F-number) and mark it verified/refuted. Where this review says [H], treat it as a hypothesis to test, not a conclusion to defend.

Reviewer: Claude (Fable 5.1), read-only. Sources traced to the primary artifacts named in `HANDOFF.md`; no jobs submitted, no research documents modified. Facts marked **[V]** were verified directly against a file; **[H]** are hypotheses the evidence suggests but does not establish.

Paths are relative to `/Users/nadan/Documents/projects/demandresponse` unless absolute (cluster).

---

## 0. One-paragraph verdict

The engineering is careful and the bookkeeping is unusually good. But the headline claims are stronger than the evidence in three ways. (1) The 9-for-9 pattern in F2: the pipeline finds GIRO's count whenever the LP says GIRO is optimal, and fails by 3–10 buses whenever the LP says GIRO can be beaten — consistent with reassembly of inherited GIRO structure rather than search. (2) The "recover GIRO's fleet" framing is not like-for-like: 12 of 40 canonical GIRO duties fail replay in the model, the recorded manifests show a single 240 kW rate with no separate 60 kW depot rate (to be confirmed in code), and there is no 15% SOC floor — so `k` is not the model's reference optimum in either direction. (3) The two research threads are disjoint: the entire scaling ladder (102 chain rows) runs under a **flat tariff**, and the demand-response result rests on **one 5-duty, 62-trip instance** pre-selected for replay eligibility, with a comparator that produces a 2.5 kWh charging event. There is also one very cheap, high-value result sitting unclaimed: a Lagrangian bound would certify fleet lower bounds for most of the 40 wall-limited CG runs at zero compute.

---

## 1. Verified facts

**Chain campaign** (`outputs/overnight_next_20260914/status_20260916T194843Z/all_chain_extension_results.csv`, 102 rows, k=16..32, chains 1–6):
- [V] All 102 rows: `hourly_prices_flat.csv`, cover, 240 kWh / 240 kW, warm-inherited pools, init `greedy_pool_partition` + `singletons` (register). Fee 5 per README; the register records `charge_start_cost = unknown` for all 324 chain rows.
- [V] Certification collapses with k: at k ≥ 25, 40 of 48 CG endpoints stopped on `wall_limit` (239–240 min). Only 8 certified.
- [V] `fractional_route_weight` (LP fleet) equals k **exactly** in 93/102 rows. In the other 9 it is k−1: w5_k27, w5_k28, w5_k29, w4_k30, w5_k30, w4_k31, w5_k31 (**certified**, LP = 30.000), w4_k32, w5_k32.
- [V] Pattern: when LP = k, the 1-hour MIP incumbent is within 0–3 of k (usually k or k+1). When LP = k−1, the incumbent is 3–10 above the LP: 29, 34, 35, 39, 37, 40, 36, 34, 37 respectively.
- [V] `duplicate_removal_validated = False` and `shared_capacity_validated = False` on all 102 rows.
- [V] Last minimum reduced costs at wall limit are small relative to the 100 000/bus objective: |rc| ≤ 0.32 in 48 of 54 uncertified rows; extremes −0.97 (w5_k27), −1.37 (w4_k30), −6.67 (w1_k32).

**Longer MIPs** (`longer_gap_results.csv`, 26 rows, same pools, new trees, 3 h fleet search):
- [V] 17/26 matched the target; all with `pool_fleet_bound = target` so they are pool-proven, none full-model proven. `duplicate_removal_validated = False` on all 26.
- [V] MIP outcomes are highly seed/tree dependent: w1_k28 original 1 h gave 31; a new tree found 28 in **14.1 min**. w1_k31: 34 → 31 in 108 min. w5_k30: 37 → 31 (bound 29), still unproven after 180 min.
- [V] Chain 5, k31: certified LP = 30; 1 h MIP 36; 3 h MIP 31; pool bound 30. This is the only case where the model provably (in relaxation) needs fewer buses than GIRO.

**Fresh vs warm** (`outputs/cumulative_budget_20260913/status_20260916T194843Z/comparison.csv`, 24 rows):
- [V] All 24 fresh CGs certified, using 5–291 min of allowances of 15–1389 min. Fresh LP = warm LP = k in every row.
- [V] Fresh MIP pools 38k–80k columns; warm pools 70k–130k (register `pool_size`).
- [V] At k=15: fresh incumbents 18, 17, 18, 19, 16, 20; bound 15.000; `fresh_fleet_proven = False` in all six — i.e. **1 h MIP timeouts**, not pool proofs. The four genuine pool exclusions are at k=5/8 (c5_k05 bound 6, c1/c4/c5_k08 bound 9).

**Controlled comparisons** (`outputs/controlled_comparison_20260913/status_20260913T063519Z/README.md`):
- [V] 3 inputs × 2 orders, 24 pairs, flat tariff. Full-pool vs 512-route inheritance: identical certified LP to 1e-4; integer 9→8, 11→10, 17→15; CG iterations 950→177, 945→222, 891→280. Descriptive evidence, correctly labelled as such.

**Zero-fee charging** (`outputs/zero_fee_validated_comparison_20260916/comparison.json`; cluster `giro_zero_start_fee_20260913/results/peak08/fee0/joint/comparison.json`):
- [V] One instance: duties 13401/13403/13405/13408/13414, 62 trips, 240 kWh / **350 kW** (not the 240 kW chain baseline), fee 0, aggregate terminal energy matched (281.17 kWh both arms), 5 buses both arms, `overcovered_trip_count = 0` after cleanup.
- [V] Continuous costs: 128.29→124.69 (−2.8%), 164.23→160.61 (−2.2%), 95.29→88.30 (−7.3%). Grid costs agree in direction. peak18 CG-arm charging gap 0.136%.
- [V] Fixed-duty comparator has **46 charging starts** across 5 buses (10, 11, 11, 7, 7), including a 2.5 kWh event (cst 376 → cet 376.43 = 26 s). GIRO's own rule: ≥3 min opportunity charge, 45 s setup for 18E2 (`giro_requirements_audit.md`).
- [V] The cohort was chosen from a replay screen: "The full canonical 40-duty screen … 28 passed" (`outputs/meeting_20260910/TARIFF_EXPERIMENT_PLAN.md`). 12 GIRO duties are rejected by the model's replay.
- [V] `original_giro` (GIRO as-is repriced) exists in the cluster comparison.json but is not surfaced in the headline; the plan doc records flat-tariff as-is 187.60 energy + 260 fees.

**Physics vs GIRO** (`outputs/model_fairness_audit_20260913/giro_requirements_audit.md`):
- [V] GIRO: usable 236.44/239.0 kWh by group; opportunity charging SOC-dependent 371.5→120 kW (Partille), 386→30 kW (Frölunda); **depot 60 kW**; hard 15% floor; route-21 blocks group-separated; 3-min minimum charge. Baseline model: 240 kWh, 240 kW everywhere, no floor, homogeneous.

**Scope**: [V] Largest instance run through CG: k=32, 779 trips. Partille full = 40 duties / 987 regular trips; Frölunda (61 tasks) untouched. Register max `trip_count` = 750 outside `union_target_feasibility_20260915`.

---

## 2. Findings, ranked

### F1. Headline fleets are coverings — fleet counts are safe, cost figures are not (downgraded after author discussion)
Every chain/longer-MIP result has `duplicate_removal_validated = False`. For the **fleet** objective this is harmless: a duplicated trip can be driven empty along the same path at the same time, leaving the route's timing and energy unchanged, so any covering converts to a partition with equal fleet. State this in the paper — it means covering and partitioning share the same fleet optimum. What is *not* safe is any electricity/charging-cost figure taken from a k ≥ 16 covering: it includes the empty duplicate kilometres. Run the conversion + replay anyway (cheap, tooling exists) so cost columns are partition costs.

### F2. The method looks good exactly when GIRO's answer is the best answer [H, 9/9 pattern]
Plain version. The LP gives the smallest fractional fleet. In 93/102 runs it equals k (GIRO's count) and the MIP nearly always finds k. In the 9 runs where the LP says k−1 is possible — the model can beat GIRO — the MIP does *worse*: 3–10 buses above the bound (29, 34, 35, 39, 37, 40, 36, 34, 37). If the MIP were genuinely searching, "there is a better solution" should not make it fail. The pattern fits a MIP that reassembles near-GIRO routes inherited through the warm chain. If so, "CG+MIP recovers the GIRO fleet" is really "CG+MIP reassembles GIRO's fleet from GIRO's parts", and the capability claim is much weaker. Test (no solver): trip-set Jaccard between each selected route and its nearest GIRO duty, split by LP = k vs LP = k−1. Near-1 similarity in the LP = k group and lower in the k−1 group confirms; uniformly low similarity refutes.

### F3. Free lower bounds are being left on the table (high value, zero compute)
40 of 48 CG runs at k ≥ 25 are uncertified, so the doc says nothing is proven. But the pricer is exact, so at the last iteration the Lagrangian bound holds: z_LP ≥ z_RMP + K·rc_min, valid for any K ≥ Σλ at the LP optimum; K = best known integer fleet works because 100 000·Σλ* ≤ z* ≤ z_incumbent. With |rc_min| ≤ 0.32 and K ≤ 40 the correction is ≤ 13 objective units against a 100 000/bus weight, so fleet LB = ⌈(z_RMP + K·rc_min − E_max)/100 000⌉ where E_max is any valid upper bound on electricity+fees in an optimal solution (computable from total trip energy × price + F·240 kWh × price + F_max × 5). Expected outcome: most wall-limited runs get a certified model-LB equal to their `fractional_route_weight`, which would make chains 1/2/3/6 at k=32 **proven optimal in the model** if F1 passes. Caveat: check that the recorded `last_pricing_reduced_cost` is the exact global minimum at the final duals, not the best of a capped batch.

### F4. Q1 is not like-for-like with GIRO — verify the depot rate first
The consequential differences: (a) **depot charging rate** — GIRO's PARX/KEX is 60 kW; every recorded manifest carries a single `charge_kw = 240`, the register's `parx_kw`/`non_parx_kw` are `unknown` in all campaigns, and `grep -riE "parx_kw|depot_kw" src/*.py` finds nothing live. If depot = 240 kW, midday refills are 4× faster than reality and the LP = k−1 cases in chains 4/5 may be an artifact. **Astra: confirm in the pinned commit `a0e0bb76` with one grep; if absent, this is fix #1.** (b) no 15% SOC floor (GIRO hard rule); (c) 12 of 40 GIRO duties fail model replay, so k is a count of partly-infeasible parts; (d) homogeneous fleet where route-21 blocks are group-separated. Battery 240 vs 236/239 kWh is negligible — ignore. The honest reference is the certified LP bound under stated physics; GIRO's k is a labelled annotation.

### F5. Q2's "fresh 6/24" is confounded by MIP budget and seed (important)
Fresh and warm reach the identical certified LP. The difference is entirely in integer recovery from pools of different size, under a single 1 h MIP run. F-longer-MIP evidence shows 1 h runs vary by 3+ buses across trees. At k=15 none of the six fresh misses is a pool proof. So the finding is real but misattributed: it is "an LP-optimal fresh pool is a poor IP pool within 1 h", not "fresh CG cannot catch up". The doc also concedes the warm chain uses GIRO's duty grouping — information unavailable in deployment. A random-trip-grouping chain control is the proper test of sequential warm-starting as a *method*.

### F6. Q4 rests on n = 1, pre-screened, and produces unrealistic charging
- Selection: the 5 duties are from the 28/40 that replay; the effect size on rejected/harder duties is unknown.
- Realism: fee 0 yields 46 starts on 5 buses, including a 26-second, 2.5 kWh event. The paper will be asked why a 3-minute minimum (GIRO's own rule) wasn't used instead of a fee. Report charge-start counts for **both** arms; they are absent from the CG-arm `summary.json` I could find.
- Missing comparator: `original_giro` (as-is under tariff) exists in the cluster JSON. Without it the headline cannot decompose value into smart charging vs smart scheduling — the core DR question.
- Tariffs: `peak08/12/18_h26.csv` are synthetic peak shapes. At least one real day-ahead series (Nord Pool SE3) would anchor the numbers.
- 350 kW vs the 240 kW chain baseline: cross-referencing Q4 with Q1–Q3 is not possible.

### F7. The scaling thread and the DR thread never meet (structural)
All 102 chain rows are flat-tariff. The DR benefit is measured only at k=5. Nothing tells us whether the 2–7% survives at k=15–30, or whether tariff-aware pricing changes CG convergence. For an EVSP-**DR** paper this is the largest missing experiment.

### F8. The full instance is not attempted
Chains stop at k=32 (~750 trips). Partille full = 40 duties / 987 trips; Frölunda 61 tasks. One long (24–48 h) CG on full Partille — even uncertified, with the F3 bound — would be the natural headline and would test whether the 4 h wall is the binding constraint.

### F9. Bookkeeping gaps (for a cheap model)
- Register: `charge_start_cost = unknown` for all 324 chain rows; `tariff_path = unknown` for 222.
- `all_chain_extension_results.csv` lacks a Lagrangian LB column and a `giro_duties_feasible_in_model` count.
- cumulative_budget `status_…/README.md` still says "largest verified match 25" while `doc_after.md` says 31/32 (late results "awaiting scientific verification"). Reconcile or label.
- `existing_pair_summary.json` capacity-pricing pairs: both arms hit 3 h with no certificate — correctly reported as non-comparable; keep it that way.

---

## 3. Comparison hygiene checklist (apply to every table)
Report per comparison: objective (fleet weight, electricity, fee), sense (cover/partition), initialization, kW at depot vs terminal, battery, floor, terminal-energy rule (aggregate vs per-bus), tariff file hash, MIP seed and time limit, and the four-level status: scheduler done / CG certified / pool-proven / physically validated partition. Never mix 1 h and 3 h MIP results in one cell without labelling. Never add speedup percentages.

---

## 4. Prioritized experiments

**P0 — no solver time, run on Luna/Astra now**
1. Covering→partition conversion + replay on all 128 selected schedules so cost columns are partition costs; fleet count is guaranteed unchanged (empty-bus argument). (F1)
2. Lagrangian fleet LB for all 54 uncertified CG rows from `weighted_lp_objective`, `last_pricing_reduced_cost`, incumbent fleet. Add column to CSV. Verify rc is exact global min. (F3)
3. **First**: grep pinned commit a0e0bb76 for a depot-specific rate; report whether PARX/KEX charge at 60 or 240 kW. Then replay GIRO's own duties under the chain baseline for each chain's k-set; report feasible count per (chain, k). (F4)
4. Jaccard(selected route trip-set, nearest GIRO duty) for LP = k vs LP = k−1 cases. (F2)
5. Surface `original_giro` cost and charge-start counts for both arms in the zero-fee comparison; add per-bus min SOC and ending SOC. (F6)
6. Fix register fields in F9.

**P1 — MIP only, hours of cluster, no new CG**
7. Fresh k15 pools ×6: 3 h MIP × 3 seeds each. If ≥ 4/6 match, Q2's headline must change. (F5)
8. w5_k31 pool: 12 h MIP or a diving heuristic to settle 30 vs 31 in pool. (F2/F4)
9. Warm k31/k32 misses (w1/w3/w4/w5 at 32): 3 h × 3 seeds to measure MIP variance before calling any gap a pool limitation.

**P2 — CG, one to a few days**
10. Tariff-responsive CG at k=15, 6 chains, peak08/12/18 + one real price series, with fixed-duty and GIRO-as-is comparators, 3-min minimum charge instead of fee. This joins the threads. (F6/F7)
11. Chain 5 to k31 under 60 kW depot + 15% floor + group-specific batteries: does LP return to 31? (F4)
12. One chain to k15 with **random** trip groupings of matched size: does warm-starting help without GIRO's grouping? (F5)

**P3 — expensive, one job each**
13. Full Partille (40 duties) CG, 48 h wall, with F3 bound. (F8)
14. Frölunda k-ladder to test generalization.

---

## 5. What I did not check
Code paths (pricer exactness, covering→partition cleanup, terminal-energy dual) — I relied on manifests and the design doc `outputs/zero_fee_full_cg_design_20260915/README.md`. `continuation_gaps15_20260916` (post-collection). Google Doc tabs beyond `doc_after.md`. Solver logs on Unicorn. If P0-2 changes a headline, have a cheap model re-derive the bound from the raw CG status JSONs rather than the CSV.
