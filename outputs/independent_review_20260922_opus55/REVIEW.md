# EVSP-DR independent review: code, evidence and next steps (22 Sep 2026)

I read the code, configs, raw CSVs and native Gurobi logs before the project's own interpretations. I did not run anything or change any files.

## Overall assessment

- **Numbers:** the evidence behind the five paper figures is recorded carefully, and nearly every number in `RESULTS_PREVIEW.md` reproduces from the raw CSVs and logs. I found no coding error that invalidates a reported result.
- **Main problem:** the proposed paper narrative is weaker than its figures suggest, for four reasons:
  1. **Sequential chains use GIRO's duties.** The "sequential inheritance" result (24/24 targets) comes from chains built by adding one complete GIRO duty at a time. The preview does not say this. The advantage has not been tested without that structure, and it does not hold at k≥33.
  2. **The main finding is already known.** "A pricing certificate doesn't give a useful integer pool" is the standard weakness of pricing first and branching afterwards (price-and-branch). The novelty has to come from a validated fix. Today that is a 4-case pilot whose fleet cap was set to the target.
  3. **The fleet target is effectively known in advance.** For k5/k10 the LP fleet weight equals a trivial peak-concurrency bound, which equals k. So "hitting the target" means recovering a known optimum. It is not a comparison with GIRO.
  4. **The model is a relaxed abstraction of GIRO's operation, and has no demand-response content yet.** It uses a flat tariff, has no end-of-day battery (SOC) requirement, zero reserve, 240 kW depot charging, no charger limits and flattened deadheads.
- **Viable paper:** "integer-directed column generation for electric-bus scheduling, evaluated at matched end-to-end budgets on realistic physics." Most of the tooling exists; the decisive experiments have not been run.

## What the evidence does establish

| Finding | Evidence |
|---|---|
| Exact pricing and the master are consistent: fresh and sequential LP objectives agree in all 24 cases (max relative difference ~7e-13). | `figure1_paired_budget.csv`, `fresh_weighted_lp_objective` vs `sequential_weighted_lp_objective` |
| Pool composition, not search time, blocks k8 integer recovery. Four fresh pools are proven 9/9; adding 7–8 routes gives proven 8/8; each augmented journal starts with the byte-identical original. | I checked `k8_witness/c1_k08/control/391804_r0/gurobi.log` myself: l.19 has 39,940 columns, l.53 root objective 8, l.150–156 best 9 / bound 9. See also `pool_identity_remote_audit.json`. |
| Witness routes are unattractive at the final duals: 39/40 trip sets are absent from the fresh pools, 37/40 have reduced cost above 1e-4, 34/40 are inherited. | Recounted from `witness_route_audit.csv` |
| More MIP time on the fresh k15 pools does not close the gap (16–19 buses, bound 15, after 43,200 s). | `k15_12h_summary.csv` |
| The indexed-replay and skipped-incidence changes keep iteration counts and objectives exactly the same. | `figure4_controlled_algorithms.csv`, rows with `lp_objective_difference=0.0` and equal iterations |

## Prioritized findings

### F1 (high): Sequential inheritance uses GIRO's duty structure; the preview omits this and it fails at scale
- Each chain step adds one whole GIRO duty's trips (`warm_chain_p5/README.md:5`; `CHAIN_CHARACTERISTICS.md:11-28`, e.g. `k04_p1→13324muw`). No GIRO route columns are injected; k2 starts from single-trip routes. But every sub-instance is exactly a union of complete GIRO duties.
- The team's Doc says this elsewhere ("uses known duty grouping", `large_chain_cleanup_screen_20260916/doc_after.md:7`). The paper claim and caption for Figure 1 do not.
- The overlap audit (`independent_review_20260916/execution/f2/README.md:10-11`) finds selected routes are only partly GIRO-like: mean nearest-duty Jaccard similarity 0.64, with 11.5% exact matches. So the method is not simply reassembling GIRO duties, but the effect of the duty grouping is unmeasured.
- **The advantage does not scale.** At k33–35 every sequential chain misses its target: 35–42 buses against targets 33–35, no CG certificate after 4 h, graph builds of 28,000–66,000 s and about 49 GB of memory (`monitor_20260922T195842Z/operations/cg_endpoints.csv`, `mip_endpoints.csv`).
- **Consequence:** "Sequential attains 24/24" is a result for k≤15 with GIRO's grouping. It is not yet evidence for a method you could deploy without an existing schedule.

### F2 (high): The fleet target at k≤10 is a known optimum
- The LP fleet weight is exactly k in all 24 Figure 1 cases.
- The selection manifest's peak-concurrency lower bound is also k for every k5 and k10 chain, and 14–15 at k15 (`CHAIN_CHARACTERISTICS.md:38`).
- So the LP adds little beyond a trivial bound, and "target matched" means recovering a solution known to exist.
- The interesting cases are where the LP differs from concurrency or from GIRO's count, e.g. the k33–35 runs with RMP weight k−1. Frame the paper around these.

### F3 (high): The baseline model is not comparable with GIRO, and "charging cost" is not an economic quantity
The baseline for Figures 1–4 (`RESULTS_PREVIEW.md:65`, operations README l.71) differs from the documented operation (`GIRO_EMAIL_ATTACHMENT_AUDIT.md:15,56-57,66-72`):

- **Reserve:** zero, versus a hard 15% floor. Saved schedules use the last kWh: 40/487 route occurrences fail if the battery drops from 240 to 239.01 kWh (`research_followup_20260921/README.md:7`).
- **Depot charging:** 240 kW everywhere, versus 60 kW at PARX.
- **Charger limits:** none, versus 1–2 opportunity chargers per site. An earlier generated schedule exceeded that inventory at every site.
- **Deadheads:** `build_problem` reads `par_ref_dhd.csv`, which has 49 symmetric reference-place pairs at an averaged base duration, with no time-of-day intervals (`audit_giro_known_columns.py:142-160`). Example: pair 3127–13330 is modelled as 16.5 min when one direction takes 18 (`par_ref_dhd.csv:12`). GIRO confirmed the matrix is interval-dependent.
- **Route restrictions:** direct trip-to-trip gaps are capped at 57 min (`pricing_dp_og.py:795-802`). Longer idle periods must pass through a station and buy at least one SOC grid step. A bus that is already full cannot do so at all (`event_pricer_network.py:495`).
- **No end-of-day energy requirement:** minimizing charging rewards ending the day empty, and extra buses bring free initial energy. Under the flat tariff, charging cost mostly measures deadhead energy and the start fee.
- **Covering, not partitioning:** up to 300 trips / 420 extra trip assignments are duplicated at k34 (`mip_endpoints.csv`, w4_k34). Fleet counts are unaffected, since a duplicate can be driven empty, but costs are not partition costs.

The Figure 1 caption states the baseline itself; what is missing is an explicit statement that these results say nothing about GIRO-comparable fleets or economics.

### F4 (medium-high): The integer-directed pilot is suggestive, not a result
- 3 of 4 dives reached 8 buses. Limitations:
  - The four cases were selected from pools already proven at 9.
  - One seed per case.
  - The C1 treatment took 3,692 s.
  - C1's dive incumbent was not passed to its final MIP (fixed later, as a separate run).
- The dive master had `fleet_cap=8`, the GIRO target (`advisor_diving_pilot_20260920/README.md:23`; `diving_pricing_pilot.py:31-34,1020`). Here that equals ceil(LP weight), so the cap is defensible, but the rule must be stated as ceil(LP bound), and the method must handle the case where that cap is infeasible.
- `run_exact_dive.py` (the other dive) re-runs CG on the residual instance each round and needs a graph per round. That will not scale without graph reuse.

### F5 (medium): Figure 4's percentage effects are within timing noise between allocations
- The same arm on the same input, with identical iteration counts, varies a lot across allocations:
  - Arm C on w1_k08: 15.76 / 15.70 min in the pool and master contrasts, but 11.09 / 10.78 min in the full_index contrast (rows 4–7, 20–21).
  - Arm B on w1_k08: 30.5–41.3 min.
  - Arm B on w3_k15: 48.1 vs 33.5 min (rows 16–17).
- Paired, order-reversed runs help, and the direction is consistent (6/6 pairs). But "12.4–18.1%" and "9.3–14.6%" from 3 cases × 2 repeats should be reported as consistent-direction results with the observed variance. Or add repetitions and CPU time.
- The pool contrast (40.6–64.5%) changes the pool itself, as the caption says.

### F6 (medium): Figure 5 conflates storage format with graph reduction
- `figure5_packed_benchmark.csv` shows **1,180,257 retained arcs for packed vs 3,319,685 for explicit** on the "same lattice".
- Part of the 530.8× pricing speed-up is therefore fewer arcs, presumably dominated ones removed. Equal minimum reduced costs are shown for only 5 dual vectors.
- State the dominance argument. It only holds without station-capacity duals, which the caption acknowledges.

### F7 (medium): Free certified bounds are not being used
- `fleet_bound.py` (early-stop branch `689d3646`, `advisor_early_stop_20260917/README.md:16-27`) implements a valid Lagrangian fleet floor from the RMP objective, the exact minimum reduced cost and a per-route cost bound Q.
- The k33–35 tables still say route weights "are not certified bounds".
- My arithmetic from recorded fields, assuming Q≈1.04e3 as in the w5_k31 example (not verified per instance), gives floors equal to the RMP weights: w5_k33 ≥ 32, w1_k35 ≥ 34, w3_k35 ≥ 35. This turns "uncertified" rows into certified gaps. Several floors are below GIRO's count, which is likely a relaxed-physics artifact (see F3).

### F8 (low to medium): Code and provenance notes
- The CG for the k33+ endpoints ran at pin `a0e0bb7` with `git_dirty=true` (operations README l.71). A diff of that working tree is needed before paper use.
- The pricing-exactness tests enumerate paths only on a two-trip fixture over the same graph (`tests/test_terminal_full_cg.py:22-52`). They check the dynamic programme but not whether the graph is complete relative to the model. Physical replay checks that generated routes are feasible, not that none are missing. An independent brute-force route enumerator on a roughly 10–15 trip instance would strengthen the "certificate" claim.
- The pool MIP de-duplicates by trip set under `--cover` (`run_exact_pool_mip.py:255-258`). The capacity CG already keys routes by charging plan (`run_capacity_speed_event_cg.py:177-185`); keep capacity MIPs on that path.

## Check of the claims in RESULTS_PREVIEW

| Claim | Numbers | Verdict |
|---|---|---|
| Figure 1: fresh 6/24, sequential 24/24 | ✓ | Accurate, but must add the GIRO duty-grouping caveat, the fact that fresh certified using only 3–30% of its CG allowance, and that sequential pools are about 2× larger (78,053 vs 39,940 columns for c1_k08, `witness_analysis.json`). Cases are 6 independent chains, not 24 independent instances. |
| Figure 2: four pools proven 9; adding routes gives 8; 3/4 pilots reach 8 | ✓ | "Independently" should say the fleet cap was 8. Selected cases, n=4. |
| Figure 2 mechanism counts (39 / 37 / 34) | ✓ recounted | Descriptive. It is the standard price-and-branch gap; cite that literature. |
| Figure 3: 16–19 buses, bound 15 | ✓ | Fine as stated. |
| Figure 4 ranges and fleets 9→8, 11→10, 17→15 | ✓ | Needs a note on timing variance (F5). |
| Figure 5: 2.90× / 14.1× / 530.8× | ✓ | Disclose the 2.8× arc reduction (F6). |

## Recommended actions, in order

1. **Zero compute:** apply `fleet_bound.py` offline to every uncertified endpoint (k16–k36 and strict runs), after confirming the recorded minimum reduced cost comes from the same iterate. Add the floors to the endpoint tables.
2. **Zero compute:** rewrite the Figure 1 claim and caption (GIRO grouping, known-optimum framing, fresh budget left unused), and the Figure 4 and 5 captions (F5, F6).
3. **Decisive control for F1:** repeat the k8/k10/k15 chains on all 6 chains, ordered by random or time-window trip groupings of the same size rather than GIRO duties. Add two reference arms: a pool seeded with GIRO's own duties as an upper reference, and matching-based blocks as a GIRO-free structure.
4. **The algorithmic contribution:** one integer-directed method (a dive inside the same master, or branch-and-price with Ryan–Foster branching). Set the fleet cap from ceil(LP), not the target. Run the full 24-case panel plus k20–k35, with matched end-to-end wall budgets against fresh+MIP and sequential, using 3 or more seeds.
5. **Realistic physics before any GIRO or economic claim:**
   - 15% reserve, 60 kW depot, the documented charger counts with capacity duals.
   - Interval deadheads from `Par_DHD.xlsm`.
   - An end-of-day energy rule or an overnight recharge cost.
   - Explain why 12 of 40 GIRO duties fail model replay (`independent_review_20260916/REVIEW.md:66`).
6. **The demand-response question:** time-of-use tariff plus demand charge on a real SE3 price series. Compare GIRO's schedule as-is, fixed-duty re-charging, and joint optimization under identical physics and partition costs, at k≥15.
7. **Scale:** graph builds of 8–18 h per instance are the bottleneck. Test SOC grid sensitivity (2.5 vs 5 vs 10 kWh; measure the LP and integer change) and reuse packed graphs.
8. Convert the covering solutions to partitions and replay them, so cost columns are partition costs (`terminal_duplicate_cleanup.py` exists).

## Artifacts missing locally that a paper needs

- **Column journals and `cg.json` files** for the Figure 1/2 fresh, sequential and augmented pools. Only hashes and remote paths are local, so the mechanism and pool MIPs cannot be recomputed here.
- A **diff or archive of the dirty `a0e0bb7` tree** used for the k33+ CG runs.
- **Per-instance Q values and final-iterate duals** for the Lagrangian floors in action 1.
- **CPU-time (user+system) records** for the Figure 4 arms (noted as missing in `experiment_settings.json:52`).

## Not reviewed

The Google Doc and Slides, Frölunda data, the tariff/fee0-vs-fee5 campaigns in detail, and most historical campaigns before 13 Sep.
