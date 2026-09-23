# Follow-up evidence audit of the Opus 5.5 review

23 September 2026 (UTC). Auditor: Claude (Opus 5.5, 1M context), Claude Code session in `/Users/nadan/Documents`. Brief: [FOLLOWUP_AUDIT_BRIEF.md](FOLLOWUP_AUDIT_BRIEF.md). This is an evidence audit, not a fresh independent review. The original [REVIEW.md](REVIEW.md) stays frozen, and I build on [ASSESSMENT.md](ASSESSMENT.md), [audit_bounds.md](audit_bounds.md), [audit_long_wait.md](audit_long_wait.md) and [audit_counts.json](audit_counts.json) without redoing their checks.

## Prior involvement (read this first)

I am not independent of several items under review:

- **Diving pilot.** I reviewed and committed the diving-with-pricing pilot (`a3392e2c`, branch `codex/diving-pricing-20260919`), fixed its cluster bridge, and submitted the original four-case pilot (jobs 586633–586644). The **fleet cap of 8 was set by that design** (`../independent_review_20260916/advisor_diving_pilot_20260920/README.md:23`). Opus-F4 criticises exactly this choice.
- **Earlier recommendations.** I wrote the 16 September review ([../independent_review_20260916/REVIEW.md](../independent_review_20260916/REVIEW.md)), the 19 September handoff that recommended "diving-with-pricing, not near-zero-rc enrichment", and the fleet-certified early stop containing `fleet_bound.py` (`689d3646`), which Opus-F7 proposes to apply.
- **Capacity short-circuit.** I implemented it (`codex/capacity-shortcircuit-20260917`).
- **Not mine.** I did not run the 21 September matched replication, the k15 wave or the k33–40 campaigns.

Where my own design is being judged, I state the evidence and let it fall where it does.

## Decision summary

1. **Opus-F4 (fleet cap from the target): verified on provenance, qualified on consequence.** Every dive cap was the GIRO duty count k: 8 at k8, 15 at k15. So the evidence is a **target-feasibility test**, not a target-free algorithm. However, a target-free rule gives the same cap in all 11 case-seed runs:
   - At k8, cap = peak trip concurrency = 8 in all four instances. This is a solver-free, model-independent lower bound, recomputed here from the raw CSVs.
   - At k15, cap = the Lagrangian fleet floor = 15. That floor comes from each fresh pool's certified CG record, with Q derived per instance, and remains 15 for any Q below about 7,180 (actual Q is 944–964).
   - Concurrency gives only 14 for k15 C1 and C3, so the LP floor, not concurrency, is what supports 15 there.

   Therefore 7/8 at k8 and C3's 15 at k15 are **reached lower bounds**, i.e. event-model fleet optima. What is untested is behaviour when the floor is *not* attainable, which needs an escalation rule.
2. **Opus-F2 (known optimum): verified at k5/k8/k10, qualified at k15, and stronger than either reviewer said.** Each of the 24 sequential hits reaches a proved lower bound, so the Figure 1 targets are **event-model fleet optima**, not only pool optima. The caption's "not full-model integer optimality" is now too cautious for the fleet (not for charging cost). This turns the Figure 1 contrast into "fresh+MIP misses a *proved* optimum in 18/24 cases".
3. **Opus-F7 (unused bounds): arithmetic reproduced with instance-derived Q.** Three of the conditions `audit_bounds.md` required are now met or narrowed:
   - Q is computed per instance (1,034–1,049 for k33–36, not an assumed 1,040).
   - The dirty-source flag is explained: untracked data only, and tracked files were clean.
   - Iterate pairing was already verified for w1_k35 and w3_k35.

   The floors equal each run's RMP route weight in 11/11 k33–36 endpoints. They are still **conditional numerical floors** in the restricted event graph, not full-model certificates, until the reporting wrapper refuses unpaired records.
4. **Opus-F1 (sequential advantage fails at scale): open, not refuted.** No fresh arm exists at k≥31, so "misses its target" is established and "the relative advantage disappears" is not.
5. **Opus-F3 (operating assumptions): verified against raw attachments, with omissions (§3).** Every listed baseline deviation is real. The first review missed the stricter campaigns (strict k16–19, matched k5) and two raw requirements that bias fleet counts: GIRO keeps 18E1/18E2 blocks separate, and there is a route-specific minimum layover. The 57-minute cap has no raw basis.
6. **Opus-F8 (dirty source): resolved for the two campaigns checked.** Both `git_dirty=true` pins, `a0e0bb7` for k33–36 and `e091a4d` for fresh k8/k15, are HEAD plus untracked instance-data directories only. Each launcher asserts a clean tracked tree before submission. The brute-force enumerator Opus-F8 requests already exists for the older DAG pricer (PR #34), but not for the event pricer.
7. **Novelty: see §5.**

## Findings table

| Opus finding | Status | Primary evidence | Effect on our claim | Smallest next action |
|---|---|---|---|---|
| **F4** cap set to target | **Verified** (provenance); **qualified** (consequence) | Pilot: `advisor_diving_pilot_20260920/README.md:23`. Replication: `research_management_20260921/integer_columns/PREREGISTRATION.md` ("cap8"), `results/*/treatment_*/dive_argv.json` `--fleet-cap`. k15: `integer_columns/k15/register_wave.py:11` `fleet_cap=15` beside `k15/manifest.json` `target_k: 15`. Cap is a hard master row whose dual μ enters pricing and a fixing stop rule: `.codex-work/integer-columns-20260921/src/diving_pricing_pilot.py:31-34,151,202,818,880` at `c50e5f20` (k8 runs used `1be819f0`; the diver differs only in budget parameterization). Target-free bounds: see §1. | "Recovers 8 without known routes" holds, since no route import is confirmed by `integer_audit/README.md`. The method was *told* the fleet size, but that size is independently provable here. It is not evidence for instances whose floor is unattainable. | Replace "target cap" with "cap = certified fleet floor" only prospectively. Relabelling history is not acceptable. Run the cap-escalation control (§6, E1). |
| **F2** target is a known optimum | **Verified** k5/k8/k10; **qualified** k15 | Concurrency LB = k for every k5/k8/k10 cell (`selection_manifest.csv` field `peak_concurrency_lb`; recomputed for k8 p1/p3/p4/p5: 8, with touching intervals allowed). k15 p1/p3 = 14 and p2/p4/p5/p6 = 15. The k15 floor of 15 comes from certified fresh CG (§1). Sequential k-bus covers: `figure1_paired_budget.csv` `warm_buses`, `warm_fleet_proven`. | The bound is known *before* solving only at k≤10. At k15 it needs the certified LP. Correction of my 16 Sep wording: the 12/40 replay failures were GIRO's *original* charging replayed at 350 kW. At 240/240, all 42 GIRO trip sequences admit feasible charging after reoptimization (`../independent_review_20260916/execution/README.md:21`), so GIRO's k is a valid upper bound in that reoptimization model. Whether each GIRO duty is representable in the *restricted event graph* (57-minute cap, positive-charge bridges, SOC grid) was not checked. The upper bound used here therefore comes from our own sequential event-graph covers. | Report "fleet optimum (event model, covering)" for the 24 sequential hits, with the bound's source per case. |
| **F7** certified floors unused | **Qualified**: arithmetic reproduced; certificate conditional | Instance Q = 240×26×0.0992 + 5(q+1), with q from interval scheduling on each `chain_extension_40/inputs/w*_k*.csv`: 1,034–1,049. Floors from `monitor_20260922T{195842,235958}Z/operations/cg_endpoints.csv` `weighted_RMP` and `last_pricing_min_rc`: w5_k33 32, w1_k35 34, w3_k35 35, w2_k34 33, w4_k35 34, w6_k34 33, w1_k36 35, w2_k35 34, w3_k36 36, w4_k36 35, w6_k35 34, equal to each route weight. Tariff hash `1f51f2e1…` matches the local `hourly_prices_flat.csv`. `STATION_PRICE_MULTIPLIER` is defined in `config.py:57` but not referenced by the pricer. | These floors are **below GIRO's k in 9/11** endpoints (all except w3_k35 and w3_k36). Against k35 incumbents of 42/37/36/42 (C1–C4), the honest gap is 1–8 buses to a floor, not "missed target". I agree with `audit_bounds.md` that a floor below k does *not* show that fewer buses are feasible. Before adoption, the collector's pairing of `final_lp` with `final.min_rc` must be replaced by the priced-iteration record, which is verified for w1_k35 and w3_k35 only. The mispairing risk is numerically small. The reduced cost would have to fall below δ_crit = ((floor−1)(M+Q)+1−z)/K, which is −1,808 to −2,166 across the 11 endpoints, before any floor drops by one. Recorded values are −0.03 to −56. | A solver-free wrapper that reads the *priced iteration* record, refuses `min_rc=None` or missing artificials, computes Q from inputs, and emits floors plus scope. No compute. |
| **F1** duty grouping; fails at scale | **Verified** (grouping, now disclosed in the caption at `RESULTS_PREVIEW.md:15`). "Fails at scale" is **open** | No fresh or `base` arm exists at k≥31 in either endpoint table (all rows are `w*` sequential). The k33 MIPs give 34–39 buses against RMP route weights of 32–33 (only w5_k33 has a computed floor, 32). | Sequential misses k at k≥33 under its budget. Whether fresh does better or worse there is unknown. A grouping-free control is still missing for all k. | Use the grouping-free chain control (E2) at k8/k15 only, since k≥33 graphs cost 9–23 h each. |
| **F3** model ≠ GIRO operation | **Verified**, with two omissions (vehicle-group separation, minimum layovers) and one miss (stricter campaigns exist) | Raw attachment trace in §3. Strict `research_management_20260922/operations/strict_k19/cg_result.json` `physics`. Matched k5 `week_20260921/cleanup_physics/README.md:11-13`. 350 kW in `research_day_20260918/matched_k3/manifest.json:63` and `meeting_20260917/noon_fixed_retry/verified/summary.json:12`. | No fleet claim is GIRO-comparable. The k5 charging comparison is the only near-matched one. The 350 kW campaigns overstate shifting flexibility. | E2/E3 below keep baseline physics. The first GIRO-comparable fleet run must separate 18E1/18E2 and enforce reserve and 60 kW PARX. That run is not proposed here. |
| **F5** Figure 4 timing noise | Covered by ASSESSMENT (supported qualification) | — | Not rechecked | Caption as ASSESSMENT proposes |
| **F6** Figure 5 arc reduction | Covered by ASSESSMENT (verified; already in guide) | — | Not rechecked | — |
| **F8** dirty source, oracle, dedupe | **Dirty flag resolved**; oracle **partly exists** | Unicorn, 2026-09-23 00:48 UTC, read-only. `/home/nc437/ladder-lite/chain_extension_33_40_20260921/code`: HEAD `a0e0bb7`, zero tracked changes, untracked `data/scale_ladder/instances/chain_extension_{31_32,33_40}…/`. `/home/nc437/ladder-lite/full_pool_recovery_20260912/code`: HEAD `e091a4d`, zero tracked changes, one untracked CSV. Launch guards: `chain_extension_33_40_20260921/campaign.py:72,251` and `cumulative_budget_20260913/campaign.py:25` assert `git status --porcelain --untracked-files=no` is empty. Oracle: PR #34 (open, head `d63a4da`) `analysis/tiny_differential_20260821_duals/REPORT.md`: 7,680/7,680 exhaustive pricing matches for the *DAG* pricer, and exact CG's final-pool integer fleet agrees in 216/240 tiny cases. | Present state plus launch-time guards support "tracked source = pinned commit". The oracle is for `pricing_dp_og`, not `event_pricer_network`, so event-graph completeness remains untested by enumeration. | Record the untracked-only explanation in provenance. Extend the PR #34 oracle to the event pricer (E3 candidate). |

## 1. Opus-F4 in detail: what the cap did, and what a target-free rule gives

**Mechanism.** The dive master has cover rows, a hard `sum x <= fleet_cap` row, and fixings `lb = ub = 1`. Pricing subtracts the fleet dual μ ≤ 0. The dive retreats once the number of fixings reaches the cap (`diving_pricing_pilot.py:818`), and it records an integer solution only if it uses ≤ cap routes (`:880`). The cap is therefore not a passive filter: it changes duals and hence the columns generated. A run at cap = k answers "is there a k-bus cover reachable by this dive?".

**Cap provenance, by stage.**

| Stage | Cases | Cap | Stated source |
|---|---|---:|---|
| Original pilot, 20 Sep | C1/C3/C4/C5 k8 | 8 | GIRO target (my design) |
| Matched replication, 21 Sep | same, seeds 20260921/22 | 8 | `PREREGISTRATION.md`: "cap8" |
| k15 wave, 21–22 Sep | C1/C3/C5 k15, one seed | 15 | `register_wave.py:11`; `target_k: 15` |

**Target-free bounds on the same instances.** Instance identity is checked by hash: each `figure1_paired_budget.csv` `input_sha256` matches a `selection_manifest.csv` row, and the file was rehashed. The Unicorn payload fields were read at 00:48 UTC, and the payload SHA-256 prefixes match `fresh_cg_source_sha256`.

| Case | Concurrency LB | q | Q | Certified z (final record) | min rc | Artificials | Lagrangian floor | Floor stays at k while Q < | Cap used |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 k8 | 8 | 50 | 874 | 800383.688 | −5.2e−10 | 0 | 8 | 14,340 | 8 |
| C3 k8 | 8 | 54 | 894 | 800212.687 | −6.6e−5 | 0 | 8 | 14,316 | 8 |
| C4 k8 | 8 | 53 | 889 | 800359.818 | −1.1e−9 | 0 | 8 | 14,337 | 8 |
| C5 k8 | 8 | 55 | 899 | 800431.772 | −7.3e−10 | 0 | 8 | 14,347 | 8 |
| C1 k15 | **14** | 68 | 964 | 1500717.373 | −2.9e−9 | 0 | 15 | 7,194 | 15 |
| C3 k15 | **14** | 64 | 944 | 1500507.420 | −1.2e−8 | 0 | 15 | 7,179 | 15 |
| C5 k15 | 15 | 65 | 949 | 1500669.159 | −8.9e−9 | 0 | 15 | 7,191 | 15 |

- **Formula.** Q = 240 kW × 26 h × 0.0992 + 5(q+1), per `fleet_bound.py:101` at `689d3646`. q is the maximum number of pairwise non-overlapping trips, which bounds the trips on any route.
- **Scope.** All seven payloads are `e091a4d`, flat tariff, 240/240, `min_soc_frac 0`, cover, singleton start, `certified_rc_optimal: true`.
- **Validity conditions.** The floor holds in the **restricted event graph**, which includes the 57-minute cap and positive-charge bridges. It uses the numerical guard of 1 unit. Payloads re-solve the final pool (`final_lp_source: final_pool_resolve`), but the `final` scalar record pairs `lp_obj` and `min_rc` from the same priced iteration.
- **Robustness.** Even the loosest admissible Q (q = number of trips) stays below 2,600. The conclusion does not depend on how carefully Q is estimated.

**What this supports.**

- A **legitimate target-feasibility test**: given a proved floor, can integer-directed pricing reach it from a fresh pool? At k8 the answer is yes in 7/8 case-seed runs, versus 0/8 for the unchanged pool at equal nominal budget. At k15 it is yes in 1/1 completed paired trial (C3). C1 and C5 had software stops, and their supplemental recoveries gave 18/17 against control 19/16 (`monitor_20260922T035403Z/README.md`, Doc current-results tab). They are not paired trials and must not be counted as either.
- Because every reached cap equals a proved lower bound, each hit is an **event-model fleet optimum**, which is stronger than the "finite-pool proof" wording now used.
- It does **not** support a general target-free algorithm. No run started from an unattainable floor, so the fallback path (exhausted dive → raise cap) has never executed. The C4/20260921 miss stopped at the wall clock without exhausting the search, which shows that a miss conveys no infeasibility.
- Replacing "cap = target" with "cap = ceil(route weight)" would be wrong in general, as `audit_bounds.md` notes. It happens to coincide here only because Q ≪ M/k.

**Escalation rule (not run).** Set cap₀ = max(concurrency LB, Lagrangian floor from a certified or paired-iterate record). Dive. If the dive exhausts or times out without an incumbent, set cap ← cap + 1 and resume, keeping the generated columns and debiting all time to one end-to-end budget. Stop when an incumbent at the current cap exists, when the final pool MIP proves the cap, or when the budget ends. Report the starting floor, the final cap and every escalation. This is a heuristic: a failure at cap c never proves that no c-bus cover exists.

**Smallest decisive control.** Two single-factor variants on the four k8 pools, one seed each, same 3,600 s budget, no compute authorised here:
- **(a) cap = 9 (floor + 1).** If the dive plus MIP still finds 8, the cap is not what supplies the target. If it returns 9, the cap row is doing the work, which is legitimate for a floor-targeted method but must be stated.
- **(b) cap = 7 (below the floor)** with the escalation rule on. This measures the cost of starting infeasible, which is the realistic case when the floor is not tight.

Of the two, (a) is the more decisive for Opus-F4 and needs no new code. (b) tests escalation machinery that must first be written.

## 2. Opus-F1, F2 and F7: proof and attribution vocabulary

Seven distinct objects appear in the Doc, Slides and reviews. They must not be merged:

| Object | Example | What it certifies |
|---|---|---|
| Full pricing certificate | fresh k8/k15 `certified_rc_optimal: true`, min rc ≥ −1e−4 | LP optimum of the **restricted event graph** (57-min cap, positive-charge bridges, SOC grid 2.5 kWh, covering) |
| Numerical fleet floor | Lagrangian, §1 table; conditional k33–36 floors | Integer fleet ≥ floor in that same graph, with guard 1.0, under a verified Q and a paired priced iterate |
| Model-independent floor | Peak concurrency | Any schedule where a bus serves one trip at a time |
| Finite-pool proof | Gurobi best = bound in stage 1 | Optimum *over saved columns* only |
| Observed feasible schedule | Native route replay plus at-least-once coverage | An upper bound in the covering event model. It does not cover duplicate removal or shared capacity. |
| Event-model fleet optimum | Floor = feasible schedule (24 sequential hits, 7 dive hits, C3 k15) | Fleet optimum for that model |
| Broader physical feasibility | Strict / k5 matched physics, continuous repair | Separate models; not implied by any row above |

**The 57-minute cap.** `audit_long_wait.md` verifies the restriction in source. PR #34 adds a constructed tiny instance where lifting it from 57 to 58 minutes reduces the exhaustive optimum from 2 buses to 1. That confirms the restriction can bind, but it is **not** a real-instance effect estimate. Rows 1 and 2 are scoped to the capped graph. Row 6 depends on the floor's source:

- **Graph-independent.** Where the floor is peak concurrency (all k5/k8/k10 cases, k15 p2/p4/p5/p6: 22 of the 24 sequential hits, and all seven k8 dive hits), it holds for any schedule. The k-bus cover is feasible in the capped graph, hence also in any relaxed graph with the same physics. These fleet optima therefore survive lifting the 57-minute cap or adding zero-charge waits.
- **Graph-dependent.** Only C1 and C3 k15 (concurrency 14 < floor 15, including C3's dive hit) and the k33–36 floors depend on the graph. An uncapped graph could admit a 14-bus cover there.

**Separating two different F1 claims.**
- "Larger sequential chains miss k": verified (k33 34–39, k35 36–42).
- "The advantage over fresh disappears": no evidence either way.
- The conditional floors suggest a third framing: at k35 several floors are 34, one below k. The target k itself is not proved attainable there, and the MIP gap to the floor is 1–8 buses.

## 3. Opus-F3: primary operating assumptions

**Method.** A subagent read every Partille raw attachment in `../meeting_20260910/giro_email_sources/`:
- `Par_Notes.docx`, including its four embedded images;
- `Par_VehicleDetails.xlsx` and `Par_DHD.xlsm`, read with openpyxl;
- the Partille December 2023 PDF, read with PDFKit.

Frölunda was checked only through `FDL_Notes.docx`. I spot-checked the model-side settings myself: strict k19 `physics`, the matched_k3 manifest, the fixed-duty summary and `cleanup_physics/README.md`. The raw correspondence is paraphrased here, with locations only. The email bodies behind `GIRO_EMAIL_CONFIRMED_ASSUMPTIONS.md` were not available, so that summary is unverified.

**Classes:**
- **R**: explicit operating requirement in a raw attachment.
- **I**: team interpretation.
- **S**: deliberate research simplification.
- **—**: not in the raw attachments.

| Item | Raw location | Class | Baseline (Figs 1–4, chains) | Strict k16–19 | Charging comparisons | Direction of effect |
|---|---|---|---|---|---|---|
| Battery | Notes image 1 (nominal 257 kWh; group efficiencies); PDF p.1 (usable ≈236.4 kWh for 18E1, ≈239 kWh for 18E2) | R | 240 (`experiment_settings.json:7`) | 239.01 (18E2) | matched k5: 236.44 (18E1); matched_k3 and fixed-duty: 240 | Slightly optimistic for 18E1 |
| Minimum SOC 15%, at all times | PDF pp.3–4; recorded minimum ≈15.1% | R | 0 | 35.8515 kWh | matched k5: 15%; fixed-duty: 36 kWh; matched_k3 and fee campaign: 0 | About 36 kWh of free energy per bus in the baseline, so optimistic on fleet and cost |
| End-of-day energy | Notes image 3: a recharge activity with 15 / 65 / 100% end-SOC bounds. No return-to-depot energy rule found. | 15% R; 65% is a recharge target (I); any return rule — | none (`terminal_floor: null`) | `reserve_only` | matched k5: per-duty floor from replayed GIRO terminal energy; matched_k3 and fixed-duty: aggregate floor | The baseline rewards ending the day empty. Floors in the comparisons are fairness devices, not GIRO rules. |
| PARX depot power | Notes image 2; PDF p.2 (60 kW, unlimited PARX chargers) | R | 240 kW | 60 | matched k5: 60; matched_k3 and fixed-duty: **350 kW (no raw basis)**; fee campaign: 240 | Strongly optimistic in baseline, matched_k3 and fixed-duty. It inflates any tariff-shifting saving in the 350 kW campaigns. |
| Opportunity power | Notes image 2: SOC-dependent taper, ≈371 kW falling to 120 kW near full | R | constant 240 | constant 240 | matched k5: taper; others constant | Mixed sign (pessimistic at low SOC, optimistic near full) |
| Charger counts, platform rules | Notes image 4, PDF pp.1–2: one or two chargers per site. PDF pp.3–4: platform blocking at two sites, first-in-first-out at one. | R | not enforced | not enforced (`capacity_enforced: false`) | matched k5: counts enforced; capacity pilots k1–k3 only; platform/FIFO nowhere | Optimistic. Observed violations exist (strict k17: 3 buses on 1 charger at 7880C, 4 on 1 at JON_A). |
| Deadheads | `Par_DHD.xlsm`: time bands (peak, morning, night, base) and ≈3,713 usable OD rows, some directional | R | `par_ref_dhd.csv`, symmetric, time-invariant, 49 pairs | same | same | Model differs from raw by ≤4 min on its pairs. Mostly optimistic in peak windows. Opus-F3 is verified. |
| 57-min direct-gap cap | **No raw basis.** The subagent reports the largest recorded non-meal direct gap is 57 min, plus one 70-min meal-break gap at 2190 (my own recount was not done). | S (graph pruning) | 57 | 57 (inherited) | 57 | Restrictive, so pessimistic on fleet in principle. It admits every recorded non-meal direct connection and removes at most the one 70-minute meal-break link, which must then route through a positive-charge station bridge. It cannot explain the 12/40 replay failures, which were measured at 350 kW under original charging (16 Sep execution README:21). |
| Minimum layover | Notes: route-specific minimum (a fixed number of minutes and a fraction of trip time) | R | none | none | none | Optimistic |
| Vehicle groups | PDF p.1: one line is served only by 18E1, local lines by 18E2; blocks are not mixed; per-group vehicle limits pp.3–4 | R | one homogeneous fleet; chain inputs mix groups | 18E2 only | mixed | Optimistic on fleet. **Links to the 17 Sep result:** the model's only fleet gain over GIRO in 9/102 cases came from mixing groups. The k33–36 floors below k (§ findings, F7) may therefore reflect a relaxation the operator forbids. |
| Energy use | 2.0 kWh/km (Notes image 1, PDF p.1); small idle draw | R | trip energy from GIRO usage; idle 0 | same | pilot only | Immaterial (about 2 kWh per day) |
| Tariff, demand charges | Not in attachments | — | flat | flat | 08:00 or noon peak scenarios; flat in fee campaigns | All tariffs are scenarios. No demand charge anywhere. |

**Verdict on Opus-F3: verified, with two qualifications.**

1. **Stricter campaigns exist, and the first review did not read them.** Strict k16–19 implements battery, reserve and 60 kW PARX, but not counts, taper, interval deadheads or group limits beyond 18E2-only. The matched k5 comparison (`../week_20260921/cleanup_physics/README.md:11-13`) implements nearly all charging physics and one charger per site, but it is a fixed-sequence charging reoptimization of five duties, not CG.
2. **Opus-F3 omitted two raw requirements that matter more than the ones it listed:** separated vehicle groups and minimum layovers.

The summaries (`GIRO_EMAIL_ATTACHMENT_AUDIT.md`) agree with the raw numbers checked, with two soft points:
- the "0.00;45" cell is read as a 45-second setup, which is an interpretation;
- "3,723 OD rows" includes 10 blank rows.

**Consequence for claims.** Only the matched k5 comparison approaches GIRO-comparable charging physics, and it is five buses with fixed trip sequences. No fleet result anywhere is GIRO-comparable. Group mixing and the missing charger counts both bias toward fewer buses, while the 57-minute cap biases toward more. Any statement comparing our fleet to GIRO's should wait for a run with separated groups.

## 4. Published claims: Doc, Slides, GitHub

Access, all read-only:
- Doc: Drive connector, 2026-09-23 ~00:50 UTC, full text (166,824 characters, all tabs).
- Slides: Drive connector, same time, text of all slides.
- GitHub: `gh`, 00:50 UTC.

| Location | Current wording | Assessment | Proposed correction |
|---|---|---|---|
| Doc lead ("EVSP DR research journal") | "their one-hour MIPs match 6 versus 24 GIRO targets. New experiments test whether integer-directed pricing closes that gap without known sequential routes." | Accurate but under-informative. It omits duty-informed grouping (present in the Figure 1 caption) and does not say the targets are proved event-model fleet optima. | "…match 6 versus 24 fleet targets. Each target equals a proved lower bound (trip concurrency at k≤10, certified LP floor at k15), so sequential attains the event-model fleet optimum in all 24 cases. Sequential chains add whole GIRO duties." |
| Doc "Integer-directed pricing" tab; Slide "how the search works" | Doc: "Starting from the fresh pool with cap K = 8". Slide: "impose the target fleet cap K = 8" | Slide discloses the provenance; Doc does not. Neither says 8 is also a proved floor. | "Cap K = 8, the GIRO duty count, which here equals the proved trip-concurrency lower bound. A target-free floor rule would have chosen the same cap." |
| Doc and Slides, 7/8 result | "Every hit proves eight within its augmented pool"; "Global optimality is unproved" | **Understated for fleet.** 8 = concurrency LB, so fleet optimality holds in the event covering model. Charging optimality and physical realism remain unproved. | "…eight buses, which equals the trip-concurrency lower bound, so the fleet is optimal for this event covering model. Charging cost, duplicate removal and shared capacity remain unproved." |
| Doc k15 paragraph | "C3 … found 15 buses and proved 15 optimal within its augmented pool" | Understated in the same way, subject to the Lagrangian floor's conditions. | Add: "15 also equals the certified-LP fleet floor (Q ≤ 964), so this is the event-model covering fleet optimum, in the capped graph." |
| Doc and Slide k33 table footnote | "fractional weights are not certified LP lower bounds" | Correct as written. It could add the conditional floors once the wrapper exists. Do not do so before then. | No change until the E0 wrapper output is audited. |
| Doc link "Independent Opus review, checked" | Points to ASSESSMENT, 57-minute cap, "effect … unmeasured" | Accurate. | Add a link to this follow-up when final. |
| `RESULTS_PREVIEW.md:15` Figure 1 caption | Discloses "complete GIRO duties … reference-informed grouping"; "Zero-length segments indicate a proved pool optimum, not full-model integer optimality" | Grouping concern **already corrected**. The second clause is now over-cautious for the fleet in the restricted event model. | "…a proved pool optimum. Where the fleet equals the case's proved lower bound, it is the event-model covering fleet optimum; charging cost is not certified." |
| GitHub | No PR for any September branch. Relevant heads: `codex/integer-columns-20260921` = `c50e5f20` (one commit after the k8 pin `1be819f0`); `codex/strict-graph-reuse-20260922` = `fedf4214`; pinned commits `a0e0bb7` and `e091a4d` resolve on GitHub. Open PR #34 (21 Aug): exhaustive oracle plus Ryan–Foster branch-and-price, not merged. | The review's B&P recommendation has an existing, unreviewed in-repo starting point. | Review PR #34's B&P before writing new B&P code. |

## 5. Narrow literature check

Performed by a subagent, 23 September 2026, about 14 searches and fetches. All passages below were read in text downloaded and extracted during the check. Where a text could not be obtained, the paper is marked **not read** and nothing is quoted from it.

| Paper | Read? | Mechanism established (location) | Relation to our method |
|---|---|---|---|
| Sadykov, Vanderbeck, Pessoa, Tahiri, Uchoa (2019), "Primal heuristics for branch and price: the assets of diving methods", *INFORMS J. Computing* 31(2):251–267, doi:10.1287/ijoc.2018.0822 | **Paper not read** (HAL download blocked). Author's ColGen 2016 slides on the same work were read: math.u-bordeaux.fr/~rsadykov/slides/Sadykov_ColGen16slides.pdf | Slide 8: diving, "further column generation after rounding", generates "'missing' complementary columns". Slide 10: limited backtracking (limited discrepancy search, LDS) for diversification. Slide 13: "Diving with sub-MIPing" runs the restricted master heuristic "with all columns generated during diving". Slide 29: diving is significantly better than the restricted master heuristic. | Our dive → fix → reprice → limited alternatives → final pool MIP over all dive columns is **this established scheme**. Obtain the IJOC text (Cornell library) before citing passages from the paper itself. |
| Lübbecke & Desrosiers (2005), "Selected topics in column generation", *Oper. Res.* 53(6):1007–1023; preprint optimization-online 2002/12/580 read | Yes (preprint) | §3.4, p.11: the RMP "may be integer infeasible"; "without branching we may miss (probably all) optimal integer solutions". §7.3, p.24: early integer solutions via rounding and temporary fixing. | Our 9-vs-8 finite-pool gap is an instance of this known limitation of price-and-branch. |
| Desrosiers & Lübbecke (2005), "A primer in column generation", in *Column Generation*, Springer | Yes | §1.3, p.5: the generated columns "may not contain an integer feasible solution". | Same point, textbook source. |
| de Vos, van Lieshout, Dollevoet (2024), "Electric vehicle scheduling in public transit with capacitated charging stations", *Transp. Sci.* 58(2):279–294; arXiv:2207.13734 read | Yes (arXiv) | §4.2.4, p.17: price-and-branch described as a restricted master heuristic over all generated columns; truncated CG fixes path variables to one and restarts CG. p.24: price-and-branch gives a poor bound (~20%) on one instance. | **Closest EVSP prior art.** It already compares price-and-branch with a dive-with-pricing on electric-bus scheduling, with capacitated chargers. The check found no hard fleet-cap row, no backtracking and no final MIP over dive columns in the parts read. |
| Parmentier, Martinelli, Vidal (2023), "Electric vehicle fleets: scalable route and recharge scheduling through column generation", *Transp. Sci.* 57(3); arXiv:2104.03823 read | Yes (arXiv) | §4.4, p.22: at each node CG completes and the largest route variable is fixed to 1; "strong diving" evaluates several candidates. Fleet enters as a fixed vehicle cost in the objective. | Diving with pricing, including multi-candidate evaluation, is established for EV route and recharge scheduling. |
| Joncour et al. (2010), *ENDM* 36; Barnhart et al. (1998), *Oper. Res.* 46(3); van Kooten Niekerk et al. (2017), *Public Transport* 9; Perumal, Lusby, Larsen (2022), EV bus scheduling review (the check found *EJOR* 301(2), not TR-C) | **Not read** (paywall, login or not fetched) | — | Do not cite passages until read. |

**Established; do not claim as ours:**
- A converged LP pool can lack good or feasible integer solutions.
- Diving with repricing and column fixing, with limited backtracking.
- A final restricted-master MIP over dive-generated columns.
- Diving for electric vehicle and bus scheduling specifically (de Vos et al.; Parmentier et al.).
- "Diving beats price-and-branch", as a general claim.

The Opus review's statement that the pool-gap finding is "already known" is therefore **supported by primary sources**. So is ASSESSMENT's caution that novelty needs a literature audit. The audit now points against novelty for the mechanism.

**What our evidence might support, narrowly and as hypotheses to position rather than claims:**
1. **A dive targeted at a proved lower bound.** A hard fleet row at a proved floor, whose dual enters pricing, used as a test of whether that floor can be attained, together with an escalation rule. None of the texts read used a hard fleet cap in the dive. However, the IJOC paper itself was not read, and a local-branching-style master row appears in the author's slides. This is at most a **variant**, and novelty is unconfirmed until the IJOC and de Vos full texts are checked for fleet or cardinality rows.
2. **A quantified failure mode inside an event/SOC-expanded pricing graph.** Pools that are LP-certified and fleet-proved in-pool miss a *proved* fleet optimum in 18/24 cases (§2). The complementary routes have reduced cost up to ~53, and the mechanism decomposes exactly as reduced cost plus the dual value of extra coverage (Doc, "Why the fresh integer pools are worse"). This is an empirical contribution in an application domain, not a methodological one.
3. **Evaluation discipline.** Matched end-to-end budgets and separated certificate types (§2 table). This could strengthen a paper but is not a contribution on its own.

The paper framing that the evidence best supports is an EV-scheduling application with careful certification and a bound-targeted diving variant, positioned explicitly against Sadykov et al. (2019) and de Vos et al. (2024).

## 6. Ranked next experiments (none run or authorised by this audit)

- **E0 (zero compute, prerequisite).** Build the bound wrapper from Opus-F7 and `audit_bounds.md`: priced-iteration record only, refuse missing min rc or artificials, Q computed from inputs. Apply it to all k16–36 baseline endpoints and the strict k16/k17 records. Strict needs its own Q, using the maximum station power and reserve-adjusted energy. Output floors with scope. This is bookkeeping, not an experiment, so it does not count toward the three.
- **E1. Cap-escalation control for Opus-F4.** On the four k8 fresh pools, one seed, 3,600 s: arm (a) cap = floor + 1 = 9; arm (b) cap = floor − 1 = 7 with escalation. Then one k15 run on C1, where concurrency (14) < floor (15), starting from cap = 14 with escalation. Arm (a) runs on the current diver unchanged and is the smallest decisive control on its own: four allocations. Arms (b) and the k15 C1 run need an escalation loop that does **not exist**, since `search_exhausted` currently stops the dive (`diving_pricing_pilot.py` restart loop). The loop must be implemented and unit-tested before those arms. All arms reuse existing graphs and pools.
- **E2. Grouping-free chain control for Opus-F1.** Rebuild k8 and k15 sequential chains for 2–3 chains where each step adds a trip block of the same size chosen by time window or random trips, not a GIRO duty. Run sequential CG plus the one-hour MIP against the existing fresh arms. Report against the proved floors, since concurrency and Lagrangian floors are computable for any trip set. Graph builds at k≤15 cost ~2.6 h each, so this is affordable, whereas k≥33 is not.
- **E3. Long-idle single factor.** Chosen over an event-pricer brute-force oracle because it tests a verified modelling restriction and the only fleet claims it can falsify. Use C1 or C3 k15, not k8: at k8 the concurrency floor makes the fleet graph-invariant, so a k8 test can only move charging cost. Change only `max_trip2trip_min` from 57 to the horizon and add zero-charge station waits. Recompute the certified LP and Lagrangian floor, then run the pool MIP. A floor or incumbent of 14 would falsify the capped-graph "optimum" of 15 for that case. The event-pricer oracle, extending PR #34, is the next candidate after these.

## 7. Indispensable unavailable evidence

- **Paired priced-iteration dual vectors** for the k33–36 floors other than w1_k35 and w3_k35. They exist in the remote `cg.json` files and were not fetched, following the brief's no-broad-download rule.
- **Execution-time source state.** The present-day `git status` plus the launch guards are strong but indirect evidence. No execution-time `git diff` was archived.
- **PR #34 artifacts** (`summary.json`, reproducer JSONs). I read only its REPORT.md, and its results were not reproduced.
