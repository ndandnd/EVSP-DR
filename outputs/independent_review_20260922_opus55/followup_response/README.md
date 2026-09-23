# Response to the follow-up audit — 23 September 2026 UTC

**The follow-up is useful, but its recommended long-idle experiment rests on a false distinction.** All 24 benchmark cases have an exact time-only fleet lower bound equal to their target, including C1/C3 k15. This strengthens the fleet claims and changes experiment priorities. The original FOLLOWUP_AUDIT.md is preserved unchanged.

## Verified result: minimum baseline fleet in all 24 sequential cases

| Target | Sequential reaches minimum fleet | Fresh reaches minimum fleet | Exact time-and-travel lower bound |
|---|---:|---:|---|
| 5 | 6/6 | 5/6 | 5 in every chain |
| 8 | 6/6 | 1/6 | 8 in every chain |
| 10 | 6/6 | 0/6 | 10 in every chain |
| 15 | 6/6 | 0/6 | 15 in every chain |

The new local check used the exact trip CSVs, matched by SHA-256 to the figure and saved MIP evidence. It used the previously audited rational shortest-travel closure and maximum matching routine, then separately checked each antichain against **all reachable paths**, not merely direct edges. All 35 certificates pass. No Gurobi solve, SSH call or cluster job was needed; elapsed local calculation was 2.520 seconds.

The existing sequential incumbents are supported by their saved native individual-route replay and at-least-once coverage checks; those physical simulations were not repeated here. This is the homogeneous baseline covering model, using its own deadhead table, battery/power assumptions and omitted capacity constraints. It is not optimal charging cost, exactly-once dispatch validation, or compliance with all GIRO operating rules.

**Proof.** Let S be the certified set of mutually incompatible trips. No time-feasible route can contain two members of S, even using intermediate trips. Summing their cover constraints gives sum_r lambda_r >= |S|. A validated |S|-route cover attains that bound. The certificate therefore applies to fractional fleet weight as well as integer fleet size; it does not establish optimality of the weighted bus-plus-charging objective.

C1 and C3 k15 have only 14 simultaneously active passenger trips, but each has a 15-trip incompatibility certificate after travel time is included. Thus their 15-bus lower bound does **not** depend on the battery grid, positive-charge bridges, the 57-minute cap, or a Lagrangian Q envelope. Their baseline fleet optima survive relaxing those connection/charging restrictions while preserving the same trip times and travel data.

The same check gives exact time-only lower bounds for all 11 larger endpoints considered in the follow-up; these equal its proposed numerical floors. They remain lower bounds without any matching integer incumbent. The interrupted/unfinished CG weighted objectives remain uncertified. See time_bounds.json for case values, input hashes, source hashes and scope; time_certificates.json stores explicit matching, vertex-cover and antichain witnesses. time_bound_check.py reproduces the calculation from the stated local source files.

## Disposition of material follow-up claims

| Finding | Disposition | Reason / action |
|---|---|---|
| Opus-F2/F4: 24 sequential hits and the successful dives attain a model fleet optimum | Verified, strengthened for the frozen benchmark panel | All 24 targets have exact time-only bounds; no Q calculation is needed for the two exceptional concurrency cases. k8 dive hits and C3 k15 inherit the same input-specific floors. The historical caps were still supplied from GIRO targets; do not rewrite that provenance. |
| Only 22/24 fleet optima survive lifting the 57-minute cap | Refuted | The additional travel-time certificates protect C1/C3 k15 too. All 24 do, under unchanged trip times/deadhead data. |
| E3 could find a 14-bus solution in C1/C3 k15 after removing long-wait restrictions | Refuted under the stated single-physics comparison | Time/travel alone already requires 15. Long-wait tests can study charging cost, route-pool usefulness and runtime here, not fleet below 15. |
| A lower bound of 14 would falsify the 15-bus optimum | Refuted as logic | A weaker lower bound is not a 14-bus feasible schedule. An expanded model's 14-bus feasible incumbent would demonstrate a restriction's effect, not invalidate a valid certificate for the original restricted model. |
| Opus-F7: conditional numerical floors on 11 larger endpoints | Superseded here by exact combinatorial bounds | The time-only certificates give the same floors without pairing LP/pricing records or choosing Q. A robust reduced-cost wrapper remains useful when time-only bounds are weaker; retain its missing-data/pairing guards. |
| Opus-F1: larger sequential target misses show the relative advantage disappeared | Still unsupported | No matched fresh comparator at those scales. |
| E1: returning 9 under cap 9 proves cap 8 was doing the work | Too strong | Current diving stops at the first integer solution at or below its cap (diving_pricing_pilot.py:753–763). Raising the cap changes both duals and the stopping rule. A miss at one seed/time budget cannot identify a unique cause. |
| E1: cap 7 on an instance with proved floor 8 is realistic default behaviour | Reclassify as a stress test | It deliberately contradicts the known lower bound. Test fallback machinery locally first. An honest weak-bound or unattainable-bound case is a separate experiment. Per-cap budgets must permit escalation before the total budget expires. |
| E3 is a single-factor change | Needs separation | Raising the direct-gap cap and allowing zero-charge station bridges are two changes. Use separate arms or explicitly label their combination. |
| Opus-F3 raw operating differences | Useful, with scope limits | I independently confirmed Par_Notes.docx paragraph 7 requires route 21 layover of at least 4 minutes and 10% of trip duration. This turn did not repeat every raw-attachment audit. Matching reference fleet counts is a valid benchmark claim; matching all operating constraints is a different claim. Group-separated/18E2-only and strict experiments already exist. |
| Opus-F8 dirty source fully resolved | Explained, not an execution-time snapshot | Current tracked-clean trees plus launch guards support the explanation. They cannot replace an archived execution-time source tree. Retain the historical dirty flag with its explanation. |
| The diving mechanism is new | Unsupported | The de Vos EVSP paper explicitly compares price-and-branch with iterative fixing and renewed CG. The broader literature also contains diving. A hard fleet cap may be a useful variant; novelty is not established by not finding that row in a few papers. |

## Research decisions

1. **Use the exact fleet certificates now.** Upgrade the 24-case benchmark and successful-dive fleet wording, with baseline covering scope. Keep weighted CG convergence, charging optimality and operational feasibility separate. The larger 11-case floor table can use exact time-only bounds; no solver rerun is needed.
2. **Evaluate the proposed method against a published diving/truncated-CG baseline.** A cap 8/cap 9 test is useful only with clear stopping/budget accounting. Preserve the existing witnessed improvement, but do not make generic diving or the existence of an LP/integer-pool gap the paper's novelty. A grouping-free control should use the identical terminal trip set, block sizes, physics and end-to-end accounting.
3. **Make long-wait and stricter-physics experiments answer achievable questions.** On these 24 cases, measure charging cost or time to recover the already known fleet minimum. For a fleet-changing test, first screen for a gap between an exact time-only floor and the current best fleet. Capacity-enabled pricing and realistic charging comparisons remain substantive unanswered application questions.

No new cluster experiments were submitted by this response. Preserve the follow-up's proposed experiments as proposals, with the corrections above, before turning them into jobs.

## Primary sources and reproducibility

- [Original follow-up](../FOLLOWUP_AUDIT.md), unchanged.
- [Existing time-only method and scope](../../independent_review_20260916/time_only_vsp_20260916/README.md).
- [Earlier independent time-only audit](../../independent_review_20260916/advisor_time_only_audit_20260917/README.md).
- [de Vos et al., preprint §4.2.4, pp.17–18](https://arxiv.org/pdf/2207.13734): price-and-branch versus repeated fixing and CG; §4.2.3 also distinguishes conservative-network and broader-model bounds.
- [Parmentier et al., scalable EV route/recharge CG](https://arxiv.org/abs/2104.03823): diving is one of its stated core ingredients.

The certificate files and script are local reproducibility artifacts. The upper-bound evidence is the already audited, hashed snapshot used by Figure 1, not a new dispatch re-simulation. Raw correspondence is not republished.

## Publication

The current Google Doc lead/review paragraph and weekly Slides 2/28 now use these exact fleet bounds and retain the baseline covering scope. [Publication verification](publication/verification.json): 21/21 checks pass. All 42 slides, images, editable tables and other notes are preserved. The current Doc changes exactly two passages; its table rows, image references and old source links are unchanged. All seven Doc tabs remain visible; other tab bodies were not re-exported. Slides 2/28 and the Doc paragraph were visually inspected. Before/after Markdown snapshots and the verifier are saved; full PPTX exports and rendered PNGs remain local to avoid adding binary export copies to Git.
