# What to share at the next meeting

Evidence checked against the **16 September, 23:44 EDT** cluster snapshot. Detailed seeds, costs, source paths and qualifications are in [DETAILS.md](DETAILS.md).

**Suggested opening:** “We have moved well beyond the early k=5 results. Sequential CG now produces GIRO-sized fleets at target 32 in all six chains, when longer integer searches and seed repeats are included. Fresh starts remain substantially weaker. On one small instance, changing duties also improves charging cost beyond charging optimization alone. We now have clearer evidence about which gaps are computational and which depend on the model.”

| Research question | What we can say now |
|---|---|
| How far does sequential CG reach? | Original one-hour MIPs reached largest targets **26/28/31/29/26/28** across chains 1–6. With selected longer MIPs and repeats, **all six reach 32**. These are largest matches, not success at every smaller k. |
| What happens without sequential inheritance? | At targets 5/8/10/15, fresh matches were **5/6, 1/6, 0/6, 0/6** versus sequential 6/6 at every size. Fresh CG received the accumulated earlier CG time. The largest fresh match in this comparison is 8, on one chain. |
| Was fresh k15 simply short of MIP time? | The new **18 three-hour fleet searches still give 0/18 matches**, using 16–19 buses. Every pool's bound remains near 15: missing target routes versus unfinished integer search is still unresolved. |
| Does joint optimization beat fixed GIRO duties with optimized charging? | **Yes, modestly on one five-duty input:** fresh-CG-derived schedules cost 2.81%,2.21%,7.34% less at peaks 08/12/18. The earlier ties came from saved-pool repricing, a different experiment. |
| Does that charging gain survive stricter rules? | Preliminary morning result with 15% reserve and 3-minute charges: **157.965 fixed vs 154.921 fresh, 1.93% lower**. Exact-once trips validated; fresh returns with more total energy. Noon comparison is missing after a startup timeout; evening still needs a duplicate-trip dispatch check. |
| Did decomposition recover the larger target? | **Not yet.** On one 32-duty parent, nine 4×8 partitions gave 34–37 buses. Combining all nine pools still gave 34. Retaining LP-support routes improved the pool bound from 33 to 32, but no 32-bus schedule was found. |

The sequential fleet results use 240 kWh / 240 kW, covering and inherited columns, without reserve, shared capacity or terminal-energy constraints. New seed results have individual-route replay; their separate dispatch-conversion audit is pending. The charging comparison uses 240 kWh / 350 kW and zero start fee. Neither is a full-GIRO operating comparison. The relaxed charging table compares five buses and equal achieved ending energy; savings are against the computed fixed-duty comparator, not a proof of global charging optimality.

**The strongest new explanation:** with vehicle groups separated, exact time-only lower bounds already require GIRO's fleet in all 102 cases. Continuous baseline charging witnesses attain that count. Mixing is necessary for the nine fractional k−1 cases; an integer electric-bus saving remains unproved.

**What to ask coauthors:** prioritize integer-aware column generation and realistic charging-cost evidence, or invest first in the larger pricing-graph redesign? Decomposition alone has not solved the integer difficulty. The code review proposes useful experiments, but its 10×/25-minute projections and “zero risk” stopping claim are not established. [Mathematical qualifications](CODE_REVIEW_RESPONSE.md).

The current Google Doc has been updated in place. Figures/history and Slides were preserved; frozen 128/102/67/35 audit counts remain explicitly separate from the new seed results. No new solver jobs were submitted.

[Budget definitions, why sequential helps, and what happened in the constraint pilots](TABLE_AND_CONSTRAINTS.md).
