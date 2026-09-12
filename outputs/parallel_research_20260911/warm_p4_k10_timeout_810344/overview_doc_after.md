EVSP–DR: current research status

Updated 11 September 2026; meeting held 10 September.

What improved—and why

The newer integer results are substantially better. Two changes matter: we moved from exact-one trip coverage (set partitioning) to at-least-one coverage (set covering), and we now retain useful columns from the previous size in some chains. The old partitioning chart and the new warm-start chart do not isolate the effect of warm starts.

There is direct evidence that inherited columns help under covering: chain 3 at k=8 required nine buses in the fresh saved pool, with nine proved optimal in that pool; the inherited-column pool supports eight. Chain 5 at k=5 similarly improves from a proved six-bus pool optimum to five buses. More MIP time cannot improve those proved fresh-pool fleet optima without adding or changing columns. This establishes a route-pool limitation in these cases, not a universal diagnosis for every difficult instance.

Four experiment groups—do not combine their charts

1\. Fresh covering chains: six nested chains, k=2–15, 84 completed cases. Each size starts independently. At k=5 the integer fleets are 5, 5, 5, 5, 6, 5 across chains 1–6. At k=10 they are 11 in all six chains. At k=15 they are 18, 17, 18, 19, 16, 19\. These are feasible incumbents; proof status must be read separately.

2\. Inherited-column covering chains: six chains, k=2–10. Each size imports and revalidates previous-k columns, adding single-trip routes for new trips. Chains 3, 5 and 6 now reach ten buses at target ten, with fleet optimum proved within their saved pools; chain 1 is blocked at k=7 after an initialization timeout. At 18:15 EDT, chain 2 k9 has also exhausted its CG budget: importing 42,732 columns took 488.75 minutes, leaving no final LP result or pricing certificate. Its pool-export job failed because there was no acceptable terminal LP record; the error mentions artificials, but absent LP fields do not establish positive artificial coverage. The k9 MIP is dependency-blocked. Imported columns were preserved and k10 is running from them. Update at 20:17 EDT: chain 4 k10 also timed out during initialization, before its first CG iteration; its column journal is empty. There is no new LP or MIP result. This repeats the chain-1 initialization failure. The predecessor k9 pool survives; no unchanged retry was submitted. Chain 2 k10 remains running. These are not warm-start results through k=15. The larger cases can spend most CG time importing and reoptimizing inherited routes, so better integer fleets do not imply faster end-to-end runs.

3\. Station-capacity and depot-speed pilot: separate small cases, k=1, 2 and 3, comparing baseline, station capacities, 60-kW PARX charging, and both changes. This is not a second six-chain chart through k=15. Return-SOC requirements were not added in this pilot. All pilot cells now have outcomes. The capacity-constrained k2 runs find three buses; k3 runs find sixteen after CG stops uncertified with only three newly generated columns. A single k3 pricing call took 7.15 hours versus 0.006 seconds for its LP: this is a pricing bottleneck, not proof that capacity requires sixteen buses. Baseline k2/3 schedules violate the proposed one-space limit.

4\. Matched return-energy charging experiment: one separate five-bus, 62-trip cohort, with tariff peaks at 08:00, 12:00 and 18:00. It imposes an aggregate final-energy floor of 280.7833 kWh, not a 65% return-SOC requirement on every bus. It uses 350-kW charging and does not enforce shared station capacity. Joint versus fixed-duty optimized charging-related costs are 261.57 versus 279.90, 332.03 versus 332.03, and 217.51 versus 232.25, respectively. Costs include electricity and the five-unit charging-start fee.

How to read the evidence

Integer buses \= number of selected bus routes. Fractional route weight \= sum of LP route weights, not the weighted LP objective divided by 100,000. CG time includes initialization and inherited-route import when reported as total run time. A pricing certificate establishes the represented LP result within the stated tolerance; a timeout does not. A MIP proof applies to the saved column pool. Individual route replay does not by itself validate duplicate-trip removal or shared charging capacity.

The old table entry “12 buses; pool fleet unproved; CG 172.3 min” means 12 integer buses, with fleet optimality not proved in that saved pool, after 172.3 minutes of CG. The minutes are not MIP time or an optimality gap. The historical tables retain their original observations and are explicitly archival.

Where to read next

Current results and research figures: research figures and the current comparison notes; historical launch records are retained as dated evidence.

Meeting archive — 10 Sep 2026: archival figure collection used for the 10 September meeting, not the current status table.

Reference — audit details and Answers to your notes: detailed methods and historical diagnoses.

Experiment register: indexed source artifacts and proof/validation fields. The reproducible local entry point is outputs/research\_register/README.md.

Next decisive comparisons

Finish inherited-column chains with matched fresh controls; report fleet, pricing stop, MIP proof and total time separately. Finish the station-capacity pilot before expanding it. Combine return-energy and shared-capacity constraints only in a clearly identified new cohort. Retain zero/near-zero reduced-cost or complementary routes as a targeted pool-enrichment experiment.