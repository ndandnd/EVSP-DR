# Partille research decisions, 3 November

## Registered experiments

| Scientific question | Saved inputs and comparator | Cluster record | What counts as an answer |
|---|---|---|---|
| Can a new set of trip connections repair a duty that cannot finish? | Revisit the three k8 failures C1, C4 and C5 on their same frozen root columns. The original single attempt missed all three. | 918045 tasks 0–2 in partille_rematch_20261103 | Stop after 32 candidate covers or 180 seconds per chain. Exclude only an exact whole duty proved unable to finish alone. A joint retiming miss may exclude the visited matching for enumeration, with no proof of route infeasibility. Require independent exact-once replay for each success. This selected sample is exploratory. |
| Does a 5 kWh grid make the problem smaller without sacrificing dispatch quality? | C2/C3/C6 k8. Historical 2.5 kWh controls authenticated and replayed. PARX 60 kW, opportunities 240 kW, minute-level plugs, otherwise identical simple-dive solver. | 917989 tasks 0–2 in partille_soc_grid_20261103 | Compare cold graph time, graph size, CG time, weighted pricing certificate, peak memory, MIP and exact-once replay. Compare charging cost only at matched fleet and compatible service, and distinguish completed from censored attempts. Host hardware can differ. |
| Can an alternative set of station visits serve the same GIRO duties under the 60 kW depot rate? | All 12 October cells with a fixed-visit infeasibility proof. Try the registered flat-price donor in the same family with identical duties, inputs, power, plugs and ending-energy requirement. | 917978 in partille_visit_transfer_20261103 | Replay all trips, station windows, SOC, plugs and end energy and reprice the actual donor energy at the receiver prices. A pass gives a feasible upper bound. A miss on this donor leaves alternative station choices open. |
| Does fresh integer-directed pricing beat a simple dive under equal resource limits? | C2 preregistered pilot and control, then conditional expansion to remaining chains, conditional k10, and sparse-prefix controls. | 905307, 905843 and original November 2 ledgers | Only terminal, hash-bound results may open the gates. k10 requires repair hits on at least 4/6, strictly more than the controls, and retention of every control hit. |

The separate three-of-five threshold for expanding the original trip-link repair to k10 was missed. The two k10 rules refer to different treatments. Do not launch additional k10 or k15 work or replace any archived control in these comparisons. The four-hour check covered the job families listed here through their terminal results on November 3.

## Verified observations at registration

- C2 eight-bus covering with duplicate service allowed was still unresolved after 7,200.01 s and 318,737 branching nodes. The original pool separately has a proof that eight exactly-once duties are impossible. [Native aggregate](https://github.com/ndandnd/EVSP-DR/blob/700f2eddb35eb76a0596b24ba11fac4a5295857b/research/partille_20261102/diagnostic/original_pool/cover_7200s.json).
- Rereading the original 350 and 60 kW comparisons, 38 of the same 72 distinct scenarios favored the alternative assignments at each rate. Median saving changed from 1.468 to 1.194 modeled units. The 12 failed slow-depot tests fixed the charging visits chosen under faster charging. [Paired audit](https://github.com/ndandnd/EVSP-DR/blob/f670dc6711bc92ffc94400471012f352b01cab13/research/partille_20261102/depot_power_audit/README.md).
- The 2.5 kWh grid controls returned 9, 8 and 8 buses on C2, C3 and C6 while serving trips 44, 5 and 3 extra times. These covering results differ from the independently verified C3 and C6 exactly-once November repairs on saved roots. [Grid registration and validation](https://github.com/ndandnd/EVSP-DR/blob/66be95aa720056c4980ef2d2dc5a492a3dc15625/research/partille_20261102/soc_grid_sensitivity/README.md).

**Visit transfer endpoint.** All 12 cells passed replay on the same passenger duties using alternative charging visits. Five distinct passenger inputs were represented. Charging cost minimization remains open. [Results and tolerances](fixed_duty_visit_transfer/RESULTS.md).

**Rematching endpoint.** All three known misses produced verified eight-bus schedules. [Case counts, timings and why the earlier pools missed](bounded_rematching/RESULTS.md).

## Next decision

The rematching result now warrants a prospective test on new eight-bus cohorts, with method and budget frozen in advance. If the coarser grid saves time, compare fleet, duplicated trips and replay first, then isolate the energy-grid effect on cost with matched-fleet, identical-service schedules. Run a station-choice optimization for missing witnesses or to measure the best cost at 60 kW. Compare candidate station choices with the same opportunity windows and terminal requirements. These experiments use distinct historical model settings and cannot by themselves validate every GIRO physical assumption.

## Campaign handback · 3 November

All 2–3 November jobs assigned to this monitor are terminal. No scoped job remains in Slurm. The separate, independently registered extended SOC-grid campaign is outside this handback.

- The two fresh C2 methods each selected nine buses. Repair pricing produced 21 extra services and simple diving produced 11. Both release checks recorded a scientific hold. All 34 conditional tasks ended without starting. Their sparse prefixes consumed zero compute. [Fresh pilot comparison](pilot_endpoint/RESULTS.md).
- The three 5 kWh runs used about one quarter of the pricing arcs and 43.6–52.5% less root CG time than the 2.5 kWh historical controls. All three selected eight buses. C6 served all 128 trips exactly once; C2 and C3 repeated 19 and 7 services, respectively. [Full grid panel](soc_grid_sensitivity/results/RESULTS.md).
- The earlier three exploratory rematch winners and 12 verified alternative-visit price cases retain their original scopes. [Rematching](bounded_rematching/RESULTS.md) · [Station-visit replay](fixed_duty_visit_transfer/RESULTS.md).

The next fleet test would freeze the rematching method on previously unused k8 inputs and use exactly-once replay as the primary outcome. A cost comparison would optimize station visits and charging times on both assignments at the same fleet and level of passenger service. Neither test was submitted by this monitor.
