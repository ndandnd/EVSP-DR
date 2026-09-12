# Full-pool inheritance: first integer results

Evidence captured 12 September 2026 at 18:00 EDT. These are new indexed full-pool runs, separate from the earlier 512-route / 15-minute importer.

| Case | Target | Earlier bounded warm start | New full-pool buses | CG minutes | Fleet proved within pool |
|---|---:|---:|---:|---:|---|
| w3_k11 | 11 | 12 | 11 | 15.6 | Yes |
| w4_k10 | 10 | 11 | 10 | 18.8 | Yes |
| w6_k11 | 11 | 11 | 11 | 19.3 | Yes |

All three CG runs have pricing certificates in their represented event graph. Selected routes passed individual physical replay and at-least-once trip coverage. The final MIP TIME_LIMIT status is from charging optimization; the first fleet-minimization stages proved their fleets in 31.3, 3.8 and 7.4 seconds, respectively. No full-model integer or shared-capacity proof is asserted.

The importer accepted 35,495, 40,954 and 33,879 inherited routes, respectively, in 2.4, 3.3 and 2.5 minutes. This shows that the indexed implementation can retain full pools without spending hours initializing these cases. The first two fleets improve by one bus over the bounded treatment; the third already matched its target. This is evidence that richer inherited pools help, not an isolated causal estimate for a single code change. All three comparisons use the same saved parent status hashes. Their certified LP objectives differ by less than 1e-8, while two integer fleets improve. For C4 k10, the old pool was proved to require 11 buses and the new pool permits 10: this establishes a pool limitation despite the same certified LP objective. C3 k11 previously found 12 with bound 11, so search difficulty in that old MIP remains another possible explanation. Code changes and the changed import size still preclude attributing all timing differences to one implementation change. CG times include inheritance and use existing graph caches; historical graph construction is excluded.

Physics: covering; 240 kWh / 240 kW; 2.5 kWh / 5-minute event graph; flat tariff; no shared station capacities or terminal SOC floor. CG objective 100,000 per bus plus electricity and 5 per charge start. CG source e091a4dba549510238507ef5e5367abea958bd30; MIP source 871d057e1067411f09581e37d78f7c1ca43f68bb. MIP budget 3600s, stage1 ≤ 1800s; stage2 fleet ≤ validated incumbent. Full case/input/output hashes are in comparison.json.

Sixteen of the 37 new full-pool CG cases are certified at this snapshot. At 18:02, 23 EVSP–DR jobs were running with no invalid dependencies. Both shared graph preparations remain running; no completed 32-duty parent result yet.

All nine recovered component MIPs have finished. Five found 8 buses and proved 8 optimal in their pools. Four found 9 buses: d04_g1 and d06_g3 retain a pool bound of 8; d06_g1 and d07_g1 proved 9 within their pools. All selected-route replays passed. These are 8-duty subsets; nine does not match the GIRO target. No new execution failure or confirmed preemption was observed.
