# Research results — 15 September, 08:23 EDT

**Chain 2 now matches its 28-bus target in the original one-hour MIP.** Its 199,248-column pool proves 28 minimal after 27.21 fleet-search minutes; the total run takes 60.19 minutes. Charging optimality remains open. Individual-route replay passes; duplicate-trip removal and shared charger capacity are not validated by this result.

| Chain | Largest target matched in original one-hour MIP | Including separate longer searches |
|---|---:|---:|
| 1 | 26 | 26 |
| 2 | 28 | 28 |
| 3 | 27 | 28 |
| 4 | 27 | 27 |
| 5 | 26 | 26 |
| 6 | 28 | 28 |

These are largest individual matches, not a claim that every smaller case matched in the original budget. The original k16–25 batch stays at 35/60 target matches; separate searches recovered all 25 misses from unchanged pools. [Every original extension case: bus count, CG time and stopping reason](CHAIN_TABLES.md).

## New results and the next question

| Case | What completed | Result | What remains open |
|---|---|---|---|
| C1, target 27 | Original one-hour MIP | 28 buses; pool bound 27; replay passes | Can these columns form 27 buses? |
| C2, target 28 | Original one-hour MIP | 28 buses; pool bound 28; fleet proved in pool | Charging minimum and full-model fleet optimality |
| C5, target 28 | CG, 239.78 minutes | Route weights sum to 27.0000; weighted objective 2,701,242.913516 | CG stopped at four hours; last reduced cost −0.271546. Integer search is running. |

C5's fractional solution covers 674 trips. Its value is not a certified full-model lower bound: pricing did not prove that no improving route remains. The model is still set covering with inherited columns, 240 kWh/240 kW, route cost 100,000 + electricity + 5 per charging start, without reserve, shared charging limits or an ending-SOC floor.

**Two useful follow-ups launched at 08:18 EDT:** unchanged-pool MIPs for C1 k27 and C5 k27. The latter originally found 29 buses with bound 26. Both bounds leave the target possible; these searches test whether the existing columns suffice. Each allows three hours for fleet search and 3.5 hours total. Source journals, ordered pools, inputs, objective, physics and frozen native solver are checked; native greedy policy is unchanged. They are separate searches, not resumed trees or a controlled measurement of time alone, because hardware may differ. [Experiment design, hashes and launch receipts](../../continuation_gaps2_20260915/README.md).

Jobs 227897 and 227898 are running independently on the default partition, each with 8 CPUs and 24 GB, with scaglione-compute-01 excluded. Both full-size Gurobi license checks passed and both attempt paths are registered. The main scientific collector started before this campaign was added: the separately timestamped launch collection verifies its two cases; the next full collection will incorporate the campaign. No new CG certificate is implied.

## Other comparisons

The twelve core/expanded-pool union and control MIPs are still running. No production endpoint is verified yet. The completed larger compact-start batch remains 8 matches, 8 pools that rule out target, and 8 unresolved misses. Stricter-model results are unchanged. [Pool-union status](compact_union_results.csv) · [Larger compact-start results](compact_large_results.csv).

C3's earlier separate k28 search remains a 28-bus pool proof. Its repeat used a different CPU; [the exact two-log comparison](../status_20260915T111319Z/SEARCH_WORK.md) shows why extra allocated time alone is not an established explanation.

Queue in the 08:13–08:23 collection: 29 running and 26 true input dependencies, excluding 33 held historical tasks. No new execution failure, impossible dependency or confirmed preemption was observed. The twelve k29–30 graph jobs continue; their CGs retain their real graph and previous-k dependencies. Held historical and EVSPV2G work are untouched. C4 k28's CG completed during the scan and released its MIP, but its scientific endpoint awaits the next source collection.

The current Google Doc updates its existing tables and explanations in place. Both figure tabs are preserved; Slides are untouched. Hourly monitoring and consolidation around 09:00 EDT remain scheduled.

Snapshot `20260915T121336Z` ran 12:13:36–12:23:17 UTC, 581.1 seconds; SHA256 `12f9d7a6b6cd974cafaa57c9f0ba84f4e7075d324e378fe07e80e48e6b30facc`. Register/workbook: 3,282 records across 74 source groups, with the exact six supplements preserved. Checks cover 321 core, 184 evening, 83 pool and 151 original-chain endpoints; 76 CG and 75 MIP results among 90 submitted original extension cases. The preemption study has 958 attempt records. [New endpoints and hashes](new_endpoints.csv) · [Document verification](doc_verification.json).
