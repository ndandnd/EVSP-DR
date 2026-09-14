# Research results collected 14 September, 18:58–19:04 EDT

This is the result snapshot. The separate evening launch plan records subsequent submissions and live queue counts.

| Evidence | Current result | What it means |
|---|---|---|
| Equal-time fresh CG | 24/24 pricing certificates; 6/24 target fleets | LP convergence alone did not produce useful enough integer pools. The 24 warm-reference MIPs all match; upstream computation is charged in the comparison. |
| Small inherited seed sets | 36 CG endpoints, 35 certificates; 26 MIP endpoints | 8 targets, 13 proved pool limits above target, 5 open fleet gaps. |
| Completed seed-method pairs | Integer seeds smaller in 5; tied in 7 | 12 paired endpoints; 6 pairs still incomplete. Coverage differs, so this does not isolate integrality alone. |
| Combine all nine decompositions: first selection | 34 buses; pool bound 33; 18,466 columns | Cannot reach 32 within this selected pool. Fleet optimum remains open between 33 and 34. |
| Combine all nine decompositions: retain LP support | 34 buses; pool bound 32; 18,456 columns | Target 32 remains possible but unproved within this selected pool. |
| Chain 3, target 26 | CG certified in 113.2 minutes | Integer result pending in this snapshot; this is not a 26-bus integer match. |

Both decomposition treatments have all 46 MIPs complete; none improved its best contributing partition. They are 92 searches on one 750-trip parent, not 92 independent instances. The all-nine searches each received four hours. No parent CG certificate is produced by combining pools.

Chain 3 k26 reaches weighted objective 2,601,094.794385 and fractional route weight 26. Its minimum reduced cost is −0.0000827103, within epsilon 0.0001 on the tested event graph. The weighted objective, fractional route count and fleet-only lower bound are distinct. Canonical result and journal hashes were independently checked in [k26 source verification](k26_source_verification.json).

Original larger-chain results: 58 CG endpoints (45 certified, 13 capped); 57 one-hour MIPs (33 targets, 24 misses). Earlier longer searches recovered 15 misses, leaving nine unresolved targets in this snapshot. Largest individual original target matches by chain are 23, 22, 24, 23, 24 and 24; a separate longer search matched chain 2 at 25. This does not imply every smaller target was recovered. The new chain 4 k25 MIP finds 26 buses with bound 25, unproved. Chain 1 k24 CG is capped at 239.6 minutes, last reduced cost −0.002991.

The cumulative-time comparison is retrospective: historical execution revisions and hardware varied. It motivates the new matched seed-content pairs, rather than isolating every cause of the warm/fresh difference.

Baseline physics: covering, 240 kWh battery and 240 kW charging, start fee 5, no shared charging-capacity or terminal-SOC floor. Fleet proofs concern supplied route pools; individual replay and shared-capacity checks are separate.

The chain 3 k8 integer-seed retry has completed at nine buses, proved within its pool. Its preempted first attempt remains recorded with lost allocation time; it is not a second independent case.

[Every small-seed result](seed_results.csv) · [Twelve completed pairs](completed_seed_pairs.csv) · [All-nine source evidence](decomposition_allnine.csv) · [Chain and equal-time tables](../../cumulative_budget_20260913/status_20260914T225805Z/README.md).

Source snapshot SHA-256: 45fcc45515507d942983916e339b29e4ca0e358fc32a0045afd7d452bcb85c09. Register: 2,930 artifact/stage records, 63 source groups; counts are not independent samples.
