# Seed-content overlap audit

None of the18pairs have identical selections. Equal sequence counts do not equal trip coverage: integer selections cover more distinct parent trips in every pair. The arms are distinct treatments, not statistically independent samples.

| Pair | Each | Exact overlap | Integer trips | LPweight trips |
|---|---:|---:|---:|---:|
| c1_k08 | 7 | 0 | 177 | 107 |
| c1_k10 | 9 | 0 | 211 | 153 |
| c1_k15 | 14 | 0 | 310 | 220 |
| c2_k08 | 7 | 1 | 151 | 73 |
| c2_k10 | 9 | 0 | 219 | 142 |
| c2_k15 | 14 | 0 | 343 | 200 |
| c3_k08 | 7 | 0 | 142 | 113 |
| c3_k10 | 9 | 0 | 191 | 158 |
| c3_k15 | 14 | 0 | 281 | 146 |
| c4_k08 | 7 | 0 | 142 | 94 |
| c4_k10 | 9 | 0 | 203 | 123 |
| c4_k15 | 14 | 1 | 326 | 193 |
| c5_k08 | 7 | 0 | 156 | 100 |
| c5_k10 | 9 | 0 | 183 | 113 |
| c5_k15 | 14 | 0 | 295 | 221 |
| c6_k08 | 7 | 4 | 113 | 91 |
| c6_k10 | 9 | 0 | 178 | 118 |
| c6_k15 | 14 | 1 | 347 | 199 |

Ordered stable-sequence and unordered trip-set overlaps coincide for these18pairs. Hash-bound source mappings and seed records are retained in seed_content_source.json.

Existing fresh C1/C2/C4k15 endpoints certify at4.85/4.33/4.11hours, beyond4hours. No historical240minutesnapshot exists. Last logged pre240min pools are at14392.86/14395.04/14399.40seconds, each fractionalrouteweight15 with negative reducedcost; none has a fullpricingcertificate. These can support retrospective iteration-cutoff pool controls with matched MIPs, not fabricated240minuteCG endpoints. A new4hCG adds prospective timing repetition, but is unnecessary to obtain this narrowly defined historicalpool comparison.
