# Combined pools and small previous-k warm starts

Verified source collection: 2026-09-14T17:29:07.740702+00:00.

## Combining three saved pools

| Case | Target | Best original / 200-column / complementary fleets | Union buses | Union fleet bound | Fleet proved in union |
|---|---:|---|---:|---:|---|
| c1_k08_union_mip | 8 | 9 / 9 / 9 | 9 | 8 | no |
| c2_k08_union_mip | 8 | 9 / 9 / 9 | 8 | 8 | yes |
| c4_k08_union_mip | 8 | 9 / 9 / 9 | 9 | 8 | no |
| c5_k08_union_mip | 8 | 9 / 9 / 9 | 8 | 8 | yes |
| c5_k10_union_mip | 10 | 11 / 11 / 11 | 10 | 10 | yes |
| c6_k10_union_mip | 10 | 11 / 11 / 11 | 11 | 10 | no |

Every union preserves its three frozen source pools and adds no new pricing run. Result, input and construction hashes are verified; selected routes pass individual replay. Fleet proofs apply only to the combined pool. Open bounds do not prove the target absent. Because some individual source searches still have open gaps, a union recovery alone does not prove that combining columns was necessary.

All three sources and the union retain their actual search budgets and timing in the CSV. These baseline experiments omit shared charger capacity and a terminal-SOC floor. No new full-model pricing certificate comes from constructing a union.

## Small warm starts with completed CG

| Case | Previous sequences selected / added | Import seconds | CG minutes | CG certified | Integer buses | Fleet proved in pool |
|---|---|---:|---:|---|---:|---|
| c2_k08_integer | 7 / 7 | 1.43 | 17.9 | yes | pending | pending |
| c2_k08_lpweight | 7 / 7 | 1.22 | 13.1 | yes | pending | pending |
| c3_k08_integer | 7 / 7 | 0.67 | 14.8 | yes | pending | pending |
| c3_k08_lpweight | 7 / 7 | 0.67 | 12.2 | yes | pending | pending |
| c6_k08_integer | 7 / 7 | 0.64 | 9.1 | yes | 8 | yes |
| c6_k08_lpweight | 7 / 7 | 0.48 | 10.2 | yes | 8 | yes |
| c6_k10_integer | 9 / 9 | 0.71 | 20.3 | yes | pending | pending |
| c6_k10_lpweight | 9 / 9 | 0.74 | 22.0 | yes | pending | pending |

These are paired treatments on 18 selected inputs, not 36 independent datasets. Compare previous integer-selected sequences with an equal number chosen by LP weight. Selected sequence counts are matched; coverage and actual child additions can differ. Prior CG and MIP costs remain in the manifest and must be included in end-to-end comparisons. Historical fresh/full-pool runs are context, not automatically matched timing controls.

Only completed publications enter this table; running attempts and late scheduler-only outputs remain separate. A CG certificate does not replace the integer search. No overall winner is inferred from the first completed pairs.

[All seed cases, values and hashes](seed_results.csv); [union results and source-pool evidence](union_results.csv).
