# Mixed vehicle groups in the saved LP solutions

**F4 VERIFIED:** all 102 recorded LP endpoints assign material positive weight to routes containing trips from both GIRO vehicle groups. **F2 causal explanation UNRESOLVED:** the association below does not prove that mixing causes the k−1 outcomes or that an unmixed alternate optimum cannot exist.

No solver was run. All 102 sparse supports were available, and every raw CG file hash and input CSV hash matched the audited evidence. “Mixed lambda” means the sum of LP route weights on routes serving both groups. A value of 5 means five fractional bus-equivalents assigned to mixed routes; it is not an integer fleet count.

| Comparison | Cases | Mean mixed lambda (range) | Mean mixed share | Pooled mixed lambda / total lambda | Mixed positive routes / all positive routes |
|---|---:|---:|---:|---:|---:|
| k−1 endpoints | 9 | 5.0409 (4.2915–5.9002) | 17.38% | 45.3678 / 261.0000 | 740 / 4852 |
| All other endpoints | 93 | 2.5545 (0.8668–4.4970) | 11.47% | 237.5719 / 2178.0000 | 3652 / 37069 |
| Other endpoints, k27–32 only | 27 | 2.4105 (0.8668–4.4970) | 8.28% | 65.0825 / 792.0000 | 1045 / 14352 |

The mixed weight is substantial, not numerical noise. Matching the k range strengthens the descriptive difference: **17.38% versus 8.28%**. It is also higher within every exact k. However, the nine cases are from chains 4/5, and the chains are nested. Instance composition, chain history, LP degeneracy and other omitted constraints remain confounders. All 93 other endpoints also mix groups, so the mere presence of mixing does not distinguish the nine cases. Group segregation is a candidate mechanism to test, not an established explanation.

| Target k | k−1 cases | Mean mixed share, k−1 | Other cases | Mean mixed share, others |
|---|---:|---:|---:|---:|
| 27 | 1 | 16.51% | 5 | 10.05% |
| 28 | 1 | 16.89% | 5 | 9.45% |
| 29 | 1 | 18.89% | 5 | 8.02% |
| 30 | 2 | 17.53% | 4 | 7.45% |
| 31 | 2 | 17.11% | 4 | 7.15% |
| 32 | 2 | 17.40% | 4 | 6.92% |

## The nine k−1 endpoints

| Case | Total route weight | Mixed lambda | Mixed share | Mixed positive routes | Pricing certificate? | Saved endpoint |
|---|---:|---:|---:|---:|---|---|
| w5_k27 | 26 | 4.2915 | 16.51% | 75 | No | Final pool re-solve |
| w5_k28 | 27 | 4.5612 | 16.89% | 80 | No | Final pool re-solve |
| w5_k29 | 28 | 5.2902 | 18.89% | 86 | No | Last solved iterate |
| w4_k30 | 29 | 5.5503 | 19.14% | 90 | No | Final pool re-solve |
| w5_k30 | 29 | 4.6166 | 15.92% | 78 | No | Final pool re-solve |
| w4_k31 | 30 | 5.6317 | 18.77% | 93 | No | Final pool re-solve |
| w5_k31 | 30 | 4.6353 | 15.45% | 64 | Yes | Final pool re-solve |
| w4_k32 | 31 | 5.9002 | 19.03% | 99 | No | Final pool re-solve |
| w5_k32 | 31 | 4.8908 | 15.78% | 75 | No | Final pool re-solve |

## Endpoint alignment

- 90 supports are final-pool re-solves. Twelve are last-good iterates whose journals contain 30 later columns without a completed subsequent LP. They match the table's declared endpoint, but are not solutions of the expanded final pool. w5_k29 is the sole k−1 case in this group.
- 15 final-pool re-solves have a different objective from the last pricing iteration. Both objectives and their alignment flags are recorded; no earlier pricing certificate is transferred to a different endpoint.
- Only 50/102 endpoints have the recorded pricing certificate: 49 in the k cohort and one, w5_k31, in the k−1 cohort. Certified-only and final-pool-only summaries appear separately in group_summary.csv. Route weight alone is not a fleet lower bound or integer proof.
- The serializer saves strictly positive lambdas only. Tiny nonpositive values within solver tolerances are omitted. The largest positive-support versus signed-mass discrepancy is 9.420731e−7 buses; largest objective discrepancy is 0.094235 currency units. All saved positive weights enter totals; route counts above 1e−7 are also reported. Nothing was reconstructed or silently rounded.

## Source mapping

The original Partille PDF, pp. 1 and 4, assigns route 21 to 18E1 and local routes to 18E2. The frozen master shows every 134-prefix regular duty on Route 5021 and every 133-prefix regular duty on local 55xx routes. The prefix mapping is therefore an inference corroborated by source rows, not a literal prefix rule quoted from the PDF. It independently matches every trip in the existing action3 group map; service-day suffixes are preserved.

Primary attachment: `outputs/meeting_20260910/giro_email_sources/Transdev Electric Partille Mon-Thu december 2023.pdf`, SHA256 `2c6d78180c43f4ec31c29b5bd495f1a4b48d96115ca76babfb4b88114d173ef6`. Source interpretation: `outputs/model_fairness_audit_20260913/giro_requirements_audit.md`.

## Evidence files

- `per_case.csv`: all 102 cases, exact source/input/support hashes, mixed and total lambda, route counts, endpoint alignment and certificate flags.
- `group_summary.csv`: sums, means and ranges for all cohorts, endpoint-quality subsets, k27–32 and each exact k.
- `per_route.csv.gz`: every positive route's lambda and original duties/groups.
- `saved_final_supports.json.gz`: archived trip lists and lambdas from published CG results.
- `summary.json`, `collection_receipt.json`, `artifact_hashes.json`: provenance and no-solver records. Three classification tests pass.

Exact historical identity uses the raw source-file hash against the audited table's `bound_source_sha256`. The generic reconstructed payload digest excludes collector-added metadata; it is not the older collector-payload digest and should not be compared to it.

## Additional no-solver interval bound

For each group, select the largest set of mandatory service trips overlapping at one time. A group-specific route can cover at most one of these trips. Summing their covering constraints gives `sum(lambda for group g) >= overlap_count(g)`. With mandatory group separation, the two groups have disjoint route variables, so the two bounds add even if their peak times differ. This bounds fractional fleet weight as well as integer fleet size.

Production `build_problem` maps `Start1`/`End1` directly to integer minutes (`audit_giro_known_columns.py`, lines 175–193; `_total_minutes`, lines 84–86). This check uses conservative half-open service intervals `[start,end)` and ignores deadheads/charging. Hours beyond 24 remain unchanged. Equal-time endpoint and overnight-time checks pass.

| Case | 18E1 overlap | 18E2 overlap | Separated-group lower bound | Observed LP route weight | Excludes that route weight? |
|---|---:|---:|---:|---:|---|
| w5_k27 | 9 | 16 | 25 | 26 | No |
| w5_k28 | 10 | 16 | 26 | 27 | No |
| w5_k29 | 10 | 17 | 27 | 28 | No |
| w4_k30 | 10 | 19 | 29 | 29 | No |
| w5_k30 | 11 | 17 | 28 | 29 | No |
| w4_k31 | 10 | 20 | 30 | 30 | No |
| w5_k31 | 12 | 17 | 29 | 30 | No |
| w4_k32 | 11 | 20 | 31 | 31 | No |
| w5_k32 | 12 | 18 | 30 | 31 | No |

**F4/F2: this particular lower-bound argument does not establish that group mixing is necessary for any of the nine k−1 endpoints.** It equals k−1 for the three chain-4 cases and k−2 for the six chain-5 cases. A weak bound is not evidence that a group-separated solution exists. The bound equals k for 43 of the other 93 cases.

`interval_bound_check.csv` contains all 102 cases and explicit witness times/trip IDs. `interval_bound_summary.json` contains the proof, source hashes and result hash. This additional check does not change the saved-lambda audit or its observational conclusions. No solver was invoked.
