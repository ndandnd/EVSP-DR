# Research results — 15 September, 00:16 EDT

**New one-hour fleet matches:** chain 1 reaches 25 buses and chain 2 reaches 26. Their CG runs did not converge before their time limits, but their saved pools support target fleets.

| Case | Buses | Pool fleet bound | Minutes to fleet proof | Total MIP minutes | CG minutes / certificate |
|---|---:|---:|---:|---:|---|
| Chain 1, target 25 | 25 | 25 | 10.9 | 60.1 | 239.4 / no |
| Chain 2, target 26 | 26 | 26 | 7.2 | 60.1 | 239.6 / no |

The pools contain 237,601 and 180,809 columns. Individual-route replay passes. There are 73 and 76 overcovered trips respectively, without separate duplicate-removal validation. Shared charger capacity and terminal energy requirements remain absent from the baseline. Charging optimality is still open. These are finite-pool fleet proofs, not full-model optimality claims.

Largest individual one-hour matches across chains 1–6 are **25, 26, 27, 23, 24 and 26**. The original k16–25 batch is complete: 60 CGs (45 pricing certificates, 15 time limits) and 60 MIPs (35 target matches, 25 misses). Separate longer searches recover 24 of the 25 misses; only C5 k25 remains unmatched across those treatments. Largest matches do not imply uninterrupted success under the original one-hour allowance. [Original chain results](../../cumulative_budget_20260913/status_20260915T040735Z/README.md).

**Compact starts have their first open integer gap.** All 36 CGs are certified; 30 MIPs are published, 29 match target, and six k15 MIPs remain unpublished.

| Chain 3, target 15 | Initial trip sequences | Final pool columns | Integer buses | Pool fleet bound | Fleet search |
|---|---:|---:|---:|---:|---|
| Core | 198 | 26,239 | 17 | 15 | Three-hour limit; gap open |
| Core expanded to 512 | 512 | 24,459 | 15 | 15 | Proved in 4.4 seconds |

Both CGs certify the same weighted LP objective to numerical precision: 1,500,507.4203245. They use the same input, model, code and limits; inherited routes differ. The core pool may still contain a 15-bus solution. Its missing target has not been proved to be a pool limitation. Different final pool composition can affect integer search even when the weighted LP objectives agree; a larger column count alone does not guarantee a better incumbent. Both selected solutions pass individual replay; charging optimality remains open. [Every compact-start result, proof flag and source hash](../../overnight_evening_20260914/status_20260915T040735Z/README.md).

**Capacity pricing: reference finished; caching reached its deadline in both fixed-state tests.**

| Frozen state | Reference call | Cached call |
|---|---|---|
| One duty, 17 trips | Completed in 214.9 minutes | No completed result by 238.4 minutes |
| Two duties, 23 trips | Completed in 191.5 minutes | No completed result by 237.1 minutes |

Each pair has identical source-pool and raw-dual hashes and the same RMP objective and matrix sizes. The reference returns negative reduced costs of −599985.576 and −799976.896. The cached runs return no completed pricing result, stop at their pricing deadlines, and carry no convergence certificate. These are timing observations with a cutoff, not completed solve times, preemptions or wrapper failures. The earlier wrapper failures remain separate. No caching speed improvement is demonstrated on these two hard states; hardware and execution timing limit a general claim.

The one-duty RMP has 7,817 rows, 49 columns and 344 nonzeros. The two-duty RMP has 7,823 rows, 50 columns and 249 nonzeros; its measured LP solve wall time is 0.024 seconds. These are individual pricing calls, not complete CG runs. [Paired times and source hashes](pricing_comparison.csv) · [Full diagnostics](pricing_calls.json).

**Larger compact starts:** the first two k20 CGs are certified, both on chain 3. The 306-sequence core takes 110.7 minutes; the 512-sequence treatment takes 102.7 minutes. Both certify weighted objective 2,000,780.100492 with fractional route weight 20. Their MIPs are pending. This larger cohort uses a0e0 code; the k8–15 cohort uses e091, so a comparison across cohorts does not isolate size. [Result values and source bindings](compact_large_results.csv).

New continuation CGs C5 k26 and C6 k27 stop at their time limits after 239.8 and 239.9 minutes, with minimum reduced costs −0.047977 and −0.060112. Their RMP objectives are not certified full-model bounds. Follow-up MIPs and previous-k dependencies remain intact.

The completed reserve screen and nine remaining-gap MIP reruns are unchanged from the prior report. No new LP-addition pair has completed. [Prior verified results and model scope](../status_20260915T030644Z/README.md).

**Queue:** 63 running / 38 true dependency waits; 33 held historical tasks excluded. No new confirmed preemption or execution failure. The study has 905 attempt records. This check submitted, cancelled and requeued no jobs. Snapshot `20260915T040735Z`, SHA256 `f2c51900c5f254eb57343afecc2f44c8f8b298310fd60256a4db44d8910a6507`. Workbook: 3,202 records across 70 source groups, all six supplements retained.

[New endpoint source paths, hashes and proof fields](new_endpoints.csv) · [Source validation](validation.json).
