# Unicorn audit — 25 September 2026, 13:30 EDT / 17:30 UTC

Unicorn is reachable. The EVSP–DR spatial-tariff campaign has **12 running jobs: five fresh CGs and seven cleanups**, with five additional cleanups waiting on their actual running CG predecessors. This is useful managed work; no recovery submission or queue mutation was needed. Every one of these 17 active/pending jobs excludes `scaglione-compute-01`. Other projects (`evsp-incentive`, `evspv2gdp`) and held historical jobs were untouched. Recurring monitoring was not restarted.

This read-only collection finishes the audit of the previously launched full40 extension and C1 k8 cap/stopping pilot. **Full40 ends with 0/48 own-k target matches and no pricing certificates. The cap pilot produces no dive incumbent in any of its four 900-second arms.** Scheduler completion is separate from those scientific outcomes.

## Full40: all 48 cases are terminal

The k33–40 extension has 48 completed CG processes and 48 completed finite-pool MIP processes, each with scheduler exit `0:0`. All CGs stop at `wall_limit` without a pricing certificate. All MIPs report `TIME_LIMIT`, with no fleet optimality proof. Final integer fleet counts exceed their own k in all 48 cases. These are the standard four-hour-CG/one-hour-two-stage-MIP results; earlier longer searches and seed repeats remain separate evidence.

| Chain | k40 trips | Final fleet incumbent | Finite-pool fleet-only bound | Overcovered trips |
|---|---:|---:|---:|---:|
| C1 | 948 | 43 | 39 | 200 |
| C2 | 947 | 44 | 39 | 262 |
| C3 | 947 | 44 | 39 | 202 |
| C4 | 948 | 47 | 39 | 270 |
| C5 | 946 | 44 | 39 | 241 |
| C6 | 947 | 45 | 39 | 242 |

Bounds above are rounded representations of values within numerical tolerance of 39. They apply to the saved finite pools. The k40 restricted masters also have fractional route weight approximately 39, and weighted objectives approximately 3,901,685–3,901,711; **neither is a certified full-model lower bound**. The separate route-weight and weighted-objective fields are preserved in the CSV.

Native individual-route physical replay passed at all 48 endpoints, under the historical physics. Duplicate-service removal and shared-charger capacity remain unvalidated at all 48. These endpoints are covering-model incumbents, not validated exact-once operating schedules. The frozen full40 inputs contain three compatible service-day variants; they are not six identical 40-duty inputs. Previously verified target32 matches on all six chains remain the latest target attainment, using the separately recorded longer/seed searches; their finite-pool fleet proofs remain C1/C2/C3/C6 only, with C4/C5 open at bound31.

Editable evidence: [all 48 cases](full40_cases.csv), [six k40 endpoints](full40_k40.csv), [native accounting](full40_sacct.txt). Each CSV row has input SHA-256, CG/MIP source paths and SHA-256 values, bound scope, route replay and dispatch-validation flags.

## C1 k8 cap/stopping pilot: completed, no incumbent-transfer comparison

| Job | Dive cap | Stopping rule | New columns | Dive wall seconds | Follow-up MIP fleet / bound |
|---|---|---|---:|---:|---|
| 889582 | Certified overlap floor 8 | First feasible | 4,026 | 851.87 | 9 / 8, open |
| 889583 | Certified overlap floor 8 | Fixed budget | 8,076 | 849.26 | 9 / 8, open |
| 889584 | Explicit cap 9 | First feasible | 8,084 | 845.72 | 9 / 8, open |
| 889585 | Explicit cap 9 | Fixed budget | 7,454 | 849.92 | 9 / 8, open |

All four scheduler jobs completed successfully. All four dives stop at their wall allowance without a validated integer incumbent. The independently budgeted 2,700-second MIPs without own-dive-incumbent transfer all finish at nine buses, finite-pool fleet-only bound eight, and `TIME_LIMIT`; none proves fleet optimality or hits target8. Their native greedy MIP starts remain present. All four with-own-incumbent branches are explicitly skipped because there was no validated dive incumbent to transfer. There are four completed MIPs, not eight paired MIP results.

The result does not establish a benefit of either stopping rule or an incumbent-transfer effect: the first-feasible condition was never met, and the transfer arm could not run. It also does not negate the earlier longer dive that found eight buses after about 2,057 seconds. Each of these four pools passes native individual-route replay, but has 31–34 overcovered trips; duplicate cleanup and shared capacity remain unvalidated. No global pricing or infeasibility certificate follows from a restricted dive-node LP.

Editable evidence: [four-arm table](dive_cap_cases.csv), [four full dive manifests and native MIP log endings](dive_details.json), [native accounting](dive_sacct.txt). The local launch manifest (local source: `outputs/research_management_20260923/dive_cap_pilot/manifest.json`; retained in the primary workspace, outside this packet) and launch record (local source: `outputs/research_management_20260923/dive_cap_pilot/LAUNCH.md`; retained in the primary workspace, outside this packet) remain unchanged.

## Current spatial-tariff work and already completed fallbacks

At the single 17:30 UTC collection, the original 72 jobs comprise 55 scheduler completions, 12 running jobs and five true dependencies. The four separately managed fallback/cleanup jobs also completed, giving 59 scheduler completions out of 76 total original-plus-fallback jobs. Scheduler completion does not mean that a usable fresh-CG schedule exists: two original cleanup attempts explicitly skipped because the finite fresh pool had no integer selection. Their already-submitted fallback pairs are complete; no duplicate fallback was submitted here.

G1 passes: the peak18 fixed-duty control costs 96.949925109715, reproducing its reference tolerance. G2 remains pending because its fresh CG 462741 is still running and cleanup 462742 depends on it. Broad new tariff conclusions should retain that unresolved reproduction gate.

| Spatial fallback | Fallback → cleanup jobs | Exact-once fleet | Grid charging cost | Continuous charging cost | Selected route origin |
|---|---|---:|---:|---:|---|
| Route21, 4808 price ×0.75 | 476611 → 476612 | 5 | 154.690 | 152.1403 | All five fixed-duty frontier |
| Route21, PARX price ×1.5 | 476622 → 476623 | 5 | 190.960 | 187.7132 | All five fixed-duty frontier |

These union-pool fallbacks combine the saved fresh-CG pool and the fixed-duty frontier. They preserve the original CG pricing certificates but are **not pure fresh-CG integer successes**. Both match their fixed-duty comparator; no trip rerouting benefit was found. Cleanup reports finite-pool fleet/charging optimality, exact-once coverage, individual-route replay, zero duplicated trips and unvalidated shared capacity. Aggregate terminal energy, zero reserve and zero start fee apply; the historical full40 physics is different. The source-bound summaries and hashes are saved in `snapshot.json` under `spatial_summaries`.

The spatial campaign is actively managed in its existing campaign record (local source: `outputs/independent_review_20260925_spatial_tariffs/campaign/README.md`; retained in the primary workspace, outside this packet). This audit does not rewrite its evolving collector output. The `spatial_collected` section of our snapshot preserves the existing **17:13 UTC** collection, which predates the fallback completions; it must not be described as a fresh 17:30 collection of final tariff tables. [Active job/dependency table](active_spatial_jobs.csv), [spatial accounting](spatial_sacct.txt), [complete user queue](queue.txt).

## Provenance and validation

- Full40 CG execution commit `a0e0bb7681c8451e3cbbbfa06aef390026d9af4b`; MIP commit `871d057e1067411f09581e37d78f7c1ca43f68bb`. Covering minimization; objective 100000 per route plus flat-price electricity plus five per charge start; 240 kWh battery, 240 kW charging, zero reserve, no terminal-energy floor, no shared capacity. Initialization reoptimizes each true previous-k route pool on the child graph. Resources: graph 2 CPU/96G/37 h; CG 8 CPU/96G/5 h allocation with 4 h solver allowance; MIP 8 CPU/24G/2 h allocation with 3,600 s total two-stage solver allowance. The manifest and input/source hashes are frozen in `snapshot.json`.
- Cap pilot execution commit `d8fbf40b922c158d53959a12ac950371e7c7c6fe`, one C1 k8 case and seed20260921, frozen fresh pool only, no imported GIRO or sequential witness, original weighted covering objective and historical physics. Each arm requests default_partition, 8 CPU,16G,3 h, requeue and compute-01 exclusion. Dive allowance900 s, follow-up MIP allowance2700 s each; only own-dive predecessor data is eligible. Source status hash `c024cc4e235234ed4244abaa1e7738f3600c264a2ffe4c05c5026198b69d56b0`; source journal hash `2e75b32ab3f2c11e7677892844b33c9fe8c0351529927d08980370c113d2845c`. Output manifests/log hashes are bound to each execution receipt.
- Spatial execution commit `4741cad889644ecf5aa0499894341f3f273ed5fa`; fallback uses its separately recorded follow-up source. Both arms use 240 kWh,350 kW,zero reserve,zero start fee,2.5 kWh/5 min graph discretization, aggregate terminal-energy targets and no shared capacity. Fresh CG starts independently for each cell; cleanup has an afterok dependency on its own selection job. Resources and complete input hashes are in the captured spatial manifest and linked existing campaign record.
- [428/428 checks pass](verification.json): full40 source/input/commit identity, CG-to-MIP hash linkage, true previous-k lineage, scheduler endpoints, proof/validation fields; cap receipt/result/manifest/log hashes and explicit skip semantics; active spatial compute-01 exclusions and true dependencies. This is an artifact consistency audit, not a new physical replay or solver execution.
- Collection script [remote_collect.py](remote_collect.py) is read-only; [audit_snapshot.py](audit_snapshot.py) reproduces the tables and checks locally. `snapshot.json` contains artifact paths, exact full-file SHA-256 hashes and metadata; large JSON list fields are omitted recursively as explicitly marked. `dive_details.json` retains complete dive manifests. [Local file hashes](SHA256SUMS.json) cover these outputs.

## Recovery and parallel backlog

No nonhistorical broken dependency appears in this scoped live queue. All five waiting spatial cleanups have running predecessors. Historical rvS jobs remain held/superseded, including job341294's failed old dependency341292 and explicit `SUPERSEDED_PENDING_AUDIT_strict_packed_20260921` comment; that is not a request to restart the old strict chain. Held array537227 is preserved.

Useful work already proceeding is: finish G2 reproduction, finish the four local CG→cleanup pairs and the local JON_A cleanup, finish the remaining route21 cleanups, then refresh paired comparisons with fallback labels and all validation gates. Full40 now needs analysis of the large finite-pool integer gap rather than an automatic rerun of completed jobs. The negative cap pilot needs a reviewed follow-up design before any different budget, cap or seed is launched. No new scientific setting or redundant run was submitted to inflate utilization.
