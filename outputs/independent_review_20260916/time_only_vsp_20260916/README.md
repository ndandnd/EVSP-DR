# Exact time-only fleet bounds for 102 chain instances

External audit received17September: independent recomputation on12instances and consistency checks on all102 agree. [Scope, archived reports and remaining qualifications](../execution/advisor_audit_acceptance_20260917/README.md).

**F4/F2 VERIFIED:** keeping the two GIRO vehicle groups separate requires at least **k buses in every one of the 102 instances**, even after removing energy and charging constraints. Each group's bound equals its own GIRO duty count. If groups may mix, the time-only minimum is **k−1 in exactly the same nine cases** and k in the other 93.

These are exact combinatorial results for a declared relaxation of the production travel model, not another restricted-column LP solve. All computation ran locally; no cluster jobs or commercial optimizer were used.

| Model | Minimum equals GIRO k | Minimum equals k−1 |
|---|---:|---:|
| Groups kept separate | 102 | 0 |
| Groups may mix | 93 | 9 |

For each group separately, the time-only minimum equals its original duty count in 102/102 cases. The proposed “local-route electrification premium of 1–3 buses” was inferred from overlap alone; adding travel compatibility eliminates that gap in this time-only relaxation. This does not show energy never matters, particularly after changing physics.

## The nine cases

| Case | 18E1 time-only minimum | 18E2 time-only minimum | Separate total | Mixed minimum | Recorded mixed LP route weight |
|---|---:|---:|---:|---:|---:|
| w5_k27 | 9 | 18 | 27 | 26 | 26 |
| w5_k28 | 10 | 18 | 28 | 27 | 27 |
| w5_k29 | 10 | 19 | 29 | 28 | 28 |
| w4_k30 | 10 | 20 | 30 | 29 | 29 |
| w5_k30 | 11 | 19 | 30 | 29 | 29 |
| w4_k31 | 10 | 21 | 31 | 30 | 30 |
| w5_k31 | 12 | 19 | 31 | 30 | 30 |
| w4_k32 | 11 | 21 | 32 | 31 | 31 |
| w5_k32 | 12 | 20 | 32 | 31 | 31 |

## What is proved, and what remains open

- Group mixing is **necessary** for a route-weight k−1 solution in these nine instances: separated routes obey the exact lower bound k. This is stronger than the earlier observational mixed-lambda comparison.
- Removing group restrictions is worth exactly one bus **in this time-only relaxation** for the nine cases. The matching paths are not claimed to satisfy battery or charging constraints.
- The mixed time-only bounds equal all 102 recorded mixed LP route weights numerically. This supplies an independent fleet lower bound. It does not certify the electricity component of the weighted objective, convert floating LP solutions into rational certificates, or find a mixed integer EV solution.
- F4's reoptimized original duties provide k-bus **continuous charging** witnesses under the baseline 240/240, reserve-zero, free-ending-SOC model. Combined with this lower bound, a verified union of those witnesses closes continuous-model segregated fleet optimality. F4 did **not** establish that those charging schedules are representable on the event lattice. Event-model integer optimality still needs an event-feasible k-bus witness; no blanket claim is made here.
- For the reviewer's 12 unmixed-pool LP results, a verified feasible route-weight k solution and this lower bound would close the fleet-only LP gap for those cases. That does not automatically prove the weighted objective optimal, an integer solution feasible, or all 102 unmixed event LPs solved.

## Travel model and why the lower bound is safe

The source hashes match the production data: `Ref_dict.csv` for location aliases and `par_ref_dhd.csv` for deadheads. Production treats reference pairs as symmetric and retains the minimum duration among duplicate entries. `Start1`/`End1` are exact integer minutes; hours beyond 24 are preserved.

A raw direct-pair graph would be unsafe as an EV lower bound: some direct entries are absent or slower than travel via a charging location. We therefore compute shortest paths through **all reference locations** using the same nonnegative travel times. This is broader than actual production charger paths, deliberately making the bound conservative. Missing direct arcs remain missing in the separate direct-only comparator; they are not silently treated as zero.

Durations are represented exactly in half-minute units. Production first tests raw-minute feasibility and stores travel using ceiling minutes. The relaxation uses unrounded durations and can therefore only make a connection easier. It also removes energy, charging duration, depot pull-out/return, maximum-wait and other graph restrictions. Every allowed production deadhead/charging path is at least as long as the shortest reference path used here.

**Check against actual LP routes:** all 41,921 archived positive routes, comprising 1,040,239 adjacent service-trip connections, satisfy the relaxation. Source hashes match the reviewed endpoints; zero violations occurred.

Direct-only and closure minima differ in nine of the 306 case/group comparisons—exactly the nine mixed cases. Direct-only would incorrectly suggest k throughout; those numbers are recorded as a distinct model and are not used as EV lower bounds.

## Exact certificates

For n service trips, connect trip i to trip j when `end(i) + relaxed_travel(i,j) <= start(j)`. Positive service durations ensure a DAG. A maximum bipartite matching M gives the minimum vertex-disjoint path cover `n−|M|`. Each result stores a matching, an equally sized vertex cover, and a path decomposition covering every trip exactly once. An independent verifier reconstructs the graph and checks all 612 certificates (102 cases × 3 group choices × 2 travel models).

For the closure relaxation, each of the 306 results also stores an antichain of size `n−|M|`. The verifier checks **pairwise non-reachability**, not just the absence of direct edges. This matters because service-trip travel can make the compatibility graph nontransitive even when deadhead distances obey the triangle inequality.

A route can contain at most one antichain trip. Summing the covering constraints of those trips gives `sum(lambda) >= |antichain|` for nonnegative route weights. Group-separated route variables are disjoint, so the two group bounds add. Matching paths give the same upper value in the relaxed model. Thus both the integer time-only minimum and fractional route-weight lower bound are certified without floating optimization.

## Files and reproduction

- `per_case.csv`: all 102 cases, per-group and mixed minima, overlap checks, direct-only comparators and hashes.
- `certificates.json.gz`: matching, minimum vertex cover, paths and antichain witnesses with ordered trip IDs.
- `verification.json`: independent certificate and saved-support connection checks.
- `summary.json`: input/source/code hashes, proof scope and cohort counts.
- `gap_distribution.csv`: group-level fleet differences for all cases.
- `time_only_fleet_counts.png` / `.pdf`: presentation figure. Its data and labels are editable in `plot_data.csv`, `plot_labels.json` and `plot.py`; interpretation remains in this text.

Run `python3 time_only_vsp.py`, then `python3 verify.py`. Five tests cover exhaustive small matching comparisons, missing direct arcs repaired by detours, alias mapping, half-minute precision and service intervals extending past midnight. Production source references are the pinned `audit_giro_known_columns.py` (location/travel mapping) and `pricing_dp_og.py` (time compatibility and stored travel). Their hashes are recorded.

This directory does not modify the reviewer's write-up, experiment register, Google Doc or existing solver outputs.
