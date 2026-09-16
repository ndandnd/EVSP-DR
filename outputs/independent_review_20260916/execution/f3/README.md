# F3 — reduced-cost lower bounds

**VERIFIED, with corrected mathematics:** useful lower bounds can be reconstructed without rerunning CG. The frozen102-row table contains **52**, not54, uncertified CG endpoints. All52 yield an integer fleet lower bound equal to their rounded reported route weight: k in the ordinary cases, k−1 in the nine exceptions. This does not say that the fleet-only LP optimum equals that weight.

**REFUTED as written:** integer incumbent fleet K alone need not bound fractional route mass under the weighted objective. From 100000 Σλ* ≤ z* ≤ U, the justified bound is **K=U/100000**, including incumbent nonfleet costs. Also, a pricing reduced cost must be paired with its own LP iteration, not a later pool re-solve.

## Derivation and implementation

For the covering LP, let π≥0 be trip duals, D=Σπ, and δ≤0 a lower bound on every route's reduced cost c_r−a_r·π. For any feasible λ with Σλ≤K:

    Σ c_r λ_r ≥ D + δ Σλ_r ≥ D + δ K.

All route costs are at least M=100000. A feasible integer incumbent costs at most U, so an optimal weighted LP solution has Σλ≤U/M. This gives the weighted-objective lower bound L=D+δU/M.

To infer an **integer fleet** lower bound, independently bound the nonfleet cost of every route by Q. Any m-bus solution then costs at most m(M+Q); consequently m≥ceil(L/(M+Q)). This is a fleet lower bound, not ceil(L/M), nor ceil(fractional route weight).

Here Q=240×26×0.0992 + 5(q+1). Charging cannot occupy more than the26-hour horizon. q is the maximum number of nonoverlapping input trips found by earliest-finish interval scheduling while ignoring all travel; it upper-bounds trips on any route. The event graph allows at most one charging activity between successive trips or after the last trip; q+1 is conservative. This bounds costs even when trips are duplicated across buses. Q≤1044.008 across all102 cases. We use U=incumbent_fleet×(M+Q), a deliberately loose upper bound, rather than extracting a possibly mismatched charging-cost field.

`analyze.py` uses the RMP objective and global minimum reduced cost from the **same recorded pricing iteration**.37/52 uncertified endpoints also retain that iteration's dual vector, allowing a direct sum check. For the other15 it invokes strong duality of the successful LP solve; their saved final duals belong to a different pool re-solve and are not substituted. The augmented CSV labels this distinction.

All bounds are **numerical certificates in the pinned event-route model**, relying on the successful floating-point LP solve and shortest-path arithmetic. A one-objective-unit guard is deducted; this is a numerical precaution, not a proved universal floating-point error bound. No exact-arithmetic certificate is claimed. Tiny solver dual-sign/feasibility residuals are retained in sources.json. An independently reproducible rational certificate would require retaining/repricing the matching dual vectors (15 are absent), or verified arithmetic. The integer rounding thresholds have large slack (see the unrounded fleet lower bound), so these are not borderline rounding decisions.

## Pricing scope

Pinned commit a0e0bb7681c8451e3cbbbfa06aef390026d9af4b:
- `src/event_pricer_network.py`, `sink_predecessor_route_batch`: computes the global shortest path first; `routes=[best]`; extra-column selection never displaces it.
- `src/exact_pricer_expanded.py`, CG loop: prices the current RMP duals and stores `lp_obj` and `min_rc` together in the same history/final record.
- `src/master_lp_gurobi.py`: nonnegative route variables without finite upper bounds; successful LP values and coverage duals are returned. Artificial mass is zero in all102 priced records.
- Ordinary chain runs use combined cost with zero fleet-cap dual, no diversification, and the audited uniform event DAG. This conclusion is not transferred to capacity-pricing pilots, arbitrary older pricing implementations, or the unrestricted continuous physical model.

## k=32

| Chain | Fleet lower bound | Original1h incumbent | Later evidence needed for optimality |
|---|---:|---:|---|
|1|32|35|Existing longer run still underway at audit start|
|2|32|34|Later32-bus result reaches this numerical model bound|
|3|32|33|One-bus gap remains|
|4|31|34|Later32-bus result leaves one-bus model gap|
|5|31|37|Later32-bus result leaves one-bus model gap|
|6|32|33|Later32-bus result reaches this numerical model bound|

Thus the review's suggestion that all four chains1/2/3/6 become proved optimal at32 is **refuted as a current-result claim**: chains1 and3 still need a32-bus incumbent. Fleet certificates remain separate from exact-once service assignment, shared charger capacity, and charging-cost optimality.

Artifacts: `sources.json` holds raw scalar extracts, hashes, physics and matching iteration records; `collect.py` fetches them read-only; `all_chain_extension_results_with_bounds.csv` preserves original columns and adds the bounds; `summary.json` summarizes the calculation. Historical source CSV is unchanged.
