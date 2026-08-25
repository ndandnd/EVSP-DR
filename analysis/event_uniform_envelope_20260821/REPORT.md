# Event versus uniform envelope

Date: 2026-08-21. Plan:
`c08df1a5277b8409cbe6302439d1083bcf854c6f58db3d7cbbd73c24dd729de6`.
No cluster jobs were submitted.

## Answer

Under equal **per-arm** compute caps, event produces a lower fleet incumbent on
four instances, the same fleet on three, and a higher fleet on two:

| cell | industrial fleet | event timed fleet | uniform-envelope timed fleet | result |
|---|---:|---:|---:|---|
| k02_s1 | 2 | 2 | 3 | event better |
| k02_s2 | 2 | 2 | 2 | tie |
| k02_s3 | 2 | 2 | 2 | tie |
| k03_s1 | 3 | 3 | 4 | event better |
| k03_s2 | 3 | 3 | 4 | event better |
| k03_s3 | 3 | 3 | 3 | tie |
| k05_s1 | 5 | 5 | 7 | event better |
| k05_s2 | 5 | 12 | 6 | uniform better |
| k05_s3 | 5 | 15 | 12 | uniform better |

The event arm reaches the industrial fleet on **7/9** instances; the best of
five uniform grids reaches it on **3/9**. Event is better on four cells, tied on
three, and worse on two.

The exact licensed observation is the 4/3/2 comparison above and the
7/9-versus-3/9 industrial-target count. The uniform result is an envelope over
five separately budgeted arms, so this is not an equal-total-compute claim.
Fleet counts do not compare schedule cost. It also does **not** license “the
event pool is intrinsically better.”

## Matched-compute construction

Each cell uses frozen 240 kWh / 240 kW / zero reserve, flat tariff, singleton
initialization, partition master, and `columns_per_iter=30`.

The event arm is the certified 2.5-kWh event lattice. Every uniform grid in
10/10, 4/5, 2/5, 2/2, and 2/1 receives that cell’s observed event certification
wall budget. The published pool includes every iteration whose route insertion
and, when needed, journal fsync completed before the budget. Later iterations
are excluded.

Every arm then receives the same native HiGHS 1.15.1 two-stage RAW MIP:
1,800 seconds, eight explicitly configured threads, zero injected routes.
Target feasibility is reported separately.

The local Gurobi license rejected most pools for model size before search.
Those attempts are excluded. Preliminary SciPy/HiGHS rows are retained outside
the normalized tables; final rows all use native HiGHS.
Target-feasibility diagnostics use SciPy/HiGHS rather than the plan's Gurobi
backend. `execution_deviations.json` binds this substitution, the exact
pricer's 60-second serialization margin, two extended Panel A certification
runs, and stage-specific producer commits.

## Panel A — decomposition

`panel_a.csv` contains all 54 representation cells and the requested
`L_model`, `I_model`, `I_pool`, and `I_timed` fields.

- **54/54** fleet LP bounds are phase-2 certified.
- **54/54 event/uniform CG sources** used for Panel A are certified.
- **39/54** model integer optima are proven by an industrial witness,
  finite-pool witness, or arc-flow sandwich.
- **42/54** finite-pool integer optima are proven.
- Unclosed searches are intervals, never imputed point estimates.

For the event representation specifically:

- `L_model` reaches 2, 3, or 5 on all nine cells;
- the physically replayed industrial partition proves
  `I_model = industrial fleet` on **9/9**;
- `I_pool` is exact and equals the industrial fleet on all k2/k3 cells and
  k05_s1;
- k05_s2 remains `I_pool ∈ [6,12]`, with `I_timed = 12`;
- k05_s3 remains `I_pool ∈ [5,15]`, with `I_timed = 15`.

Thus the event representation itself has no measured representation or LP
integrality gap on these nine cells. The combined unresolved finite-pool and
search shortfalls at k05_s2/k05_s3 are not model infeasibility. Panel A does
not separately identify their composition and search components.

The uniform results are strongly non-monotone. Examples:

- k02_s1: model optimum 2 on 4/5 and 2/2, but certified-pool optima are 3;
- k02_s2: pool optima range from 2 to 5 across certified grids;
- k05_s2 equal-compute timed fleets range from 6 to 24.

## Provenance and physical validity

Every normalized Panel A and Panel B row records:

- source CG certification;
- source stop reason;
- source iterations, wall time, peak RSS, and pool columns;
- MIP optimality scope;
- physical-witness validity;
- SHA-256 identities for CG status, phase-2 certificate, MIP result, and
  target result where applicable.

All published timed incumbents pass strict final physical replay. A timed
incumbent with `fleet_proven=false` remains an upper bound. Uncertified
matched-wall CG rows make no LP-bound claim.

`panel_b.csv` preserves all 54 arms. `panel_b_envelope.csv` contains only the
per-instance envelope comparison. `evidence_manifest.csv` binds every consumed
artifact. Full journals remain external because the execution tree is 11 GiB;
their hashes are transitively bound by the MIP, phase-2, target, and snapshot
artifacts.

The inbox-only records policy applies to this experiment and policy-effective
commits. The long-lived branch contains pre-policy B0031-B0033/D0031-D0033
history; this experiment neither rewrote nor extended those ledgers.

## Validation

- focused envelope, identity, replay, feasibility, native-HiGHS, and arc-flow
  tests: **43 passed, 12 subtests**;
- Python source/test compilation: passed;
- full repository suite: **547 passed, 135 subtests**;
- two historical deterministic-artifact tests fail only because their embedded
  producer-code hashes detect the modified realization/MIP sources. Their
  scientific membership tables were not overwritten.
