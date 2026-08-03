# Peel-and-price: instance-level dive-and-price

`src/run_peel_and_price.py` builds an integral cover from genuinely DP-generated
columns by exploiting an asymmetry the Goal-1 audits established: pricing
against uniform/artificial duals reliably produces deep routes (NO_CHEAT
recovered 37/42 exact single duties in 60 s; split-GREEDY duals recovered 0/20),
while pricing against sparse degenerate seed duals starves. Each peel stage
therefore re-creates the favorable landscape:

1. run a short CG burst on the current residual instance (artificial-only start);
2. solve the restricted master LP over the returned pool (SciPy/HiGHS);
3. fix ONE route, delete its trips, write the residual instance;
4. recurse until every trip is fixed.

This mirrors truncated column generation with node removal (de Vos,
van Lieshout & Dollevoet, Transportation Science; arXiv:2207.13734), which was
both faster and better than single-shot price-and-branch on 512-trip instances.

## Route-fixing rules (`--pick`)

- `lookahead` (default): fix the route minimizing `1 + LB(residual)`, where
  `LB` is the residual's peak trip concurrency — a valid fractional fleet lower
  bound. This is the anti-straddling rule: a route mixing trips from what
  should be two buses leaves the residual LB unchanged and scores worse than a
  duty-shaped route. Ties break toward higher LP weight, then depth.
- `lp`: fix the maximum-LP-weight column (classic diving).
- `max_trips`: naive depth pick (control).

## Two-duty validation (2026-08-03, M5 Pro, flat prices)

Instance `data/Practice_Custom_TwoDuty_13301_13302.csv` (86 trips, duties
13301+13302, certified fleet LB 2):

- Reference: MATCHING + CG reaches LP route weight **2.0** with zero
  artificials at objective 200,151.01 — **56.78 cheaper than GIRO's own two
  duties** (200,207.79; the historical duties price at +56.78 at our optimum,
  so exact-incidence rediscovery is the wrong success criterion).
- Peel with `--pick lp` and `--pick max_trips` (60 s pricing, 0.04 h stages):
  complete integral partitions in 5–7 minutes wall, but **3 buses** — both
  fixed a 38-trip route straddling the two duties at stage 1.
- Peel with `--pick lookahead` at the same 0.04 h stage budget: ALSO fixed the
  38-trip straddler — the stage-1 pool (≈2 minutes of CG) contained **no route
  whose removal reduces the residual peak-concurrency LB below 2**, so all
  candidates tied and the tie-break reproduced the LP pick. Removing an exact
  duty leaves residual LB 1 by construction, so the lookahead rule works
  exactly when the pool contains a duty-shaped route.

Interpretation: the DP finds deep routes under stage-local uniform duals; the
fixing rule converts depth into fleet-minimal covers only when the stage pool
contains duty-shaped candidates. **Stage-1 pricing budget is therefore the
primary knob** — the Unicorn sweep scans it (0.04 h vs 0.15 h) across 20 duty
pairs and both pick rules.

## Local usage

```bash
PYTHON=/Users/nathan.cho/.pyenv/versions/3.10.6/bin/python   # pandas + SciPy
cd src
"$PYTHON" -u run_peel_and_price.py \
  --csv Practice_Custom_TwoDuty_13301_13302.csv \
  --initializer nocheat --pick lookahead \
  --stage-active-hours 0.04 --pricing-seconds 60 \
  --run-tag peel_twoduty
```

Residual instances land in `data/peel_tmp/` (do not commit); per-stage runner
logs, pools, and `peel_summary.json` land under `src/results/peel_<tag>_*/`.

## Suggested Unicorn sweeps

Peel jobs are single-core-friendly and embarrassingly parallel across
instances — good use of idle August capacity:

1. **Duty-pair ladder**: sample ~20 pairs from the 42 single duties, build
   two-duty instances (concatenate the tracked single-duty CSVs, re-sort by
   start time, re-index `count_trip_id`), run `--pick lookahead` and `--pick lp`
   at 0.04 h stages. Metric: buses vs the pair LB of 2 (or the pair's
   peak-concurrency LB where duties overlap less).
2. **Scaling rungs**: 5-duty and 10-duty unions, stage budget 0.1 h,
   60–120 s pricing; compare buses against `audit_structural_route_bound.py`
   LB and the MATCHING baseline.
3. **Random suite**: 10/15-bus `random_goal1_instances` replicates; peel
   (nocheat) vs MATCHING-seeded portfolio CG — does peel close the 17→15 gap
   on 15-r04 that single-queue CG could not?
4. Always record `peel_summary.json` plus per-stage pools; the union of stage
   pools is itself a candidate column portfolio for one final master+MIP.

## Caveats

- Peeling is a greedy dive without backtracking: it certifies nothing and can
  strand awkward trip subsets in late stages. Treat results as upper bounds
  and construction heuristics, not optimality claims.
- The residual peak-concurrency lookahead is a lower-bound heuristic; the
  reachability-antichain bound is tighter when concurrency underestimates.
- Peel complements — does not replace — the integrated pricing portfolio and
  the planned exhaustion-certificate work (tight completion bounds or an
  SOC×time-expanded exact pricer): those are still required for trusting CG
  at stalled duals, per `GOAL1_STATUS.md`.
