# Exact CG via the SOC x time expanded pricing network

`src/exact_pricer_expanded.py` implements de Vos/van Kooten Niekerk-style
pricing: discretized battery state lives in the network nodes — (trip,
SOC-at-trip-start) and (station, time-block, SOC-before-charge) — so each
pricing call is a plain shortest path in a static DAG, processed once in
topological order. No labels, no dominance, no queue policies, no timeouts.
Termination at min reduced cost >= -eps is a genuine rc-optimality
CERTIFICATE over the expanded route space. Charging may begin at any block
after arrival, so delayed-start (price-responsive) charging is native — the
restricted labeling DP cannot express it.

It reuses `audit_giro_known_columns.build_problem()` (the standalone issue20
restricted-graph loader: 57-minute trip gaps, ref-substituted deadheads,
full-horizon station waiting) and the SciPy/HiGHS restricted master. One
process, no Gurobi.

## Two-duty results (86 trips, duties 13301+13302, M5 Pro, flat prices)

| grid | result | wall |
|---|---|---|
| 15 kWh / 10 min | **certified** rc-optimal: weight 2.3065, obj 230,875.92 (1,357 iters, 1 col/iter) | 89 s |
| 15 kWh / 10 min | same certified optimum via 30-col batches (584 iters) — independent consistency check | 106 s |
| 5 kWh / 5 min | **route weight 2.0000, obj 200,199.63, zero artificials** — cheaper than GIRO's own duties (200,207.79), 0.02% above the runner-model optimum 200,151 | minutes |

The 15 kWh certified optimum exceeds 2.0 purely through conservative SOC
flooring at every hop (phantom energy loss over 36-50-trip duties) — the
discretization interplay documented by de Vos et al. (5.4.2); finer grids
close it. Contrast with the heuristic DP: 60-3,000 s per non-exhaustive,
uncertifiable pricing call vs ~0.05-0.5 s per exact iteration here.

## Usage

```bash
cd src
python exact_pricer_expanded.py \
  --csv Practice_Custom_TwoDuty_13301_13302.csv \
  --prices_csv hourly_prices_flat.csv \
  --soc-step 5 --block-min 5 --max-iters 4000
```

Instances under `data/duty_pairs/` work directly (`--csv duty_pairs/<name>`).
Grid guidance: SOC step must not exceed the per-block charge gain
(300 kW x block/60), or charging rounds to zero; 5 kWh / 5 min reaches
runner-comparable optima on pair instances, 15 kWh / 10 min certifies fastest.

## Honest scope

- The certificate covers the expanded (conservatively rounded) route space:
  its optimum upper-bounds the continuous-model optimum. A de Vos-style
  optimistic "dual network" for true lower bounds is future work.
- Recharge count is not a node dimension: the $5 start fee discourages excess
  and violations of MAX_DAILY_RECHARGES are reported, not forbidden.
- Master robustness: degenerate pools can stall HiGHS dual simplex; the driver
  falls back to `highs-ipm` / `highs` automatically.
- Memory/build time grow roughly linearly in (levels x blocks); pair-scale
  instances build in seconds-to-minutes in pure Python. For 300+-trip
  instances, port the pass to numpy or trim arcs before scaling up.
