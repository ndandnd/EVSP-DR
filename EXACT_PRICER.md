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
process, no Gurobi. Current runs seed a real depot-trip-depot singleton for
every directly feasible trip and use a set-partitioning LP by default. Use
`--master-sense cover` only to reproduce the legacy covering campaigns.

## Two-duty results (86 trips, duties 13301+13302, M5 Pro, flat prices)

These historical numbers used the legacy set-covering master. They remain
valid covering-LP results but are not strict integer-partition certificates.

| grid | result | wall |
|---|---|---|
| 15 kWh / 10 min | **certified** rc-optimal: weight 2.3065, obj 230,875.92 (1,357 iters, 1 col/iter) | 89 s |
| 15 kWh / 10 min | same certified optimum via 30-col batches (584 iters) — independent consistency check | 106 s |
| 5 kWh / 5 min | **CERTIFIED rc-optimal: weight 2.000000, obj 200,192.59, zero artificials** (1,160 iters, 24,091 columns, min_rc = -3.5e-10) — beats GIRO's own duties (200,207.79) by 15.20; 0.021% above the runner-model value 200,151 (different charging discretizations) | 871 s |

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
  --soc-step 5 --block-min 5 --max-iters 4000 \
  --master-sense partition
```

Instances under `data/duty_pairs/` work directly (`--csv duty_pairs/<name>`).
Grid guidance: SOC step must not exceed the per-block charge gain
(300 kW x block/60), or charging rounds to zero; 5 kWh / 5 min reaches
runner-comparable optima on pair instances, 15 kWh / 10 min certifies fastest.

## Strict pool MIP on Unicorn

Use the tracked Bash worker, not an `sbatch --wrap` command. Slurm executes
wrapped command strings through `/bin/sh`, which cannot run this project's
Bash/conda setup reliably. The worker checks the frozen column journal before
importing Gurobi and writes a job-specific result beside the snapshot.

Legacy exact-CG snapshots contain priced routes but no guaranteed integer
partition. Prepare a non-destructive copy with same-grid singleton seeds first:

```bash
python -u src/prepare_exact_pool_mip.py \
  --result /absolute/path/to/INSTANCE.snapshot.json
```

This writes adjacent `*.partition_ready.snapshot.json` and
`*.partition_ready.columns.jsonl` files while leaving the frozen originals
unchanged. Validation of the prepared copy must report that strict partition
feasibility is guaranteed by one singleton per trip.

From the repository root, use the validated launcher. It is a dry run unless
`--submit` is present, rejects placeholder or missing paths before allocating
a node, writes logs below ignored `src/logs/`, and keeps non-resumable Gurobi
MIPs on the Scaglione partition:

```bash
python src/cluster_campaign.py mip \
  --result src/results/exact_big/INSTANCE.partition_ready.snapshot.json \
  --minutes 120

# Submit only after the printed preflight and Slurm command look correct.
python src/cluster_campaign.py mip \
  --result src/results/exact_big/INSTANCE.partition_ready.snapshot.json \
  --minutes 120 --submit
```

The launcher pads the allocation by ten minutes, gives Gurobi the requested
minutes, uses eight threads, enforces exact partitioning, and passes
`--partition=scaglione --no-requeue`. A Gurobi requeue would restart from zero;
it cannot preserve the branch-and-bound tree. Add `--cover` only for an
explicitly labeled covering sensitivity run. Only immutable `*.snapshot.json`
inputs are accepted. On submission, the launcher copies the snapshot and its
column journal into a uniquely reserved campaign directory, validates the copy,
and records source and staged SHA-256 hashes in `submission.json`. The queued
MIP therefore solves the frozen campaign copy rather than a journal that can
continue growing.

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
