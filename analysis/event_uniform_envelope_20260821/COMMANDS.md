# Executed command templates

All commands ran locally from repository root. No Slurm command ran.

## Certified combined-cost CG

```bash
python3.12 src/exact_pricer_expanded.py \
  --csv <instance> --prices_csv hourly_prices_flat.csv \
  --g-kwh 240 --charge-kw 240 --min-soc-frac 0 \
  --soc-step <step> --block-min <minutes> \
  --master-sense partition --initial-pool singletons \
  --max-iters 10000 --columns_per_iter 30 --checkpoint-every 25 \
  --wall-limit-s 21660 --out <status>
```

Event adds `--time-model event --event-arc-mode lazy` and uses 2.5/5.
Panel A continuations used `--resume`; 2/1 k05 cells received a final
86,400-second cumulative ceiling where the initial certification ceiling was
insufficient.

## Matched-wall snapshots

```bash
python3.12 src/freeze_exact_cg_at_wall.py \
  --result <uniform-status> \
  --budget-s <paired-event-wall-s> \
  --telemetry <uniform-phase-telemetry> \
  --out <matched-wall-status>
```

## Fleet LP phase 2

```bash
python3.12 src/certify_fleet_lp_bound.py \
  --result <certified-combined-cost-status> \
  --max-iters 10000 --wall-limit-s 21600 \
  --out <phase2-status>
```

Phase-2-added columns are separate evidence and are not included in the RAW
pool MIP.

## Timed RAW MIP

```bash
python3.12 src/run_exact_pool_mip_highs.py \
  --result <raw-status> --solver native \
  --timelimit 1800 --mipgap 0.0001 --threads 8 \
  --out <mip-result>
```

Native solver: HiGHS 1.15.1. Size-limited Gurobi failures and preliminary
SciPy/HiGHS runs are excluded from normalized rows.

## Target feasibility

```bash
python3.12 src/target_pool_feasibility.py \
  --result <raw-status> --target <fleet> \
  --timelimit <1800-or-7200> --threads 8 --seed 0 \
  --solver highs --out <target-result>
```

## Model integer witnesses

```bash
python3.12 src/audit_event_known_partition.py \
  --status <event-status> \
  --fleet-lower-bound <phase2-bound> --out <witness>

python3.12 src/audit_uniform_known_partition.py \
  --status <uniform-status> \
  --fleet-lower-bound <phase2-bound> --out <witness>

python3.12 src/arcflow_oracle.py \
  --csv <instance> --soc-step <step> --block-min <minutes> \
  --g-kwh 240 --charge-kw 240 --min-soc-frac 0 \
  --journal <journal> --journal-route-index 0 \
  --objective feasibility --integrality service \
  --fixed-fleet <ceil-phase2-bound> \
  --fleet-lower-bound <phase2-bound> \
  --time-limit-s 7200 --mip-rel-gap 0 --out <oracle>
```

## Normalization

```bash
python3.12 src/summarize_event_uniform_envelope.py \
  --plan analysis/event_uniform_envelope_20260821/plan.json \
  --execution-root /tmp/evsp-event-uniform-envelope-c08df1a5277b \
  --output-dir analysis/event_uniform_envelope_20260821
```
