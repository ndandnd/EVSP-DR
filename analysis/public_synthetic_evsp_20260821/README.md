# Public synthetic E-VSP structural family

This family is independent of the proprietary Partille/GIRO exports. Never
merge its instances, pools, or result rows with duty-union results.

## Identity

- Generator: `src/generate_public_synthetic_evsp.py`
- Generator family: `evsp-dr-public-structural-v1`
- Seed: **20260821**
- Tracked input:
  `data/public_synthetic/seed_20260821/public_structural_s20260821.json`
- Tracked controlled pool:
  `data/public_synthetic/seed_20260821/public_structural_s20260821.pair_pool.json`
- Canonical generated-problem SHA256:
  `852c39743739e0ae3b435a50e4ed235c74603299687db9e2421b14e070dad3a9`

## Feature summary

| feature | value |
|---|---:|
| trips | 6 |
| timetable span | 90 minutes |
| simultaneous-trip lower bound | 2 buses |
| battery | 60 kWh |
| charging | constant 60 kW |
| trip energy | 30 kWh |
| charging station | one unlimited-capacity hub |
| deadhead time/energy | zero |

There are three waves of two simultaneous 20-minute trips. Consecutive waves
are separated by 15 minutes. A two-bus solution requires each bus to serve one
trip per wave and gain 15 kWh in each layover.

- At 5-minute resolution, each layover contains three charge blocks, so two
  buses are feasible.
- At 10-minute resolution, each layover contains only one complete block, so
  the two-bus routes receive 20 kWh rather than the required 30 kWh. Three
  buses are necessary.

This isolates a one-bus time-discretization penalty without using private data.

## Executed results

| arm | grid | fleet LP | integer fleet | gap |
|---|---|---:|---:|---:|
| full model, fine | 5 kWh / 5 min | **2** | **2, proven** | reference |
| full model, coarse | 5 kWh / 10 min | **3** | **3, proven** | +1 discretization |
| pair-limited pool, fine | 5 kWh / 5 min | — | **3, pool-optimal** | +1 pool composition |

Both full-model integer solutions used all-arc integrality and passed exact
coverage and physical replay.

The pair-limited pool contains every singleton and every temporally feasible
two-trip route, but deliberately excludes three-trip routes. Its three-bus
optimum versus the full fine-grid model's two buses is a controlled
pool-composition example. It is **not** labelled RAW CG evidence and does not
claim that column generation naturally produces this particular pool.

The same six public trips therefore reproduce all three qualitative findings:

1. changing only time resolution changes the feasible route space;
2. coarse discretization costs one whole bus;
3. a restricted, individually feasible column pool can cost another bus even
   when the full model has a better partition.

Machine-readable rows are in `results.csv`.

## Reproduction

```bash
python3 src/generate_public_synthetic_evsp.py \
  --seed 20260821 \
  --output-dir /tmp/public_synthetic_20260821

python3 src/run_public_synthetic_evsp.py \
  --problem /tmp/public_synthetic_20260821/public_structural_s20260821.json \
  --pool /tmp/public_synthetic_20260821/public_structural_s20260821.pair_pool.json \
  --out /tmp/public_synthetic_20260821/result.json
```
