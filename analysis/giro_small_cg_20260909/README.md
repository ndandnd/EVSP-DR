# Partille nonlinear-physics k2/k3 CG prototype

This study compares covering and partitioning on eight deterministic Partille
cohorts under the documented single-vehicle battery and charging physics.

Each RAW arm begins from singleton routes. Covering and partitioning run their
own trip-dual weighted pricing loops. Their columns are then united, and every
final comparison uses that identical frozen union pool:

- covering versus partitioning;
- one-minute charger-capacity rows versus a no-capacity relaxation;
- restricted-pool LP and binary MIP.

Pricing first uses the capacity-constrained restricted master. If the best
trip-dual route is already present because capacity duals were omitted, the arm
switches to a no-capacity master for pool enrichment rather than declaring
convergence. Every such switch and every wall, label, iteration, or duplicate
stop is recorded.

## Reporting limits

No LP value is a full-model lower bound. Pricing omits charger-capacity duals,
has explicit wall/label guards, and uses static symmetric reference deadheads.
Charger occupancy is conservatively rounded to one-minute rows and does not
model `JON_A`/`2190L` platform blocking or `4808` FIFO movement. A covering
incumbent may assign duplicate passenger trips; the report describes a possible
nonrevenue reassignment but does not certify that repair.

The weighted pricing DP itself is materially different from the earlier greedy
k2 peel: labels retain a Pareto frontier in collected trip-dual reward and SOC.
It labels with the monotone hold-until-departure charging policy, then adds an
early-disconnect replay of the selected trip sequence when feasible so the
finite master receives a less capacity-intensive variant.

## Cohorts

| Cell | Duties | Trips |
|---|---|---:|
| `e1_short_k2` | 13408, 13401 | 23 |
| `e1_short_k3` | 13408, 13401, 13414 | 35 |
| `e1_long_k2` | 13409, 13407 | 34 |
| `e1_long_k3` | 13409, 13407, 13404 | 51 |
| `e2_short_k2` | 13323, 13311 | 22 |
| `e2_short_k3` | 13323, 13311, 13307 | 37 |
| `e2_long_k2` | 13303, 13302 | 104 |
| `e2_long_k3` | 13303, 13302, 13312 | 150 |

The manifest records immutable input hashes and verifies unique
`Ordered_Trip_ID` values. All cohorts keep one vehicle group and exclude
weekday variants of the same base duty.

## Command

```bash
python3 src/run_giro_small_cg.py \
  --instance analysis/giro_small_cg_20260909/cohorts/e2_short_k2.csv \
  --expected-instance-sha256 0847a1ec0c0f40593e765e3909b44edca3ee47e0f7468cfba692c075f92dc14f \
  --vehicle-profile 18E2 --seed-mode raw \
  --cg-wall-s 600 --pricing-wall-s 30 --pricing-label-limit 250000 \
  --cg-max-iters 200 --mip-wall-s 300 --relaxed-mip-wall-s 60 \
  --threads 4 --out result.json --pool-out pool.jsonl --log-dir logs
```

A post-fix local smoke on `e2_short_k2` used ten iterations per arm and a
shared53-route pool. Both constrained senses and both no-capacity controls
returned LP2 and proved MIP2. A15-second `e2_short_k3` smoke produced a
217-route pool; the covering MIP found4 while the partition MIP found11. Those
short-budget numbers are engineering checks, not research outcomes.
