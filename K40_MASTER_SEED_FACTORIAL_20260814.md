# Controlled k40 master-sense x initialization diagnostic

## Purpose

This is a diagnosis campaign, not a paper-results campaign. It tests two
leading explanations for the difference between the successful August
`exact_big` k40 trajectory and the later `cg-bigtar` trajectory:

1. set covering (`coverage >= 1`) versus strict set partitioning
   (`coverage == 1`); and
2. Big-M artificial-only initialization versus a complete real singleton
   partition.

The expanded SOC-by-time network, route pricing, instance, tariff, physics,
grid, batch size, and runtime budget are identical across all four arms. This
is a controlled current-code factorial, not a complete reproduction of every
historical code difference.

## Frozen comparison cell

- Instance: `Practice_Custom_DutyUnion_k40_r2.csv`
- Deterministic generator: the archived `duty_unions_big` sequence, seed
  `20260803`, sizes `15,20,30,40`, six requested replicates per size
- Required instance SHA-256:
  `3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd`
- Trips: 947
- Tariff: `hourly_prices_flat.csv`
- Required tariff SHA-256:
  `1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200`
- Physics: 300 kWh, 300 kW, zero reserve
- Discretization: 15 kWh SOC, 10-minute time blocks
- Columns per pricing iteration: 30
- Primary comparison horizon: 24 hours
- Cumulative job wall: 25 hours, leaving time to publish and validate the
  immutable 24-hour snapshot before the 26-hour Slurm allocation ends
- Immutable snapshots: 1, 3, 6, 12, 22, and 24 hours; 22 hours matches the
  historical `exact_big` application wall

The generator's random state advances across requested sizes. The prep job
therefore reproduces the full archived sequence and rejects any input whose
bytes do not match the frozen hash above. The first campaign launched from
commit `eb85ca0` generated these exact 947-trip bytes under the temporary name
`k40_r1`; its results remain valid by hash and must be compared with the
historical **k40_r2** result (LP route weight 39.252981 after about 22 hours),
not the historical k40_r1 result. Later launches use the correct r2 name.

## Arms and Slurm names

| Arm | Master | Initial pool | Job name |
|---|---|---|---|
| CA | cover | artificial-only | `K40-CA24` |
| CS | cover | singletons | `K40-CS24` |
| PA | partition | artificial-only | `K40-PA24` |
| PS | partition | singletons | `K40-PS24` |

The current production default remains **partition + singletons**. The new
switches exist only so the historical behavior can be diagnosed without
silently changing operational defaults.

All four outputs are fresh and record `initial_pool` in every durable status.
Do not use this branch to resume a genuinely pre-`0722e2e` artificial-only
pool whose legacy status lacks that field: the older migration schema cannot
attest its initializer. That compatibility case is separate from this fresh
campaign.

## Cluster behavior

`launch_k40_factorial.sh` must run from a detached, tracked-clean checkout. It
submits a small `K40-PREP` compute job, which regenerates and hashes the input,
then starts the four arms only after prep succeeds. Each arm has a distinct
result stem and journal, uses `--resume`, requeues after preemption, and has a
25-hour cumulative recorded-runtime limit. Timed snapshots are taken from the
last completed durable pool/LP at each boundary; a master solve may be cut at a
boundary to make the snapshot publishable. The scientific comparison uses the
immutable 24-hour snapshot, not the later graceful-stop status.

`wall_s` is the cumulative runtime recorded by durable statuses. A hard process
or node loss can omit work since the last durable iteration, such as an
in-flight master solve or network rebuild. Preserve Slurm elapsed-time records
alongside the results and disclose that limitation; requeue/resume is not a
perfect wall-clock stopwatch.

The four arms may land on different CPU models in Unicorn's heterogeneous
default partition. Each allocation is recorded in an arm-specific TSV. Compare
both fixed recorded time and iteration trajectories; if the treatment effects
are modest, repeat the screen or pin a comparable CPU class before attributing
the difference to a model factor.

Launch:

```bash
bash src/launch_k40_factorial.sh
```

The launcher prints the campaign identifier. Monitor the newest campaign with:

```bash
bash src/monitor_k40_factorial.sh
```

An explicit campaign identifier may be supplied as the first argument when
more than one campaign exists.

## Interpretation

- CA reproducing the historical rapid decline would be evidence against a
  broad regression in the current expanded-network pricer; it would not prove
  historical code equivalence.
- CA outperforming CS estimates the effect of singleton initialization under
  this frozen current-code cell.
- CA/CS outperforming PA/PS estimates the effect of exact-partition
  complementarity under this frozen current-code cell.
- PA outperforming PS would justify an artificial-only Phase I before adding
  a validated exact partition for the final MIP.
- Cover results remain discovery diagnostics. They are not valid exact-trip
  schedules when selected routes overcover trips.

Do not expand this to all four k40 variants or all tariffs until this one-cell
factorial identifies the mechanism. A single peak-tariff replication is the
next step only after the flat-price verdict.
