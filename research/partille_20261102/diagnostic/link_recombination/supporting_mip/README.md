# Exact augmented-pool check of archive-edge recombination

One deterministic C2 k8 pilot uses only observed archive edges to build eight new passenger sequences, retimes those sequences, independently replays them, and supplies their eight columns as a constructive MIP start. It reads no saved duty grouping or GIRO-informed witness. This tests recombination of an existing priced archive; it is not an arc-fixing CG implementation or a fresh-from-scratch speed comparison.

The exact model has 39,390 original columns plus eight new columns, 173 exact trip equalities, 3,640 finite station-minute plug rows, a fixed fleet of eight and the original zero aggregate terminal-energy floor. It minimizes charging-related cost at fixed fleet. Physics: 240 kWh initially full, zero reserve, 2.5 kWh SOC/5-minute event charging grid, 240 kW opportunity stations, PARX 60 kW, original finite opportunity plugs/unlimited depot plugs, flat tariff and fee 5. `physics.coverage` preserves the historical source metadata; `actual_master_trip_rows` records this experiment's exact partition rows.

Gurobi 12.0.1 on macOS ARM with two threads accepted the start. It stopped at the 120-second time limit after 120.0533 solver seconds with that one solution: eight buses, grid charging cost 361.752 and finite augmented-pool charging bound 238.368953 (34.1071% charging gap). The weighted values add 800,000; they do not change the charging gap or prove a full-model bound. No improved solution was found. Independent final replay serves all 173 trips exactly once, with zero plug-overload minutes and 14.631001 kWh terminal energy; continuous repricing is 343.233143. Eight simultaneous trips independently prove the fleet floor under this model. No charging-cost optimality or operational portability is claimed.

Measured clocks: upstream archive-edge matching 1.657660 s, fixed-sequence retiming 5.096592 s; this driver's initial independent replay 1.419864 s, pool load/incidence 3.330757 s, model build 1.999330 s, optimization 120.053088 s, final replay 1.263081 s, total driver 129.195592 s. The measured incremental sum of matching, retiming and driver is 135.949844 s, excluding invocation gaps. It reuses the archive. Historical owner-summary clocks are graph 7,965.730486 s, root CG 8,465.745420 s (LP and pricing components included therein), a 5,400 s requested dive that timed out, and fleet MIP 1,800.318659 s. Actual dive/charging-MIP elapsed times are absent. Do not treat the incremental sum as cold time or compute a fair end-to-end speed ratio from these incomplete ancestor clocks.

`aggregate.json` contains sanitized source/artifact hashes, exact status, objective/bound scopes and separate clocks. `solve_arc_pool.py` is the exact executed driver. Inputs, full schedules, native logs, replay receipts and pool references live under the sibling private directory and are excluded from publication. The project diagnostic `validate_selection.py` supplies independent replay using arm `augmented`, which checks no historical grouping.

Reproduction requires the authenticated v15 source tree and authorized input files. Set SOLVER, DATA, INSTANCE, OWNER_SUMMARY, POOL, ROUTES, PATHS, RETIME_RECEIPT, LINK_RECEIPT, VALIDATOR and PRIVATE to those paths. The driver verifies source/input hashes and path correspondence, independently checks the frozen start and independently checks the final MIP selection. It captures start fee 5 through argv before importing solver modules.

```sh
PYTHONPATH="$SOLVER/src" PYTHONDONTWRITEBYTECODE=1 python3 solve_arc_pool.py \
  --owner-summary "$OWNER_SUMMARY" --pool "$POOL" --routes "$ROUTES" \
  --paths "$PATHS" --retime-receipt "$RETIME_RECEIPT" --link-receipt "$LINK_RECEIPT" \
  --data "$DATA" --instance "$INSTANCE" --validator "$VALIDATOR" --out "$PRIVATE" --start-fee 5
```

The exact driver writes a private aggregate. Packaging adds only start-acceptance, hardware metadata and hashes of private log/replay artifacts to the public aggregate; these additions do not alter solver or physical results. Native routes/logs remain private.
