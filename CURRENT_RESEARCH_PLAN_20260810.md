# EVSP-DR corrected research status and plan

This file supersedes the top-line claims in `HANDOFF_20260810.md`.

## What has been achieved

- The elementary heuristic pricing DP has been replaced by an SOC-by-time
  expanded-network exact pricer. On its discretized route space it can provide
  a real reduced-cost certificate rather than a timeout heuristic.
- Pair and small-union Partille instances provide strong exact-pricing and LP
  validation evidence.
- Delayed charging is native: a bus may wait for a later time block before
  beginning a contiguous charging run.
- Exact-CG columns are journaled; timed pools are published as immutable,
  recoverable status/journal pairs. A hard preemption can repair only a
  truncated final journal/trajectory record, while interior corruption is
  refused.
- Strict binary partition MIPs exist, and injected MATCHING/GIRO trip sequences
  are re-realized and replay-validated under the target physics.
- On constructed k=40 Partille unions, validated 39-bus schedules have been
  found where the selected GIRO seed duties number 40.
- A concrete large-master defect was confirmed in our post-processing:
  individually small primal values and duals were zeroed before feasibility
  was checked. The raw-primal fix and regression test are in place; the new
  raw-pool audit will determine whether this fully explains the observed
  large-run failures.

## What those results do not prove

- The 39-bus solutions are optimal over their finite augmented pools, not over
  every feasible route. k=40 pricing is not certified and the structural fleet
  interval remains 35--39.
- The constructed k=40 unions are not a verified single Partille service day.
  Weekday-variant mapping still needs to be confirmed.
- Existing charging results assume no shared station-time charger limit and are
  therefore optimistic unlimited-capacity results; their costs are lower
  bounds on a corresponding capacity-constrained model.
- Repricing GIRO under a peak tariff measures exposure, not savings. The earlier
  0.07% comparison used different tariffs on the two sides and is withdrawn.

## Engineering campaign now ready to run

1. Validate raw large-master residuals on the latest terminal immutable pool
   from each selected k30/k40 run. This is a representative gate, not an audit
   of every historical snapshot.
2. Preserve the last valid LP across solve failures and Slurm requeues, with an
   explicit source label, cumulative iteration/time accounting, and no false
   certificate.
3. Measure integer fleet quality versus both CG time and MIP time. Fleet search
   receives the full MIP budget unless it proves the fleet early; only then is
   remaining time used to optimize charging cost.
4. Continue six no-stall controls from immutable six-hour snapshots to 72
   total hours. Make a paired comparison with the old marginal-return stopping
   trajectory only when its terminal status, journal, and iteration log were
   successfully archived; otherwise report the continuation forward-only.
5. Archive every selected source status/journal pair, available terminal
   baseline trajectory, generated big-instance inputs, runtime commit, and
   result evidence with hashes. All validation, hashing, and compression runs
   in Slurm compute jobs, not on the Unicorn login node.
6. Analyze CG time using each snapshot's recorded actual cumulative wall time,
   not only its nominal `m60`/`m180`/etc. publication label. MIP-curve cells
   are comparable only when the complete common GIRO seed was re-realized.

## Work required before the demand-response experiment

1. Define a terminal-energy policy. Every modeled bus starts full but currently
   may finish at reserve, so raw cost comparisons can spend free initial energy
   or defer replenishment beyond the horizon.
2. Decide whether the $5 charge-start term is a real operating cost. It is too
   large to call a harmless tie-break and is currently counted differently in
   fixed-route and joint-routing models.
3. Use station-specific charger power or explicitly retain uniform 300 kW as a
   synthetic convention. The 240 kWh battery is data-supported; uniform 220 kW
   Partille charging is not.
4. Report electricity, kWh, peak-window kWh, deadhead, physical charge visits,
   initial/terminal SOC, fleet, and charger concurrency separately.
5. Re-realize both fixed GIRO sequences and final joint-selected sequences with
   the same continuous postprocessor, then cross-price every schedule under
   every tariff.
6. Hold fleet and terminal treatment fixed when attributing charging-only and
   incremental rerouting savings.
7. Use all four legitimate 40-duty weekday-variant combinations until Transdev
   identifies their actual weekdays; do not use the mixed 42-duty file as one
   operating day.

## Work deliberately skipped for now

- More heuristic DP, peel, or greedy-dive campaigns.
- Blanket reruns of completed small matrices.
- Treating 240/220/20% as the headline Partille physics.
- Frölunda replication before the Partille model and experiment are frozen.
- A full capacity-aware pricing master before installed charger counts are
  confirmed. A post-hoc concurrency audit is mandatory in the meantime.

## Intended final experiment

For each valid Partille duty set and each tariff:

- **Observed**: GIRO schedule as recorded, cross-priced.
- **Fixed sequence**: GIRO trip sequences retained, charging re-optimized.
- **Joint**: routing and charging both optimized, then normalized through the
  same continuous charging postprocessor.

The charging-timing value is the peak-tariff cost of the flat-policy schedule
minus the peak-tariff cost of the peak-aware schedule. The incremental routing
value is the corresponding joint saving minus the fixed-sequence saving.
