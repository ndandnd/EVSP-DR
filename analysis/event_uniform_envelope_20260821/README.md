# Event versus uniform envelope experiment

This plan implements the approved two-panel comparison at frozen 240 kWh /
240 kW / zero reserve. It does not launch Slurm work.

## Pool identity

Every `I_pool` and `I_timed` value uses the RAW pool from the unchanged default
combined-cost column-generation path:

- singleton initial pool;
- partition master;
- `columns_per_iter=30`;
- no injected or known routes.

Lexicographic phase 2 starts from an immutable copy of that pool and writes any
additional fleet-pricing columns to a separate journal. Those added columns
certify `L_model`; they are deliberately excluded from the RAW pool used for
`I_pool` and `I_timed`. This keeps the pool-composition comparison aligned with
the existing uniform RAW experiments.

## Panel A

For each of nine instances and six representations:

1. run default combined-cost CG to certification;
2. certify the fleet-only LP over the complete discrete route space;
3. run the 1,800-second two-stage RAW pool MIP;
4. establish exact `I_pool` from `fleet_proven`, or by target-feasibility
   bracketing if the timed solve is unproven;
5. establish `I_model` by the LP/incumbent sandwich, falling back to all-arc
   arc-flow only when no physically validated witness closes the bound.

The representations are event 2.5 kWh with instance-induced times and the
uniform grids 10/10, 4/5, 2/5, 2/2, and 2/1.

## Panel B

The event source is its certified Panel A combined-cost pool. Each of the five
uniform representations is rerun with the paired event cell's observed
certification wall time as its compute budget. The runner receives an
additional 60 seconds solely for the exact pricer's serialization reserve; the
scientific budget remains the event `wall_s`.

All 54 arms receive the same 1,800-second, 8-thread, two-stage RAW MIP and an
industrial-target feasibility solve. The uniform envelope is the best
physically valid timed integer incumbent across all five uniform grids; every
underlying row remains published.

Licensed wording must report the exact lower/tied/higher fleet counts under
equal **per-arm** caps and the target-attainment counts. The uniform envelope
uses five separately budgeted arms, so no equal-total-compute or aggregate
schedule-quality claim is licensed. No claim that the event pool is
intrinsically better is licensed without Panel A's representation-relative
pool-gap decomposition.

## Provenance and censorship

Every row must retain:

- `source_cg_certified`;
- `source_cg_stop_reason`;
- `source_cg_iterations`;
- `source_cg_wall_s` and peak RSS;
- `optimality_scope`;
- `physical_witness_valid`.

An unproven timed incumbent is an upper bound, not an exact pool result.
Uncertified CG gives no model LP bound. Findings go only to
`records/inbox/cursor-event-based-pricer-2969.md`; the authoritative ledgers
are not modified by this branch.
