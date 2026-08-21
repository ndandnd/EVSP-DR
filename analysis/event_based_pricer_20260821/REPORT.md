# Event-based pricer — local gate report

Date: 2026-08-21. Branch: `cursor/event-based-pricer-2969`.
No cluster jobs were submitted. Machine-readable executed rows are in
`results.csv`.

## Outcome

The prototype passes G1 and G3, but fails G5 as written. Per the work order's
stop rule, the k3/k5 table was not run.

### G1 — pass

With `--time-model` omitted, the actual k02_s1 regression is byte-identical to
the pre-change source:

- column journal bytes and ordering;
- ordered route hashes;
- reduced-cost strings;
- complete `iters.csv`;
- final certification and route weight.

Operational timing fields remain excluded.

### G3 — pass

At frozen 240 kWh / 240 kW / zero reserve, with a 2.5-kWh SOC lattice:

- k02_s2 event CG certified route weight **2.0000000000** in 117 iterations;
- duty 13413's previously blocked local transition `14→16` is represented;
- duty 13411 is represented as one complete event route, including all five
  previously recorded failures (`46→53` twice, `53→59` twice, `73→77`).

k02_s1 also certified at 2.0000000000. A 10-kWh event run on k02_s2 certified
at 2.0909090909, so event timing does not eliminate the need for a sufficiently
fine SOC axis; 2.5 kWh remains the tested event setting.

### G4 — pass on the motivating instance, with a scaling warning

For k02_s2:

| model | nodes | arcs |
|---|---:|---:|
| uniform 1 kWh / 5 min | 456,910 | 5,806,603 |
| event, 2.5-kWh SOC | 2,322 | 1,920,614 |

The event DAG has **99.49% fewer nodes** and **66.92% fewer arcs**. However,
direct event-transition factorization still scales poorly:

- k02_s3: 5,077 nodes, 7,810,190 arcs;
- k05_s2: 8,784 nodes, 22,161,911 arcs, 544.66 s build,
  26,814.64 MiB peak RSS.

Thus the stored DAG is smaller than uniform 1/5, but its Python construction
and arc representation are not yet a credible large-scale implementation.

## G5 — fail (stop condition)

The work order requires every event column to pass
`realize_expanded_path` **unchanged**. An exhaustive audit of the completed
k02_s2 event journal found:

| unchanged realization result | columns |
|---|---:|
| pass | 39 |
| reject: non-grid window at `2190L_0` | 1,041 |
| reject: non-grid window at `4808_0` | 515 |
| reject: non-grid window at `PARX_1` | 115 |
| **total** | **1,710** |

All inserted event columns already pass the unchanged restricted-graph,
timing, SOC, reserve, battery-cap, charger-power, tariff-block, exact-trip-
coverage, and continuous-block validators. The failure is specifically the
realization function's hard requirement that `cst`, `cet`, and duration be
multiples of one global integer `block_min`.

This is a specification conflict:

1. G3 succeeds because irregular arrival/deadline-induced windows are retained.
2. Snapping those windows to the uniform lattice reintroduces the restriction
   the event model exists to remove.
3. Extending `realize_expanded_path` with an event-mode branch would resolve the
   conflict, but would violate the current gate's word **unchanged**.

No exact event-route-space claim should be published until the operator chooses
between:

- preserving G5 literally and accepting uniform-aligned emission (which may
  lose G3), or
- versioning the realization contract for event windows while preserving all
  physical constraints.

## Additional executed validation

- reviewed continuous fixed-duty optimizer: 15 tests passed after retaining
  solver-tolerance micro-energy in contiguous replay segments;
- event network and known-transition tests: 8 passed;
- default k2 bit-identity gate: passed;
- no cluster or branch-and-price work was performed.
