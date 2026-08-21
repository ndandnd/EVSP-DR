# Provisional findings: cursor/branch-and-price-experiment-2969-1451

These labels are local and non-authoritative. A curator assigns project IDs.

## LOCAL-1

**Claim:** `run_exact_pool_mip.py --progress-dir` writes observational
convergence snapshots only. It does not serialize or restore the Gurobi search
tree, node queue, bound state, LP basis, cuts, or pseudocosts. It does not load
an incumbent from a prior progress directory into a relaunched solve.

**Evidence:** `analysis/mip_progress_audit_20260821/REPORT.md`

**Producing commit:** `3d5e11fa4fd4eaafdef678117447d4bbe109dbfb`

## LOCAL-2

**Claim:** Across 240 deterministic tiny SOC-time instances, exhaustive route
enumeration, corrected branch-and-price, and direct arc-flow agreed on every LP
bound and every integer fleet. Exact CG also matched every LP bound, but its
final-pool integer MIP overestimated 34 fleets (33 by one bus, one by two).
After shrinking, all 34 trip/station-irreducible reproducers retain a one-bus
pool excess; the smallest has 7 trips and one station.

**Evidence:** `analysis/tiny_differential_20260821/REPORT.md`

**Producing commit:** `1c6258b806970b84d905dbad35ac5d7eca019ac1`

## LOCAL-3

**Claim:** The 57-minute trip-gap cap, reserve-SOC floor, and prohibition on
station-to-station arcs are each binding. Targeted mutations changed the
exhaustive optimum from 2 buses to 1, and all four methods agreed before and
after each mutation.

**Evidence:** `analysis/tiny_differential_20260821/summary.json`

**Producing commit:** `1c6258b806970b84d905dbad35ac5d7eca019ac1`

## LOCAL-4

**Claim:** A relaunched toy solve re-explored nodes already searched by the
interrupted process. Two independently signalled launches explored 20 and 19
nodes respectively and ended with the same fleet incumbent 35 and bound 6.
Reusing the original progress directory was rejected; the second solve started
with a new directory and node count zero.

**Evidence:** `analysis/mip_progress_audit_20260821/REPORT.md`

**Producing commit:** `3d5e11fa4fd4eaafdef678117447d4bbe109dbfb`

## LOCAL-5

**Claim:** Large pool MIPs still benefit from a protected scaglione partition
when a high-quality start is needed. Progress snapshots do not remove that
need. Any repeated MIP start on relaunch is reconstructed from the pool or an
explicit partition, independently of prior progress metadata.

**Evidence:** `analysis/mip_progress_audit_20260821/REPORT.md`

**Producing commit:** `3d5e11fa4fd4eaafdef678117447d4bbe109dbfb`
