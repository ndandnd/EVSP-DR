# Capacity CG pricing-timeout checkpoint fix

## Failure diagnosis

Commit `7d38efdd39857438c4a6e30b43b09e973ce51086` checks the
`--cg-wall-s` budget only at the top of the column-generation loop.  The next
exact capacity-pricing call has no deadline.  That call traverses the explicit
event graph and evaluates one-minute capacity-dual charging-window choices.
It can therefore run beyond both the eight-hour application budget and the
nine-hour Slurm allocation.

The failed-run Gurobi logs under
`/home/nc437/ladder-lite/capacity_speed_pilot_20260910_timeout6_rerun_7d38ef/logs`
show the restricted LP continuing to solve in 0.00--0.01 seconds with only
46--73 columns and 7,814--7,835 rows immediately before timeout.  The master
LP is not the long operation.  Because the old runner writes `pool.jsonl` only
after the CG loop exits, Slurm termination during pricing leaves no pool.

## Bounded behavior and scientific semantics

Exact event pricing now accepts an absolute monotonic deadline and checks it
through the explicit arc traversal, inside capacity-window selection, before
and after route reconstruction, and in the lazy traversal.  Reaching it raises
a dedicated `PricingDeadlineExceeded`; the CG runner records
`stop_reason=pricing_deadline`, `status=incomplete`,
`certified_rc_optimal=false`, and `terminal_exact_min_reduced_cost=null`.
No partial pricing traversal is interpreted as an optimal-pricing result.

The pool is atomically replaced after singleton seeding and after every
accepted negative-reduced-cost route.  A hard kill can lose the in-flight
pricing pass and at most the column that pass had not yet returned; it does not
lose earlier accepted columns.  Each route carries a checkpoint identity that
binds the implementation commit, model schema, master and pricing senses,
reduced-cost tolerance, source hashes, physics, capacity rows, and objective
constants.  Resume rejects missing or mismatched identities, duplicate routes,
invalid trip membership, non-finite costs, and routes that fail physical
replay before rebuilding the same restricted master.

`--resume` requires an existing checkpoint pool and a new status output path.
It restores routes in their saved order and continues iteration numbering.
It neither resumes nor certifies the interrupted shortest-path traversal.
The existing two-stage MIP code based on `9bf3f75` continues to classify a
trip-covering timed pool as usable but explicitly uncertified; pricing proof,
finite-pool MIP proof, and physical/capacity validation remain separate.

## Tests

Run from the detached code checkout:

```text
python3 -m unittest tests.test_capacity_speed_event_cg tests.test_event_pricer_network
```

The focused suite covers the real capacity-row master behavior, deadline
termination, interrupted atomic replacement preserving the previous pool,
checkpoint identity and physical-replay rejection, byte-for-byte equality of
resumed and uninterrupted synthetic pools, timed-pool classification, and the
two-stage MIP invariants.

## Five-case retry plan

Retry only original indices `5,9,11,13,15`.  Their old timeout directories
contain no pool, so the first run on the fixed immutable commit is fresh.  Do
not pass `--resume` on that attempt.  Keep `--cg-wall-s 28800`, the nine-hour
Slurm allocation, one CPU/thread, 24 GiB, `default_partition`, and
`--exclude=scaglione-compute-01`.  Update the launch-level commit, every selected
task commit, `--expected-commit`, code root, source-campaign provenance, and
wrapper hashes to the fixed detached clean commit.  Keep the MIP code root at
the independently frozen `9bf3f75` two-stage implementation.
Some generated instance/reference inputs are untracked in the source checkout;
copy only the required files into the new isolated code root and verify them
against the launch manifest hashes.  `--require-clean` intentionally checks
tracked files, while the wrapper must continue enforcing hashes for these
untracked inputs before invoking CG.

Use a unique attempt directory for each later resume.  Preserve the previous
attempt unchanged, atomically copy its `pool.jsonl` to the new attempt's pool
path (the repository's `durable_io.atomic_copy` provides the required
temp-file, fsync, and replace sequence), pass `--resume`, and give `--out` a
new nonexistent path.  The copy is required because CG atomically replaces
the pool it is extending.  A pool from `7d38efd`, including index 7's usable
35-column pool, cannot resume under the new code because it predates the bound
checkpoint identity.  Index 7 is already complete for the pilot and should
not be rerun.

After each CG task finishes, its own freeze/MIP job may consume that task's
hash-bound status and pool.  Use individual successful-task dependencies so
one slow or failed array element does not stall the other finite-pool MIPs.
An incomplete pool is eligible only if it covers every trip and is still
reported as an uncertified timed exact-event CG pool.
