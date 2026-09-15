# Longer search for chain 3, target 28

Question: does the saved 166,052-column pool support 28 buses? The original one-hour MIP found 29, with bound 28, after a 30-minute fleet stage. This is an open search gap, not a proved missing-column result. Its CG also hit four hours without a pricing certificate.

Keep input, source journal, 871d057e solver, covering, physics, objective, eight threads, default Gurobi Seed0, and the greedy initialization policy unchanged. Increase the fleet allowance from 1,800 to 10,800 seconds and total allowance from 3,600 to 12,600 seconds. This launches a new tree; it does not resume the earlier tree. Verify the ordered pool and realized initializer again when results arrive.

The preparation script checks all existing manifests for a longer MIP on the same input and journal before submission. Native source hashes and original solver settings are checked. Default partition, 8 CPUs, 24 GB, 4.5-hour allocation, requeue with private attempt directories, and exclusion of scaglione-compute-01. Inputs are complete, so there are no data dependencies. Held and other-project jobs are untouched.

Submitted job **222757**. Scheduler observation 2026-09-15T07:14:15.264518+00:00: **RUNNING**, reason None, node scaglione-cpu-01. Full-size license probe passed.
