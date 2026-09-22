# Frozen-pool structure and fleet-parameter pilot

Five source pools, five arms each. The preparation reruns the original pinned physical admission gate once per case, authenticates the ordered route hash and incidence dimensions, and writes a compact read-only matrix for all five arms. Both objectives use the same original route costs and covering semantics. No CG, shared-capacity rows, changed physics, or new charging integer optimization is introduced.

`core.py` contains bitset-indexed incidence diagnostics and exact integer-arithmetic dual certificates with lower/upper variable-bound contributions. Diagnostics do not remove rows/columns from parameter experiments. Dominance is safe for nonnegative charging costs and binary cover with optional at-most fleet cap; it is not claimed for arbitrary side constraints. Ten-minute diagnostic limits are explicit; incomplete column scans report partial counts.

`runner.py` performs two 300-second LP diagnostics and 1,800-second fleet-only MIPs. Gurobi 12.0.3, eight threads, Seed 0, identical column order, unit fleet objective, baseline full greedy start. Arms change only MIPFocus=1, MIPFocus=2, PreSparsify=1, or the offline source stage-one incumbent start. The saved-incumbent arm is a proof/search diagnostic; its incumbent acquisition cost is excluded and it is not a new timed start algorithm. Strong-start identity mismatch blocks that arm explicitly.

`launch.py` records submission intent and each job ID atomically under a lock. Each of five preparation jobs gates only its own five arms. There is no arbitrary array throttle or all-case preparation barrier. All requests exclude scaglione-compute-01, use default_partition, 8 CPUs, 32 GiB and two hours, with restart-specific output paths and retained attempts. Preemption loses the branch-and-bound tree.

Run local tests with `python3 -m unittest discover -s experiments/mip_structure_20260922 -v`. They include randomized exhaustive objective-preservation and certificate checks. Preparation jobs also execute a tiny licensed LP/certificate smoke test on compute before heavy work.
