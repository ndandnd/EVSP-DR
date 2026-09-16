# First longer MIPs for chains 2 and 4 at target 32

Question: can the unchanged CG column pool match 32 buses when the fleet search receives more time? The original one-hour runs found 34 buses in each case; bounds were 32 and 31 respectively. Neither excludes the target.

This is a fresh MIP tree on the same ordered columns and greedy initialization policy. All non-time settings and the native execution commit are checked against the original result. Hardware and solver timing can differ, so this is not a pure elapsed-time causal experiment. It does not create a CG certificate or validate duplicate removal/shared charging capacity.

Budget: 12,600 solver seconds total; 10,800 seconds for fleet minimization. The second stage constrains buses to at most the best first-stage fleet. Resources: 8 CPUs, 24 GB, 4.5-hour allocation, default partition, requeue with private attempt paths; scaglione-compute-01 excluded. No dependencies: both canonical parent outputs are complete and hash-verified. No duplicate longer submission was found by the preflight.

Jobs: 307592 (chain 2), 307593 (chain 4). Both passed the native Gurobi license test and started running. Collection 20260916T144440Z began before submission; these receipts are separate evidence and the next collection will include the campaign.

The immutable manifest records input, code, physics, objective, covering sense, pool hashes, initialization, comparator, resources and output paths. Validation and submission receipts are retained alongside it. No solver code changed.
