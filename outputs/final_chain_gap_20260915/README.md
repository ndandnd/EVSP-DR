# Last original-chain gap — 15 September

Question: does the unchanged chain5 k25 pool support25 buses when its integer search has a longer allowance? The original one-hour MIP found26 buses with bound25. All other original k16–25 misses have target recoveries from separate longer searches.

This treatment changes the fleet-stage limit from1800 to10800 seconds and the total MIP limit from3600 to12600 seconds. It starts a new search tree with the same saved columns, input, cover formulation,240kWh/240kW physics, charging-start fee5, and871 solver. A recovery would show that the original pool contains the target; it would not prove extra allocated time alone caused the improvement. No new CG certificate is created.

The preparation script scans existing manifests for a longer MIP on the same input and journal and skips duplicates or already-matched targets. All source hashes, execution commit, original comparator, resources and exact arguments are frozen in manifest.json. Execution uses the identical previously validated worker and submission code from remaining_chain_gaps_20260914. Each attempt checks the full Gurobi license and replays pool feasibility. Private restart paths preserve preemption attempts.

Default partition,8CPUs,24GB,4.5-hour allocation,scaglione-compute-01 excluded. No dependencies: the source pool is already complete. This job does not change existing previous-k dependencies, held jobs, or EVSPV2G experiments. Results live under /share/scaglione/nc437/evsp-dr/final_chain_gap_20260915, with manifests/logs under /home/nc437/ladder-lite/final_chain_gap_20260915.

The02:09 scheduler check found27 running research jobs and19 true input dependencies. This targeted follow-up fills an identified gap while the existing larger-chain/compact-start work continues. The launch is separately timestamped from the hourly collection already in progress.

Submitted job **220545**. Observed RUNNING at02:14EDT on sablab-cpu-01; full-size Gurobi license check passed,8CPUs/24GB and required exclusion verified. One restart-specific preemption-registry entry exists. [Launch status](launch_status.json) · [Matched settings](matched_design.json).
