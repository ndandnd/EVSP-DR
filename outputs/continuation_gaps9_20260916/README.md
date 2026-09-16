# Chain 6, target 31: longer search of the same columns

The original one-hour MIP found 32 buses with a saved-pool bound of 31. This experiment asks whether those same columns can supply 31 buses with a longer integer search. It does not generate new columns or supply a GIRO solution.

Job 288847: default partition, 8 CPUs, 24 GB, 4.5-hour allocation; solver budget 12,600 seconds including 10,800 seconds for fleet minimization. The original 241,121 columns and greedy initialization policy are retained. A new search tree and different node can affect behavior, so elapsed time is not the only causal difference. Every attempt uses a private output directory and verifies the native Gurobi license and physical pool.

Input, code and ordered-pool hashes are recorded in manifest.json. Preflight and submission checks passed. No existing longer search of this pool was found. No dependencies are needed because its source CG has finished. scaglione-compute-01 is excluded; held jobs and EVSPV2G are untouched.

The 09:42 UTC collection began before this submission. The next full collection will include this campaign; scheduler receipts are separate operational evidence.
