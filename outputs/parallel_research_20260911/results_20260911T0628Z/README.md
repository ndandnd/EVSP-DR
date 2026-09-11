# 02:28 EDT results

All18 fresh covering comparisons at k=5,8,10 are complete across six chains. Exact results and bounds: fresh_six_chains.csv.

| Chain | k=5 buses | k=8 buses | k=10 buses |
|---|---:|---:|---:|
|1|5|9|11|
|2|5|9|11|
|3|5|9|11|
|4|5|9|11|
|5|6|9|11|
|6|5|8|11|

These are validated incumbents, not uniformly proved optima. Newly completed P2/P4/P6 k10 MIPs have fleet11 versus pool bound10 at the one-hour limit.

Fresh P4k8 has certified CG termination: weighted objective800359.8182244832, fractional route weight8.0, zero artificials, min reduced cost−1.085e−9 against epsilon1e−4. Its saved-pool MIP proves9buses; stage2 charging remains time-limited. This is direct evidence of a fractional/integer gap in this generated pool, not proof that the full routing model requires9. Selected routes replay individually, but29trips are overcovered; duplicate removal/shared capacity remain unvalidated.

Default trial:23completed started attempts,zero recorded preemptions,includingfive full-budget solves. No stationary reliability or causal partition advantage is established. Warm P2 also now matchesk6. No execution failure required a rerun.
