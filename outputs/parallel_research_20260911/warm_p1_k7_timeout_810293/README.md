# Warm chain-1 k7 timeout

## Warm chain 1, k=7 — initialization timeout, job 810293

Verified 11 September at 20:14 UTC: Slurm TIMEOUT after 08:17:13 against 08:15:00 allocation. The event network loaded in 7.03 seconds (15,642 nodes, 74,597,352 arcs). The persisted status remains initializing, with zero iterations, no final LP, no pricing certificate and an empty column journal. The 7.66-second status wall time is the initial publication timestamp, not the completed runtime. The inherited-event-pool audit is null. Evidence places the timeout during initialization before the first recorded CG iteration; it does not identify a particular inherited route or establish infeasibility.

No usable child pool was saved, so its MIP cannot run and chain-1 k8–10 remain blocked by their true dependencies. The predecessor k6 pool and cached network remain available; this does not resume the lost child initialization work. No blind rerun was submitted. Recovery needs bounded, checkpointed initialization or a measured justification for a larger budget, tested against the same inputs and physics. Other chains and capacity retries remain active.

Evidence: outputs/parallel_research_20260911/warm_p1_k7_timeout_810293/evidence.json; collection monitor/20260911T201444Z.json.
