# Integer-directed pricing replication — 21 September 2026

Execution source: branch `codex/integer-columns-20260921`, commit `1be819f0` (full hash in generated manifest). Isolated local worktree `.codex-work/integer-columns-20260921`. Remote root `/home/nc437/ladder-lite/integer_columns_20260921`.

Question: can integer-directed pricing restore fleet eight on the four fresh pools with numerical nine-bus finite-pool proofs, using only each fresh pool and original instance/physics? Original pilot is retained separately; C1 follow-up628441 was additional work, not a retroactive benchmark replacement.

## Prespecified balanced first wave

C1/C3/C4/C5 at k8, seeds20260921 and20260922, two arms each: unchanged-pool control and dive plus augmented-pool MIP. Total16 independent allocations. Both MIPs use eight threads, covering, two-stage fleet then charging objectives, gap1e-4. Treatment dive uses cap8, max2400s wall, reserve60s,30columns/iteration,400iterations/node,150s/node,40nodes,3alternatives,2restarts, rc1e-4. Seeds affect Gurobi; no claim of independent stochastic pricing trajectories.

Each arm has3600s shared **dive subprocess wall plus MIP solver** budget. Control gets3600solver seconds (fleet stage1800). Treatment deducts the entire measured dive subprocess wall, including cache loading, hashing, source loading, master setup, pricing, publication and incumbent export; final MIP gets floor(3600−elapsed), with no minimum-floor extension. Fleet stage gets half that remainder. If exhausted, MIP is skipped. MIP setup, physical pool preparation and final replay are measured external overhead; full end-to-end elapsed is recorded and **not claimed capped at3600**. Gurobi's practical time-limit granularity is reported using actual runtime. Both arms retain independent physical validation. Historical one-hour solver baselines stay separate.

A hash-verified graph is a shared prerequisite. Original graph build costs are recorded separately; no graph is falsely claimed to have been rebuilt within this experiment. Graph cache from a warm campaign contains the graph only: graph construction/pricing methods, physics, instance, tariff, reference, deadhead identities and cache bytes are checked; no warm route/dual/witness columns are imported.

Treatment incumbent handoff exports full route records already in its own augmented journal, matched on incidence and expanded-grid cost within1e-7 absolute. Full chosen record hashes and source line ordinals are saved. The MIP independently replays these records; result gates require validated fleet matching the dive and zero added/replaced pool records. The generic historical `added_giro_route_count` field is not provenance; these routes arise from this dive. No sequential/GIRO witness injection.

Physics unchanged:240kWh,240kW, zero reserve, SOC step2.5kWh, event grid/block5min, flat tariff, charging-start fee5, bus coefficient100000, covering, no shared station-capacity constraint. Weighted objective is distinct from route weight/fleet. Artificial slack remaining after penalized pricing closure is **not** an unpenalized node infeasibility certificate. Global CG/branch-price certificate remains null; finite-pool fleet proof, target attainment, route replay, duplicate-trip removal and charger-capacity validation remain separate columns.

Resources: default_partition, --requeue, unique job/restart output directories, eightCPUs,16G,2h allocation, exclude scaglione-compute-01. No dependency and no concurrency throttle. Memory reason: original treatment MaxRSS1.9–4.0GB (sacct586633/636/639/642),16G gives margin while avoiding prior blanket96G over-request. Held537227 and stochastic project untouched. Full remote resource policy read before submission.

## Conditional next wave (not yet submitted)

Prepare six k15 chains, selecting bounded treatment/control comparisons after first-wave execution and physical-handoff gates pass. Fresh pools only. Exact case count, parameters and resources will be preregistered before launch and coordinated with root. Do not repeat12h unchanged-pool searches.
