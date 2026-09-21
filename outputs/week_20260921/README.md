# Week of 21 September 2026 — EVSP–DR

[Concise research journal](RESEARCH_JOURNAL.md) · [Full solver evidence and log-reading guide](evidence/README.md) · [Capacity and strict-physics audit](capacity_strict/README.md) · [Matched k5 figure](cleanup_physics/one_bus_k5_joint_matched.png)

The eight jobs from the previous update have finished. Six capacity CGs produced completed pool MIPs; the C1 incumbent-transfer follow-up now proves eight buses; strict k15 used its budget building the graph and completed zero pricing iterations. The [journal](RESEARCH_JOURNAL.md) gives the eight-job table, interpretation, algorithm work and concise dated history.

The main new evidence is auditable locally:

- **Fresh k8:** four finite pools provably require nine; appending eight sequential witness routes to each restores eight. [Complete logs and exact proof lines](evidence/LOG_EXCERPTS.md), [paired results](evidence/k8_witness_summary.csv).
- **Why:** the LP splits eight buses over 80–101 fractional routes, while integer scheduling needs compatible whole routes. [Dual-cost decomposition, route origins and source checks](evidence/README.md), [recomputed values](evidence/mechanism_summary.csv).
- **Fresh k15:** all twelve 12-hour fleet searches and subsequent charging stages are finished, with fleets 16–19 and fleet bound 15; none proves that 15 is absent. [Completed results and log locations](evidence/k15_12h_summary.csv).
- **Charging-aware duplicate removal:** a real k5 repair removes its repeated trip, preserves five buses and validates exactly-once service. [Result and before/after metrics](cleanup_physics/cleanup_result/summary.json), [validation](cleanup_physics/cleanup_result/validation.json).
- **Strict-physics algorithm benchmark:** on one matched 26-trip case, packed construction is 2.90× faster, peak memory 14.1× lower and mean pricing 530.8× faster, with five identical reduced costs and all15 route replays passing. This is a small-case benchmark, not large-case or full-CG evidence. [Verified metrics and scope](capacity_strict/README.md); k16 recovery646675 now runs.
- **Physics-matched k5:** the old figure was not fully matched. [Replacement figure](cleanup_physics/one_bus_k5_joint_matched.png), [data](cleanup_physics/one_bus_k5_joint_matched.csv), [validated constraints](cleanup_physics/joint_validation.json), [settings and input hashes](cleanup_physics/manifest.json). Post-hoc charging of saved trip sequences validates five buses and 62 exactly-once trips in both arms, including charger counts; fee0/fee5 use42/30 starts. This is not fresh CG or a complete GIRO constraint validation. [Full scope and logs](cleanup_physics/README.md).

For source retrieval, all 35 full Gurobi logs and 193 hashed copied source files are under `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/`. [The source manifest](evidence/source_manifest.json) maps every file to its original Unicorn path and SHA-256. The [evidence builder](evidence/build_evidence.py) reproduces and checks these tables without rerunning optimization. Capacity artifacts and new recovery receipts are indexed separately in [capacity_strict](capacity_strict/README.md).

The historical experiment entry point remains [the research register](../research_register/README.md). New dated entries supersede specific conclusions/statuses while retaining original logs, failed attempts and experimental budgets. Scheduler success, pricing certificates, finite-pool proofs, physical validation and GIRO target attainment remain separate.
