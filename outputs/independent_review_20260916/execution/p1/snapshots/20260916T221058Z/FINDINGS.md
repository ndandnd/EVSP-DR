# P1 endpoint check — 16 September2026,22:10–22:13UTC

27 new P1 runs remain active; no new completed endpoint was available. Existing job336492 (chain1,k32,seed0) is still active. Three completed reused seed0 endpoints were re-read and checked against frozen controls.

| Case | Integer buses | Gurobi fleet bound in this pool | Fleet optimum proved? |
|---|---:|---:|---|
| Chain3,k32,seed0 |33|32.00000000000003|No|
| Chain4,k32,seed0 |32|31.000000000000064|No|
| Chain5,k32,seed0 |32|31.00000000000006|No|

For each completed cell, verified: output SHA256; exact ordered-pool hash and column count; identical deterministic MIP-start record; source journal identity; Seed0;8 threads;10800-second fleet stage;12600-second total allowance; stage2 fleet<=incumbent; covering sense; matching physics; clean expected code commit; individual physical route replay; no added/repaired/rejected columns. `endpoint_audit.json` records each check. Original pool and current solver proof are distinct from a full-model proof.

| Finding tested | Result | Evidence/consequence |
|---|---|---|
| F2: chains4/5 at k32 currently miss the GIRO fleet target | **Refuted** | Their reused completed runs already found32. The review's earlier one-hour misses remain historical facts. |
| F2: the saved result proves that chain 3 cannot use 32 buses | **Refuted as a claim about the proof record** | The incumbent is 33 and the bound is about 32. Whether this pool contains a 32-bus solution remains **unresolved**. |
| F2: MIP seed variance explains the remaining warm gaps | **Unresolved** | All new seeds are still active. Three seed0 results cannot measure seed variance. |
| F2/F4,item8: chain5 k31 has a30-bus integer solution in its saved pool | **Unresolved** | The12-hour fleet search is running (job340968). A30-bus fractional LP does not establish a30-bus integer solution. |
| F4: changing depot power/SOC rules explains pool fleet differences | **Unresolved by P1** | These repeats retain original physics. The separate strict-physics CG experiment tests the changed model. |
| F5: at least four of six fresh k15 pools match15 with longer/three-seed search | **Unresolved** | Zero of18 new fresh endpoints completed. Do not change the scientific headline on submissions or partial search logs. |

These are reused findings, not new solver improvements. Do not count them again as independent replications. Source scheduler snapshot and result identities are preserved beside this note.
