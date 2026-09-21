# Integer-directed columns — 21 September 2026

**Status at2026-09-21T20:24:24UTC:**16balanced k8 allocations submitted; **four treatments completed and prove fleet8**, and12jobs remainRUNNING. Completed treatments are C3andC5 at both seeds20260921/22: two repeated cases, not four independent instances. All four own-dive incumbents were accepted as eight-bus MIP starts, with zero added/replaced start columns and physical route replay passing. End-to-end times are14.1–18.8minutes. Controls are not yet complete, so no overall paired success-rate or causal-performance claim is made. Conditional k15 gate668797 retains its true afterok links to all16k8jobs; no new queue poll or k15submission was made during this result audit.

## Executable change

Isolated branch `codex/integer-columns-20260921`, local worktree `/Users/nadan/Documents/projects/demandresponse/.codex-work/integer-columns-20260921`.

- `1be819f0b4e3ea9dc9766497efa169a03e476ad5`: full own-journal incumbent export, explicit penalized-artificial reporting, matched replication runner and six regression tests. k8 execution pinned here.
- `c50e5f207869bac25507adfae90bb830a44039b7`: registered budget parameterization, used only in separate k15 checkout; k8 code was not changed after submission.

55dive/cache/handoff tests passed locally and on Unicorn for the k8 commit (working Gurobi license);55passed locally after parameterization. The six added regressions cover full physical record export, cost collisions, missing physical records, invalid cover/duplicate incidences, no-incumbent behavior and budget-floor removal. The final runner independently replays the exported routes and requires zero added/replaced columns and matching validated fleet. Full record hashes/source line ordinals make incidence collisions auditable. Artificial slack has an explicit false unpenalized-infeasibility-certificate flag.

## Evidence and interpretation

| Experiment | Fresh pool control | Integer-directed outcome | Proof / limits |
|---|---|---|---|
| Original pilot C1 |9 | Dive8, final MIP9 | Handoff defect; elapsed3692s exceeded a strict3600s claim |
| Original pilot C3/C4/C5 |9 each | Final MIP8 each | Native dive columns; route replay passed; no shared capacity model |
| Additional C1 follow-up628441 |Same original augmented pool |8 | Existing dive start; finite-pool fleet8 proof, charging447.44; additional work, not benchmark replacement |
| New balanced k8 replication |8controls stillRUNNING at dated snapshot |4treatments prove8(C3/C5,bothseeds);4stillRUNNING | Accepted own-dive handoff; controls incomplete, no overall paired claim |
| Conditional paired k15 C1/C3/C5 |3controls prepared |3treatments prepared | Gate668797 must validate native k8 handoff/budget/physical replay first |

Original evidence is `outputs/week_20260921/evidence/README.md` and `outputs/research_execution_20260921/pilot_audit.json`; those source artifacts are preserved. The new experiment asks whether native pricing restores the target without external oracle columns. The original pilot and completed C1follow-up establish possibility on these four pools; four new treatments replicate target8 on two cases; remaining treatments and controls remain pending. The new k15 fresh-pool integer-directed pilot has no result yet; the historical fresh-pool searches establish no15-bus impossibility proof.

Balanced k8: four fresh cases×two seeds(20260921/22)×two arms.3600s shared dive-subprocess-wall+MIP-solver budget; all treatment cache/hash/setup/pricing/publication time charged. MIP physical preparation/replay are external measured overhead; actual end-to-end recorded and never claimed hard-capped. Graph caches are hash/method/physics verified; original graph-build cost is separate prerequisite accounting. MIP gets floor(total−divewall), no minimum-floor extension. The unchanged one-hour baseline remains separate. Scientific settings and source hashes are in `manifest.json` and `PREREGISTRATION.md`.

k8 job ledger `jobs.tsv`:668455,668457–668471.8CPU16G2h/default/requeue/exclude scaglione-compute-01;16G justified by measured original treatment1.9–4.0GB peaks. No throttle. Gate668797:1CPU1G10min/default/requeue/same exclusion. Every submission exclusion verified. Complete Gurobi logs are retained in unique remote job/restart directories. No held job, stochastic project, Google Doc, Slides or global register was edited by this subtask.

## Verified completed treatment receipts

All48copied files match remote SHA-256 values and the execution receipt output hashes. Full dive/MIPGurobi logs, result, own-incumbent export, source/augmented manifest and budget/handoff receipts are in `results/`. `artifact_manifest.json` records exact local/remote paths, hashes and proof lines. All four MIPlogs accept objective8atline30.

| Case | Seed | Job | New columns | Charged dive+solver s | Actual end-to-end s | Fleet8/bound8 proof line |
|---|---:|---:|---:|---:|---:|---:|
|c3_k08|20260921|668461|15823|1091.19|1125.92|65|
|c3_k08|20260922|668463|12148|817.53|848.11|61|
|c5_k08|20260921|668469|11169|1008.56|1064.64|70|
|c5_k08|20260922|668471|14049|1064.20|1120.15|66|

Each MIP solver budget equals floor(3600−measured divewall), with no minimum-floor extension; actual charged usage stays below3600s. All four source pools remain immutable. These are numerical finite-pool fleet proofs and route-level replay results. Duplicate-trip removal is **not validated** for these four witnesses; shared charger capacity is **not imposed/validated**. No full global branch-and-price certificate, complete dispatch validation or unrestricted eight-bus optimality claim follows. No external witness column was used.

The native handoff/budget/physical eligibility evidence now exists, but the scheduled gate still waits for all16k8jobs to exit successfully. Preserve that scheduled continuation unless the root manager deliberately invokes the already-authorized idempotent command after inspecting failures.

## Manager continuation

Remote k8 root `/home/nc437/ladder-lite/integer_columns_20260921`; k15 root `/home/nc437/ladder-lite/integer_columns_k15_20260921`.

Refresh evidence (read-only):

```bash
ssh nc437@unicorn-login-01.coecis.cornell.edu 'python3 /home/nc437/ladder-lite/integer_columns_20260921/collect_replication.py /home/nc437/ladder-lite/integer_columns_20260921'
```

Gate668797 automatically invokes the preapproved k15submitter after all16k8 jobs exit successfully. The submitter independently requires one completed native own-incumbent handoff with validated fleet, zero added/replaced columns, final route replay, immutable source, exact no-floor budget accounting and recorded end-to-end wall. It refuses known failed/cancelled/OOM/timeout k8 jobs. **A target miss alone is not an infeasibility certificate or reason to alter scientific settings.** Investigate actual code, physical-validation or resource failures and repair true dependencies before continuing.

The same safe continuation command may be invoked by the4hmanager when native eligibility is verified:

```bash
ssh nc437@unicorn-login-01.coecis.cornell.edu 'bash /home/nc437/ladder-lite/integer_columns_k15_20260921/submit_wave.sh'
```

It returns2 when no native gate has passed,3 for known k8 failures, and submits nothing in either case. It uses `flock`, preserves job/restart outputs, and skips each already-ledgered case/arm/seed. A mock Slurm test ran it twice and observed exactly six total submissions/six unique ledger cells. The real pre-gate invocation returned2 and created no k15ledger. If a manager submits before gate668797runs, the later gate is harmless because all six ledger cells are skipped. A completed gate is operational status only; inspect the six resulting research jobs and their proofs separately.

k15 preregistration is `k15/PREREGISTRATION.md`: C1/C3/C5, one seed, three control/treatment pairs,7200shared budget/max5400dive, cap15,8CPU32G3h. These six new allocations are authorized conditionally by root. Cache/source/method preflight passed; all six chains inventoried in `k15/six_chain_inventory.json`.32G request allows3.38–4.86GB graphs plus transient deserialization, larger pool and master; review actual peaks. No GIRO/warm/12h-incumbent route import.

After completion, rerun collector and copy `evidence_latest.*`, `scheduler_latest.txt`, `jobs.tsv`, the conditional gate receipt and k15ledger locally. Copy full Gurobi logs/results/receipts while leaving large journals intact on Unicorn; receipts hash every output. Keep scheduler completion, finite-pool proof, native target hit, route replay, duplicate removal and station capacity as separate columns. Root owns updates to the global experiment register/current Google Doc; Slides remain untouched.
