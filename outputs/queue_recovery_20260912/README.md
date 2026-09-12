# Queue recovered — 12 September 2026

Checked at **16:48 EDT**. The queue was blocked by failed or obsolete prerequisites despite available CPUs. Ready work is now running. Scientific source files were preserved.

| Work | Running | Waiting for required input |
|---|---:|---:|
| Full-pool warm chains through k=15 | 6 | 68 |
| Shared graph preparation and decomposition | 2 | 22 |
| Earlier bounded warm chains | 2 | 2 |
| Recovered MIPs with finished CG | 4 | 0 |
| Separate V2G project | 9 | 0 |

This is 14 EVSP–DR jobs plus 9 V2G jobs running; no job is marked DependencyNeverSatisfied or waiting for resources in this snapshot. Thirty-three held historical tasks remain untouched. These are dated counts, not a live display.

## What was wrong and what changed

- Nine decomposition MIPs were waiting on failed array dependencies even though their corresponding CG tasks had completed successfully, reported pricing certificates, and had no artificial coverage. We verified each saved status, journal and input identity, submitted array **39577**, then cancelled its nine obsolete predecessors. Slurm documents `aftercorr` as depending on the corresponding array task; the observed scheduler state was inconsistent with that expectation. We have not established the scheduler's internal cause. [Slurm documentation](https://slurm.schedmd.com/job_array.html).
- Four exact superseded chain-2 entries were removed after checking their replacements.
- Fourteen old full-pool chain entries depended on parents that had run out of time while checking inherited routes. They are replaced by indexed full-pool recovery, retaining true previous-k dependencies.
- Sixteen old decomposition entries depended on missing component solutions or graph builds that had timed out. Ready component MIPs were recovered; one missing component CG and ten parent combinations now have explicit prerequisites and share graph preparation.

**43 obsolete pending entries were cancelled.** Old artifacts and cancellation mappings remain in `evidence/`. More useful future jobs are queued now; removing dead entries does not imply that the total queue should be shorter.

Five of the nine recovered MIPs have already finished. Cases **d04_g2, d05_g1, d05_g3, d06_g2 and d07_g0** each found **8 buses for an 8-duty subset**, proved eight optimal within the saved column pool, and passed individual-route physical replay. This does not prove full-model integer optimality or joint station-capacity feasibility. Results are in `evidence/new_results/` and the dated experiment register.

Separately, the earlier bounded chain-2 k=14 MIP finished with **16 buses**, a pool fleet bound of **14**, and no fleet optimality proof after the one-hour budget. Its CG was certified after 159.5 minutes. The result is in the 20:48 UTC collector snapshot and remains a distinct treatment from the new full-pool chains.

## Which dependencies remain

| Job type | Must wait for | Why |
|---|---|---|
| Next-k CG | Previous-k CG | It reads that run's generated route pool. |
| Case MIP | Its own CG | It needs the generated columns. It can run alongside the next-k CG. |
| 32-duty parent CG | Shared graph cache and required component MIPs | It starts from validated component integer routes. |
| Parent MIP | Its parent CG | It selects integer routes from the expanded parent pool. |

There is no new arbitrary concurrency cap. All independent ready jobs were released. The recovered MIP array has a limit of 50, exceeding its nine tasks. Thirty-seven CG jobs along six chains cannot all start at once without discarding the requested previous-k experiment.

## Authoritative launch map

| Campaign | Cluster directory under ~/ladder-lite | Jobs | Source commit |
|---|---|---|---|
| Ready component MIPs | queue_recovery_20260912; outputs remain under overnight_extension_20260912 | 39577 indices 17,18,21,23,25,26,27,28,29 | MIP 871d057e1067411f09581e37d78f7c1ca43f68bb |
| Indexed full-pool warm chains | full_pool_recovery_20260912 | 37 CG + 37 MIPs | CG e091a4dba549510238507ef5e5367abea958bd30 |
| Shared graphs and decomposition, current attempt | graph_recovery_retry2_20260912 | caches 42508/42509; 11 CG + 11 MIPs 42510–42531 | CG a0e0bb7681c8451e3cbbbfa06aef390026d9af4b |

Full-pool root jobs: C1 k7 **41567**, C2 k9 **41568**, C3 k11 **41569**, C4 k10 **41573**, C5 k11 **41574**, C6 k11 **41575**. Each extends through k15. Exact dependencies and resource requests are in each campaign's `jobs.json` and `case_jobs.json` under `evidence/`.

All submissions use default_partition and exclude scaglione-compute-01. No Scaglione or V2G running job was cancelled. No held historical task was changed.

## Scientific settings and budgets

Both new CG treatments use covering, 240 kWh batteries, 240 kW charging, 2.5 kWh / 5-minute event discretization, flat electricity prices, no minimum return SOC and no shared-station capacity constraints. CG route objective is 100,000 + electricity + 5 × charge starts; 30 columns per iteration; reduced-cost tolerance 1e-4. These runs are baseline recovery, not the harsher GIRO-capacity treatment.

**Full-pool chains:** replay all inherited route sequences with the fixed-sequence index; no 512-route or 15-minute import cap. Overall solver budget is 8 hours for recovered k≤10 and 4 hours for k≥11. Eight CPUs and 96 GiB per CG. They must be compared separately with the earlier bounded512 treatment. Existing graph-cache files were hash-checked; producer compatibility was checked and retained in cache_compatibility.json.

**Decomposition:** build the missing d00_g3 and common parent32 graph once each, allowing up to 12 hours of graph preparation. d00_g3 CG then has 4 hours; each parent CG has 2 hours. Graph time is additional and must be reported separately. Parent initialization replays the selected integer routes from its four component solutions, then ordinary CG can find routes crossing component boundaries. This differs from the old import of at most512 sequences from entire component pools. No GIRO solution columns are injected; the grouping itself uses GIRO duty membership. Cache jobs request2CPUs and32/128GiB, component CG8CPUs/32GiB, parent CG8CPUs/128GiB.

**MIPs:** same validated source 871d057 for all new MIPs. One hour of optimization: first minimize buses for up to30minutes, then minimize electricity plus start charges with fleet ≤ the validated first-stage incumbent and remaining time. An unproved fleet is labelled unproved. Two-hour scheduler allocation permits loading and physical validation. Chain/component MIPs8CPUs/24GiB; parent MIPs8CPUs/64GiB.

## Validation and failed attempt retained

Indexed import plus shutdown repair: 30 tests passed and1 skipped on local and Linux validation. Combined graph code:32 tests and2 subtests passed. The graph tie-key optimization preserved graph equality in small real-data tests; its median benchmark improved11.466→4.650 seconds (2.47×). This is not a measured full-size speedup. See `graph_combined_tests.log` and `graph_optimization/`.

The first graph launch, under graph_recovery_20260912, passed the license check but rejected a cache-only CLI argument before building anything. Its dependent jobs cancelled automatically. The corrected attempt removes that argument, passed validation of the complete commands and the Gurobi license preflight, and both42508/42509 are now running with no stderr output. All first-attempt records are retained. Startup alone is not a completed cache or optimization result.

New full-pool and graph jobs use kill-on-invalid-dep=yes to prevent another impossible waiting backlog. They use exclusive attempt paths and no automatic requeue; after preemption or failure, validate any surviving data and submit a fresh recorded attempt. The ready-MIP worker supports automatic requeue with unique job/restart result paths. All new MIP cohorts are included in the default-partition preemption study.

## Clean queue command and monitoring

On Unicorn:

```sh
~/ladder-lite/drq
~/ladder-lite/drq --details
~/ladder-lite/drq --json
```

The command is read-only and groups EVSP–DR, V2G and held history separately. It reports waiting for input, waiting for resources and invalid dependencies separately. The existing collector includes all four recovery roots and their job maps. Monitor real completion and failure events; do not bypass required route data merely to clear a scheduler dependency.

Protect the first graph root: retry2 borrows its code and data. Protect efficiency_validation_20260912/code-baseline: full-pool and graph code checkouts borrow its Git objects. Do not clean up source caches, hard links, code object stores or active journals while these jobs need them.

`before.json` retains the original queue audit; `evidence/queue_snapshot.json` retains the16:48EDT view. SHA256SUMS binds the local evidence files. Files named launch_overnight.py, prepare_overnight_inputs.py and overnight_worker.py/.sub are preparation drafts; the authoritative submitted launchers are those copied from the cluster into evidence/ and listed above.
