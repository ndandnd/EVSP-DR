# Scoped operations — 23 September 2026, 00:01:28 UTC

One scoped SSH collection succeeded. **13 EVSP–DR jobs run:12 baseline (two graphs,six CGs,four MIPs), plus strict cached-CG824877.** Graphs42/44 are complete, one more than21:31. All52 pending baseline jobs retain genuine registered dependencies. No allocation failure, broken dependency or additional graph preemption was found; cumulative graph PREEMPTED records remain10, each with a running/completed replacement. No submissions, queue mutations or repeated polls. Held537227 and V2G were untouched.

The two remaining graph attempts661616_33/35 have elapsed22:18:03, below36h watchdogs/37h allocations. Baseline CGs run at C1/C3/C4k37,C2/C6k36 and C5k34; MIPs run at C1/C3/C4k36 and C6k35. Current pending dependencies are genuine, with no recovery indicated.

## Newly completed baseline results

Five CG allocations complete0:0 at `wall_limit`, zero artificials and no pricing certificate. All last minimum reduced costs remain negative. Restricted-master route weights and weighted objectives are not certified full-model bounds.

| Case / job | Target / trips | Iterations | Route weight | Weighted RMP | Last min RC | CG wall s |
|---|---:|---:|---:|---:|---:|---:|
|w1_k36 /661625|36 /871|51|35.000000000|3501592.518573|-56.289057033|14397.407405|
|w2_k35 /661640|35 /841|168|34.000000000|3401467.085179|-0.360977261|14394.484961|
|w3_k36 /661678|36 /851|222|36.000000000|3601490.549935|-0.054419224|14380.807087|
|w4_k36 /661697|36 /853|113|35.000000000|3501516.270129|-0.465991747|14371.734302|
|w6_k35 /661729|35 /831|130|34.000000000|3401482.521647|-0.433063510|14395.519919|

| Case | Separate graph build s | Cache I/O s | Wrapper wall s | App RUSAGE_SELF MB | Slurm batch MaxRSS KiB |
|---|---:|---:|---:|---:|---:|
|w1_k36|70558.046093|151.061916|14559.477201|54168.730469|300513356K|
|w2_k35|83681.350816|112.534644|14513.283841|50699.667969|272206436K|
|w3_k36|61890.314526|114.093606|14501.207515|51531.875000|272524324K|
|w4_k36|40566.177895|143.192780|14681.293461|51838.457031|283864452K|
|w6_k35|30999.746131|134.336469|14645.977279|49148.867188|269327012K|

All five reused prepared graphs. Graph preparation is separately charged; cache I/O is a reported component, not blindly added to CG/wrapper wall. Smaller ancestor runs are excluded from these per-case times. New saved positive supports reconstruct objectives within1e−5 and cover input trips within recorded1e−6 tolerance. Preserve earlier C2k33 numerical disclosure and prior C3k35MIP interruption accounting in previous reports.

**C2k35 MIP661641:37 buses / finite-pool bound34, target35 missed, fleet optimum open.** Both stages TIME_LIMIT.841trips;237,336 accepted columns, zero repairs/rejections, no GIRO augmentation. Native selected-route physical replay passes; independent route counting confirms coverage,172overcovered trips and226extra assignments. Duplicate cleanup and shared charger capacity remain unvalidated. No independent physical simulator/full-journal replay was run in this audit.

| Fleet optimize s | Charging optimize s | Charging incumbent / bound | Pool preparation s | Before-publication wall s | Scheduler / MaxRSS KiB |
|---:|---:|---:|---:|---:|---|
|1801.161067|1786.511133|2490.984000 /1367.307624|391.903397|4012.321181|01:07:36 /8198720K|

Full Gurobi endpoint lines agree with stage objectives/bounds; selected-route charge independently reconstructs. The charging bound applies only under the unproved37-bus incumbent cap. See editable CSVs and proof_lines.json for exact values.

## Strict cached CG824877 startup

RUNNING1:39:24 on snavely-cpu-01,restart0,8CPU16GiB. **15/15 startup checks pass.** The remote manifest equals approved206fafab; execution matches clean-model fedf4214 and the approved worker hash. The guarded CG_STARTED receipt establishes that native source/input/lineage/cache-byte checks and the Gurobi12.0.3 license probe passed before CG launch. This local audit verifies those receipt/command/hash bindings; it does not independently rehash the remote multi-gigabyte cache. Graph construction and MIP flags remain false.

The log tail confirms active331-row restricted-master solves, most recently9,611model columns/280,080nonzeros and printed weighted objective1.100468132e+06. This is interim LP-log evidence, **not a final route weight, CG endpoint, pricing certificate, integer fleet or full-model bound**. No result/COMPLETE/FAILED receipt exists yet; live MaxRSS is unavailable from sacct. The full-run16GiB memory requirement remains unproven. The approved command reuses the saved graph and starts its4hscientific allowance after graph loading; it never resumes the incompatible oldk19 checkpoint. Keep next MIP/k20 gated pending a verified endpoint.

## Source and audit scope

**252/252 baseline/copied-artifact checks pass**, plus the separate15/15strict startup checks.143downloaded file hashes verified; source/input identity, genuine dependencies, execution gates, saved LP coverage/reconstruction and both new MIP log endpoints verified. Exact files: `verified_summary.json`, `cg_endpoints.csv`, `mip_endpoints.csv`, `proof_lines.json`, `source_hashes.json`, `graph_attempts.csv`, `strict_cg/startup_audit.json`. `collector_provenance.json` preserves previous/current collector hashes and excludes28previously finalized solver jobs. Completed unrelated campaigns and graph gates were not recollected.

Baseline physics unchanged:240kWh/240kW,zero reserve,no terminal floor or shared capacity,flat tariff/startfee5,SOC2.5kWh,event5min,covering master. CG pina0e0bb7681c8451e3cbbbfa06aef390026d9af4b still reports git_dirty=true; native MIP871d057e1067411f09581e37d78f7c1ca43f68bb is clean. Retain both application RUSAGE_SELF and Slurm batch memory: forked inheritance workers are excluded from SELF, but this does not numerically reconcile Slurm or quantify shared-page accounting. No OOM/resource change. Registered allocations remain graph2CPU96G37h,CG8CPU96G5h,MIP8CPU24G2h.
