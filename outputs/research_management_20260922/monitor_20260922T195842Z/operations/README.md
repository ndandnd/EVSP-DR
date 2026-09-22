# Scoped operations — 22 September 2026, 21:31:53 UTC

The directory retains the 19:58:42 heartbeat label; **the actual single SSH collection was 21:31:53 UTC**. SSH succeeded. **9 baseline jobs run: 3 graphs and 6 CGs, no MIPs; 41/44 graphs are complete**, eight more than the15:58 snapshot. All62 pending baseline jobs have genuine registered predecessor dependencies. No failed allocation, broken active dependency or additional graph preemption was found; cumulative preemptions remain10 and each has an active/completed replacement. Held537227 and V2G were untouched. No submissions, recovery mutations or repeated polls were made.

The six running CGs are C1/C3/C4k36,C2/C6k35 and C5k34. All pending cases retain true predecessor order. The three graph tasks661616_33–35 have current elapsed19:48:28 against37h allocations/36h watchdogs; the longest running CG is C2k35 at3:22:52 against5h allocation/4h scientific allowance. No deadline or memory recovery is indicated.

## Six newly terminal CG endpoints

All complete0:0 on `wall_limit`, zero artificials, no pricing certificate; all last reduced costs remain negative. Fractional route weight and weighted restricted-master objective are separate from certified full-model bounds or integer fleets.

| Case / job | Target / trips | Iterations | Route weight | Weighted RMP | Last min RC | CG wall s |
|---|---:|---:|---:|---:|---:|---:|
|C1 k35 /661622|35 /834|51|34.000000000|3401539.076369|-36.053183700|14400.352368|
|C2 k34 /661638|34 /824|184|33.000000000|3301404.815012|-0.683935541|14390.517585|
|C3 k35 /661676|35 /827|180|35.000000000|3501465.873119|-0.053931946|14395.207682|
|C4 k35 /661695|35 /830|106|34.000000000|3401492.419026|-0.499641287|14385.643089|
|C5 k33 /661707|33 /768|226|32.000000000|3201404.360670|-0.031846327|14380.560643|
|C6 k34 /661727|34 /804|222|33.000000000|3301460.702601|-0.173543200|14394.731259|

| Case | Separate graph build s | Cache I/O s | Wrapper wall s | Scheduler | App RUSAGE_SELF MB | Slurm batch MaxRSS KiB |
|---|---:|---:|---:|---|---:|---:|
|w1_k35|54836.939510|205.026069|14665.269394|04:04:39|49446.589844|278411368K|
|w2_k34|58279.165315|130.006949|14517.683541|04:02:09|49014.851562|263434392K|
|w3_k35|65726.234985|138.111690|14676.947204|04:04:48|48576.566406|257945076K|
|w4_k35|38224.010098|123.977113|14511.503464|04:02:02|49036.554688|270124240K|
|w5_k33|51400.097265|94.137385|14481.596840|04:01:28|41726.265625|236847932K|
|w6_k34|28218.016765|100.965895|14502.726427|04:01:50|45849.589844|252977484K|

All six reuse prior graph caches; graph preparation is separately charged. Cache I/O is a reported component and must not be added blindly to CG/wrapper wall. These are single-case times, excluding ancestor-prefix costs. Each new positive-route objective reconstructs within1e−5 and coverage within recorded1e−6 tolerance. The historical C2k33 positive-support discrepancy remains disclosed in the previous report; it was not recollected or erased.

## Ten newly terminal finite-pool MIPs

All complete0:0 with both optimization stages TIME_LIMIT, no proved fleet optimum and no target attainment. Native pool preparation accepts all columns with zero repairs/rejections and zero added GIRO routes. The recorded source status/journal hashes and unchanged ordered-pool hashes agree. Native selected-route replay passes; independent saved-route counting confirms trip coverage and the duplication counts below. Duplicate-service cleanup and shared-capacity validation remain false. No independent physical simulator or full-journal scan was run by this audit.

| Case / job | Target / trips | Fleet / pool bound | Accepted columns | Overcovered trips / extra assignments |
|---|---:|---:|---:|---:|
|w1_k34 /661621|34 /821|40 /33|286,834|223 /299|
|w1_k35 /661623|35 /834|42 /34|288,377|259 /317|
|w2_k34 /661639|34 /824|36 /33|232,279|183 /253|
|w3_k34 /661671|34 /797|35 /34|208,301|106 /128|
|w3_k35 /661677|35 /827|36 /35|213,731|133 /166|
|w4_k34 /661694|34 /816|42 /33|257,589|300 /420|
|w4_k35 /661696|35 /830|42 /34|260,783|275 /401|
|w5_k33 /661708|33 /768|39 /32|268,700|266 /397|
|w6_k33 /661726|33 /796|38 /32|251,615|154 /182|
|w6_k34 /661728|34 /804|40 /33|258,283|245 /294|

| Case | Fleet optimize s | Charging optimize s | Charging incumbent / bound | Pool preparation s | Before-publication wall s | Scheduler / MaxRSS KiB |
|---|---:|---:|---:|---:|---:|---|
|w1_k34|1800.815894|1788.931337|2316.160000 /1289.095173|339.168540|3953.838698|01:06:13 /15265056K|
|w1_k35|1801.072337|1788.217034|2380.344000 /1349.862561|397.445826|4013.216658|01:07:25 /10119596K|
|w2_k34|1801.765915|1784.107746|2404.536000 /1301.479274|419.356327|4043.762224|01:07:59 /8053996K|
|w3_k34|1801.065535|1792.082056|2383.800000 /1397.619550|274.918076|3893.387870|01:05:14 /10519284K|
|w3_k35|1801.265222|1791.238273|2426.968000 /1430.401265|263.741927|3880.778541|01:03:12 /8793584K|
|w4_k34|1802.299960|1782.785354|2693.280000 /1258.275968|493.599679|4115.767467|01:09:16 /9054792K|
|w4_k35|1801.481595|1787.143884|2704.144000 /1317.085836|351.188773|3965.507125|01:06:30 /11635076K|
|w5_k33|1800.944107|1790.496354|2439.984000 /1232.637886|322.484316|3941.148489|01:05:57 /14078820K|
|w6_k33|1800.579499|1790.782627|2278.368000 /1283.052547|274.753293|3886.488160|01:05:07 /11917476K|
|w6_k34|1800.927309|1789.461430|2442.504000 /1259.128563|380.668867|3999.480078|01:07:12 /12713344K|

C1k34 has a small numerical distinction: Gurobi stage2 variable objective is2316.159981080766, while the selected-route cost sum is2316.159999999218, a difference of0.000018918452. The log agrees with the recorded variable objective; the reported selected cost independently reconstructs from selected route costs. Both are retained in the CSV rather than conflated. Charging bounds apply under the respective unproved incumbent fleet caps.

**MIP restart evidence, separate from graph preemptions:** C3k35 job661677 has preserved attempt r0 marked `interrupted`, return code−15, signal15, watchdog false, and wrapper elapsed3786.563429seconds; r1 is the finished audited36/bound35 endpoint. These receipts establish an interruption and restart, but the collected scheduler rows do not establish its cause as PREEMPTED. The cumulative10 figure above refers only to graph PREEMPTED records. The MIP table reports r1 timings; retain r0 work separately in total campaign accounting. No additional scheduler query was made.

## Strict graph validation receipt

Job772820 completed0:0 in3:10:10 on joachims-cpu-02, batch MaxRSS6,375,228KiB, restart0. The small COMPLETE/cold/reload/result receipts report passed, two fixed diagnostic dual vectors,8397 initial columns and separate cold/reload processes. No cache payload was downloaded. Exact scientific interpretation and next-step gate are delegated to the strict review in `../strict_review/`; scheduler completion alone is not asserted to be graph parity. No CG/MIP, pricing certificate, target attainment or shared-capacity result follows from this graph job.

## Source, physics, memory and audit scope

Baseline source/physics remain unchanged:240kWh battery/initial SOC,240kW charging,zero reserve,no terminal floor,no shared capacity,flat tariff/fee5,2.5kWh SOC grid,5-minute events and covering master. CG objective is100000 times fractional route weight plus charging-related cost. CG execution pina0e0bb7681c8451e3cbbbfa06aef390026d9af4b reports `git_dirty=true`; retain that flag. MIP source871d057e1067411f09581e37d78f7c1ca43f68bb is recorded clean. These claims are verified source provenance, not a new source-diff diagnosis.

Memory scopes remain distinct: pinned exact_pricer_expanded.py:609 forks8 inheritance workers; exact_cg_telemetry.py:29 uses RUSAGE_SELF, excluding worker RSS. This establishes different accounting scope but does not reconcile the numeric Slurm aggregate or quantify shared pages. No OOM or automatic resource change. Registered graph/CG/MIP allocations remain2CPU96G37h /8CPU96G5h /8CPU24G2h.

**423/423 audit checks pass.** The audit verifies171 copied-file hashes, source/input identities, execution gates, genuine dependencies, positive-route reconstruction and coverage, all20 Gurobi stage endpoint lines, finite-pool proof scope, native physical flags and selected-route coverage/cost. `cg_endpoints.csv`, `mip_endpoints.csv`, `verified_summary.json`, `proof_lines.json`, `source_hashes.json` and `graph_attempts.csv` preserve exact values. `collector_provenance.json` identifies the adapted prior collector, its source hashes and the12 excluded already-audited solver jobs. No completed unrelated campaign was recollected. Root owns shared records and publication.
