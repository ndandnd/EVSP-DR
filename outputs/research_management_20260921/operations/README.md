# Operations and strict scaling, 21 September

**Verified at 16:08 EDT: 47 jobs running in this scope.** All 44 baseline k33–40 graph builds, strict k17 CG, and both representation benchmarks are running. The remaining 99 pending solver stages have genuine dependencies: 96 baseline stages and three new strict stages. No broken active dependency was found. Held historical array537227, old strict descendants and EVSP V2G work were untouched.

[Final scheduler/graph snapshot](snapshot_20260921T200821Z/snapshot.json) · [48-case status table](snapshot_20260921T200821Z/case_status.csv) · [Dependency/resource/attempt audit](snapshot_20260921T200821Z/campaign_audit.json).

All 48 baseline cases pass current dependency and reserved-node exclusion checks; graph array concurrency is44. Six graph attempts were **preempted**, then automatically requeued: array tasks6,14,26,27,37,41. Their lost wall time totals84.6minutes. All six were running again. Fifty attempt directories preserve execution/progress ledgers; no completed graph cache exists yet. Current process-memory samples range1.255–7.490GiB and are startup observations, not final peaks. There is no resumable partial graph artifact: future checkpoint work could reduce the cost of interruption, but these44running pins/submissions were not changed.

## Strict packed endpoint: computational recovery, with an integer pool gap

The case is **C5 parent global prefix k16, 18E2 subgroup: 256 trips and9 reference duties**. It is not a global16-bus fleet experiment.

[Full CG result](strict_packed/cg/646675_r0/result.json) · [Full final MIP result](strict_packed/mip/646676_r0/result.json) · [Compact endpoint and source hash](k16_endpoint_summary.json) · [CG metrics and hash](k16_cg_summary.json).

| Evidence dimension | Verified result |
|---|---|
| CG scheduler |646675 COMPLETED;4h00m13s |
| CG batch MaxRSS |4,124,768KiB =3.933685GiB |
| Graph construction |6,922.798699s =115.379978min; **included** in original4hCG budget |
| Graph representation |121,027,720 packed arcs;1,936,443,520 packed bytes;zero Python arc objects |
| CG progress |2,453 completed iterations;3,086 pool columns;zero artificials |
| Restricted LP |Weighted objective900,393.189605; fractional route weight9 |
| Pricing certificate |**No**; pricing deadline. Last completed minimum reduced cost−20.344798 |
| MIP scheduler |646676 COMPLETED;1h00m09s; MaxRSS1,614,492KiB =1.539700GiB |
| Finite-pool fleet proof |**10 buses / bound10**, proved in26.933967s |
| Charging stage |Time limit after3,573.080497s; charging-related cost736.184; bound452.136867; gap38.5837% |
| Individual-route physical replay |**Pass:** all3,086 pool routes replayed after authenticated trip remapping in native validation668429 |
| Coverage / exact-once dispatch |All256 trips covered; **57 extra assignments across53 trips**, so not exact-once |
| Shared station capacity |**Fail:** 7880C peak3 connections versus1 charger; JON_A peak2 versus1 |
| Shared capacity enforced in this arm |No; this is a diagnostic of a constraint omitted from the optimized model |
| GIRO subgroup target9 |**Not attained**; no full-GIRO dispatch-validity claim |

The saved pool provably cannot attain9 buses, although its fractional route weight is9. This is an integer gap in this finite pool; no full-model lower bound or proof that the physical problem needs10 buses follows. The result does not provide a pricing certificate, charging-cost proof, exact-once dispatch, or shared-capacity feasibility.

The old k16 attempt timed out during graph construction after6h02m with211.9GiB peakRSS. Packed storage now enables2,453CG iterations within the unchanged4h scientific budget, at3.934GiB measured peak. Nodes and stopping behavior differ, so this is not a controlled old-versus-new full-run speed ratio. The earlier26-trip same-node fixed-dual benchmark is retained separately.

Physics remain the original single-factor arm: actual239.01kWh battery,35.8515kWh reserve,PARX60kW,non-PARX240kW,2.5kWhSOC/5min event lattice, reserve-only terminalSOC, set covering, and no shared capacity. CG minimizes100000×fractional route weight plus charging-related cost; final MIP minimizes fleet then charging-related cost. These are not full-GIRO constraints.

## Seven new allocations, four scientific cases

| Allocation | Job | Question / input | Request | Dependency / latest state |
|---|---:|---|---|---|
| Native validation |668429|Replay all3,086 parent routes under k17 remapping; inheritance tests|2CPU/8G/15min|Completed in24s; passed |
| k17CG |668430|277 trips /10 reference duties; inherit k16 pool|8CPU/16G/6h allocation;4hCG including graph|Validation668429; running |
| k17MIP |668432|Fleet then charging in k17 pool|8CPU/24G/75min allocation;1h solve|Own CG668430; dependency wait |
| k19CG |668433|331 trips /11 reference duties; inherit k17 pool|8CPU/16G/6h allocation;4hCG including graph|k17CG668430; dependency wait |
| k19MIP |668434|Fleet then charging in k19 pool|8CPU/24G/75min allocation;1h solve|Own CG668433; dependency wait |
| Representation benchmark |668478|53 trips: original explicit / new explicit / packed|4CPU/16G/2h|Independent; running |
| Representation benchmark |668479|90 trips: original explicit / new explicit / packed|4CPU/48G/4h|Independent; running |

All use default_partition, exclude scaglione-compute-01, and retain unique job/restart output directories. Both benchmarks are independently eligible. Each executes its three variants serially on one node with five identical deterministic dual vectors, input/event-lattice identity checks and physical replay of all15routes. They produce fixed-dual algorithm evidence, not CG certificates or integer endpoints.

The16GiB successor request follows measured k16 peak3.934GiB and a331-trip quadratic projection6.58GiB, leaving2.43× headroom. k18adds no18E2duty; the original true group dependency is k17→k19. k20is not queued: its367-trip quadratic graph projection is237minutes, nearly the4h scientific budget. Inspect actual k17/k19timing first. No matching strict graph cache exists; baseline cache physics differ. Benchmark memory requests follow the measured26-trip explicit peak2.063GiB with approximately1.9× headroom over quadratic projection.

[Successor manifest and input hashes](strict_successors/manifest.json) · [Exact commands/dependencies/receipts](strict_successors/jobs.json) · [Native full-parent replay](strict_successors/validation/validation/668429_r0/result.json) · [Native six-test log](strict_successors/validation/validation/668429_r0/tests.log) · [Benchmark manifest](strict_benchmarks/manifest.json) · [Benchmark receipts](strict_benchmarks/jobs.json).

The solver remains the clean pinned35770aae2c08e7d5a356cc3b673e67608e5b1036 checkout. Packed→packed inheritance needs no compatibility exception; all hashes, trip mappings, physics and full route replay remain mandatory. Wrapper source is isolated on `codex/strict-successors-20260921`, commits e0f2bfa2 and550bc5b0. Six local and six native inheritance tests pass. No running solver source was modified.

## Preservation and monitoring notes

The first raw collection at19:55:03UTC omitted a sacct start-date filter and therefore includes reused historical job IDs. Its accounting view is superseded by19:55:26UTC and later snapshots, which explicitly restrict accounting to21September while retaining duplicate/requeue attempts. Prior snapshots and source artifacts remain preserved.

New remote roots are `/home/nc437/ladder-lite/strict_packed_successors_20260921/` and `/home/nc437/ladder-lite/strict_representation_benchmarks_20260921/`. The scoped collector here knows both roots; the parent task owns global collector/register integration and Google Doc publication. Original historical holds and Slides remain unchanged.
