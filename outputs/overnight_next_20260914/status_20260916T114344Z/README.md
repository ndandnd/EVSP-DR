# Overnight research check

## 16 September 07:50 EDT — chain 5 reaches 31; completed result recovered

Snapshot114344 completed11:50:27UTC in403.1seconds, previous104308. Register3416rows/88groups;99CG/98MIP endpoints among102continuation cases. All seven summarizers passed.995preemptionattemptrows. One newly recorded FAILED job228599 (watchdog124, not preemption); no broken dependencies. SSH available.

Longer C5k31 completed31/bound30/open after180.179fleetminutes and210.149totalminutes. Original one-hour result36; unchanged254068-column pool and initializer count verified. This establishes the original covering pool can match31, but not fleetoptimality or executable duplicate-free schedules. Charging unproved. Original maxima26/28/31/29/26/28; includinglonger30/30/31/29/31/30. Largest observed matches are not monotone cutoffs.

New originals: C1k31=34/bound31; C3k32=33/bound32; C5k32=37/bound31. All open, individualreplaytrue, duplicate-removal/sharedcapacityfalse. C6k32CG239.569min, weighted3201343.193900256/weight32, minRC−0.1275508, timecapped/uncertified; MIPrunning. Remaining C1/C2/C4k32CGs continue.

Recovered C2k31 publication: job228599 failed its6300-second subprocess watchdog after result.json was written at3974.6seconds. Native runner prints result then returns; exact reason process did not exit remains undetermined (do not assert destructor/Gurobi cause). Recovery independently checked scheduler ended, execution/source/input/route hashes, full725trip coverage,32selectedroutes, and original individual-replay/two-stage validation flags.122trips are duplicated; no new physical replay/duplicatecleanup claim. Original result bytes copied to canonical mip_result.json with separate recovery provenance; failed execution/Slurm status preserved. See outputs/mip_publication_recovery_20260916. Recovery enabled prepared follow-up; no original MIP recomputation or budget change.

New first longer searches:294595(C2k31, root11),294800(C1k31),294801(C3k32),294802(C5k32; latterthree root12). All RUNNING/native-license passed; manifest/source/non-time/pool/no-duplicate checks passed. Each12600total/10800fleet seconds,8CPU24GB4.5h/default/requeue/privateattempts/excludecompute01; no dependencies because verified sources complete. Root11 manifestSHA c46146565099d407ff8a0cf5e4907604d9a284c6b9b6fd80176cadec33fdda07; root12 a1dbcda6515c9af7b4302c31feaf84d9dacfc20953184606d089254dc1872ec3. Collector/register/summarizer wired for nextfullsnapshot; currentcollection beganbefore their submissions. Root11 is now submitted, superseding prior PREPARED ONLY note. Existing C4k31/C6k31 longer searches remain. Held537227/V2G untouched.

Doc currenttab table/date/source updated; Q4 now explains C5's target recovery and remaining30vs31gap. Figures/history preserved; Slides untouched. Ten LPendpoint corrections remainvalid; no new change. Morningconsolidation: perform at next08:40EDT heartbeat, ready around09:00 even with pendingresults. Show actual31/32counts, CGstop/source and concise remainingquestions. Next scientificpriority is larger-chain duplicate-removal validation, then additional matchedzero-fee instances. Do not repeat completed longersearches just to raise utilization.

[Chain table](CHAIN_TABLES.md) · [Detailed results](all_chain_extension_results.csv) · [Longer searches](longer_gap_results.csv) · [Publication recovery](../../mip_publication_recovery_20260916/README.md).

SnapshotSHA:ad103f86761f65026c3df24efe06c661b7c73ec1239b57afa64b27c29ca89979.
