# Approved sequencing — 16 September 2026

This replaces the earlier blanket hold only for the specific actions below. It does not authorize a general refill of the cluster.

| User instruction | Execution |
|---|---|
| (a) One-arm action3 timing pilot; five more only if it works | Control job **342321 passed**: exact254,068-column pool,20/20 physically feasible sequences, no unknowns/errors,118.22seconds total and4.24GiB peak stageRSS. The authorized five further20-sequence pilots were submitted as **342380–342384**. No full-pool replay or CG/MIP was launched. [Pilot and gate](single_factor_pilot/README.md). |
| (b) Action2: plan dependencies and conditional seeds | **No jobs submitted.**24 seed-zero MIPs planned with `afterok` on their own item10 CG job; seeds1/2 only after an audited unresolved miss. Hard fleet cap k and three-hour fleet search retained. [Revised plan](../advisor_followup_20260916/mip_followup/README.md). |
| (c) Hold full40, fix recovery, check memory | CG341405 and downstreamMIP341406 held. Existing graph341404_0 continues. Fallback implementation and11 tests completed; the verified versioned fix is staged on Unicorn. Solver/scientific settings unchanged; held jobs still reference the original wrapper and must not be released blindly. **No Scaglione resubmission:** measured aggregate k32 MaxRSS exceeds120GiB. [Fix and tests](checkpoint_fix/README.md), [memory/timing evidence](memory_summary.json). |
| (d) Keep Doc67/35 with labels | Preserved: **67/102 matching the numerical event-model fleet lower bound**, **35/102 with an open fleet gap**, including26 audited longer searches. The different **70/102 matching GIRO's count** remains a separate column. No new P1 seed outcomes mixed into this frozen cohort. |
| (e) Wait for all requested endpoints | Four-result report gated on P1 items7/8/9 and all action1 F6 arms. No unrelated submissions. [Report gate](report_gate/README.md). |

## Memory: two measurements, not interchangeable

The scheduler query was restricted to user nc437, the correct submission window and exact k32 job names. Slurm job IDs recur historically; an unfiltered historical lookup is not valid evidence for these jobs.

| Chain | CG job | Slurm batch MaxRSS (GiB) | Solver process peak (GiB) | Final matrix + LP re-solve (seconds) |
|---|---:|---:|---:|---:|
|1|228596|232.18|41.32|24.25|
|2|228600|211.72|39.45|24.90|
|3|228604|206.17|39.37|11.75|
|4|228608|214.40|38.97|10.75|
|5|228612|206.78|36.78|10.70|
|6|228616|228.38|42.34|6.63|

All six jobs requested96G. Each used eight inheritance workers, while full40 is a fresh start without inheritance. Summing RSS across processes can count shared pages more than once; this is a possible explanation of the larger scheduler number, not a measured proportional/private-memory decomposition. The smaller solver-process peak alone does not establish total job memory. No PSS/cgroup peak was saved in this audit.

**Right-sizing decision:** the evidence does not establish that the full40 job can safely fit a request<=120G. Therefore the conditional authorization to resubmit on Scaglione is not exercised. The existing128G request remains held; retaining it is not a claim that128G has been validated as adequate. A live graph-stage snapshot showed4.68GiB, but the build was unfinished and is not a CG peak measurement. Further sizing needs a process-tree/physical-memory measurement or a justified fresh-run model; do not silently divide Slurm RSS by eight.

## Cost of the final LP after stopping

The six ordinary k32 wall-limit exits spent **6.63–24.90seconds** on their final matrix construction plus LP attempt, about0.05–0.17% of a four-hour allowance. Chains1/2 hit the final LP time limit; the other four completed it. The native solver's backend time is only part of that overhead, so the table includes construction and wrapper time.

These were ordinary wall-limit exits, **not actual preemption/shutdown signals**. The earlier signal tests establish that SIGTERM/SIGUSR1 can still invoke another LP; an early signal can leave a much larger remaining application budget. The observed6.63–24.90seconds is not a universal shutdown bound. This change fixes checkpoint selection/copying; it does not silently change solver mathematics or signal handling.

## Four-result delivery

After the gated endpoints arrive, report: (1) fresh-k15 target hits out of18 seeded trials; (2) C5k31 twelve-hour fleet/bound/status; (3) constrainedk5 original/fixed/fresh costs under each tariff at matched fleet; (4) k32 results per chain and seed, with defined variance and unresolved outcomes. Failed or physically invalid results are separate from a valid failure to reach the target. Report all seeds, not just their best result. SE3 science remains internal-only.

The hourly automation now uses this sequence. It may collect, validate and document results; it may not submit action2 or unrelated work merely because the queue is empty or the report gate becomes ready. Historical held jobs, EVSPV2G work and the excluded GPU node are unchanged.
