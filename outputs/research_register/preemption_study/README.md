# Default-partition MIP reliability study

User authorization: 11 September 2026 UTC. Use overnight default-partition capacity for saved-pool MIPs despite possible loss of branch-and-bound search. Main fleet/charging MIPs have a 3600-second optimization budget, including at most 1800 seconds in stage 1 and the remaining budget in stage 2. Scheduler allocation includes preprocessing and validation overhead; it is not the solver budget. The capacity pilot uses 1500/750 seconds instead.

## Cohorts and execution

Move only pending fresh-covering MIPs to default_partition. Preserve source pools, physics, two-stage objective, solver settings and corresponding freeze dependencies. Keep warm-chain and capacity MIPs on Scaglione as operational reference cohorts. They are different cases, not randomized controls. Do not duplicate running jobs. Exclude scaglione-compute-01 from CPU-only work.

Automatic requeue is disabled for these MIP attempts: existing wrappers refuse preexisting progress artifacts and cannot resume the Gurobi search tree. After confirmed preemption, a retry must use a new attempt/job ID and separate output/progress paths, linked to the same scientific case. Preserve any usable incumbent, but claim recovery only after validation. Never overwrite the interrupted attempt. Scheduler requeue is not solver resume.

## What is collected

The immutable case registry maps each scheduler task to a scientific case and cohort. `collect_attempts.py` reads allocation records through `sacct --duplicates --allocations`, retains source output, and samples priority factors and preemption configuration. Every collection is dated and hashed. Registry revisions are embedded in each snapshot. At launch, workers also record their node, partition, time and priority where supported.

Report separately:

- Submitted cases, started attempts, still-running attempts, completed allocations, validated usable results, and retried cases.
- Explicit PREEMPTED states; other cancellation, time-limit, memory and application failures remain distinct unless additional evidence identifies their cause.
- Eligible-to-start wait, allocation duration, allocated CPU-seconds and actual CPU usage when available. Allocated time lost to preemption is an upper bound on lost solver work, not a claim that saved incumbents have no value.
- Preemptions / ended started attempts, accompanied by the denominator and a descriptive interval. Pending jobs are excluded; running jobs are right-censored. This fraction is not a one-hour survival estimate: easy MIPs can finish in seconds and node/time hazards differ.
- Outcomes by node, start time, requested memory/CPUs, partition and sampled priority. A single night cannot establish a stationary preemption rate; node and day correlation also makes a naive binomial interval optimistic.

The practical decision is time to a validated result, including queue wait and retries, plus compute consumed. Prefer default when its shorter waiting time outweighs interrupted work; retain Scaglione for cases that repeatedly lose substantial search time. Do not claim a causal partition advantage from these unmatched cohorts.

## Scheduler evidence

Unicorn currently reports `PreemptType=preempt/partition_prio`, default partition tier10, `PreemptMode=REQUEUE`, no exempt time, and multifactor scheduling with a 14-day fair-share decay half-life. Partition tiers determine preemption eligibility; usage-dependent priority affects queue order and does not alone imply immunity from preemption. See [Slurm preemption documentation](https://slurm.schedmd.com/preempt.html). Preserve live samples because configuration can change.

A lack of preemptions in a small sample is not proof of safety. Reassess after enough started jobs and several nights, with larger and longer MIPs shown separately.

## Refreshing the tables

After collecting a full snapshot, run `python3 outputs/research_register/preemption_study/refresh.py <snapshot.json>`. It preserves a dated study snapshot and updates `attempts.csv`, `summary.json` and the source-hashed `latest.json`. The full collector embeds the current registry and raw accounting output.

Validation caught two Slurm details: job numbers are reused across years, so queries are restricted to user nc437 and this study's submission window; job identity retains database/SLUID/submit/start fields. Some pending array tasks have no individual accounting row yet. These remain listed as missing accounting records and are visible separately in the live queue sample; they are not treated as failed or completed attempts.

Current authoritative default arrays: **812766** (45 small cases, throttle30) and **812767** (30 larger cases, throttle20). Both retain a 3600-second solver budget and two-hour scheduler allocation. Old Scaglione arrays810588/810589 and a superseded pre-execution default launch812608/812619 were cancelled while wholly pending. They consumed no solver time and remain administrative history only.
