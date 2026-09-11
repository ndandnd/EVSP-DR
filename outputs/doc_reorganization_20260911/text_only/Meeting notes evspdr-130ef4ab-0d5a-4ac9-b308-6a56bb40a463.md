# **Experiment register**

**Purpose:** one place to find each experiment, its settings, results, logs and limitations. Figures and interpretation remain in the other tabs; this index points to the underlying evidence.

## **Where to look**

| Question | Source |
| ----- | ----- |
| Which campaigns have we run? | outputs/research\_register/REGISTER.md — campaign index with source directories and report links. |
| What happened in each case? | EVSP\_DR\_Experiment\_Register.xlsx — editable, filterable tabs for CG/LP, integer solves, charging comparisons and execution records. |
| What supports a number? | register.json and register.csv — original artifact paths, hashes, code revision, inputs and recorded settings. |
| How does this connect to April and summer? | HISTORICAL\_EVIDENCE.md and historical\_inventory.csv — 39 evidence families, with comparability warnings and missing-file notes. |
| Which failures need action? | EXECUTION\_ISSUES\_20260910.md — exact errors, solver versus scheduler status, and recovery evidence. |

Local project folder: /Users/nadan/Documents/projects/demandresponse. The register lives in outputs/research\_register. The workbook lives in outputs/01a07ecc-9b77-79b3-9782-e4308a80ba07.

## **Evidence captured**

The 11 September, 1:54 p.m. EDT snapshot contains 1511 artifact/stage records across 31 campaign/source groups. These are not 1511 independent experiments: CG, MIP, workflow and audit observations can describe the same input. Counts change when live queue observations disappear; earlier dated snapshots are retained. The historical inventory adds 39 evidence families and explicitly lists its coverage limits.

| Keep separate | Why |
| ----- | ----- |
| LP objective and fractional buses | The weighted objective includes bus and charging costs. Sum of route weights is a different number. Fleet-only LP prototypes are labeled separately. |
| Pricing certificate and pool-MIP proof | Pricing is certified only within its recorded graph, assumptions and tolerance. A pool-MIP proof applies only to the supplied columns. |
| Original, fixed-duty and joint charging | All three arms have separate records. Original charging may be an interval when charging power within a recorded window is unknown. Match terminal energy and physics before comparing savings. |
| Solver result and execution status | A timeout, missing output field or publication error is not evidence of mathematical infeasibility. Failed and superseded attempts remain visible. |

## **Cluster policy**

**Default-partition CG arrays: request 50 concurrent tasks, or the number of tasks if fewer.** Use a smaller cap only for a documented cluster restriction or a measured resource problem. The current capacity/speed array has only 16 cases; its live throttle was raised to 50\. Slurm still decides placement from the requested CPUs and memory. True previous-k warm-start dependencies remain sequential.

CPU-only jobs exclude scaglione-compute-01 so GPU work can obtain CPUs. Held job 537227 stays held. Resource policy is recorded in the project AGENTS.md, RESOURCE\_POLICY.md and /home/nc437/ladder-lite/SCAGLIONE\_RESOURCE\_POLICY.md on Unicorn.

## **Refresh rule**

Preserve raw artifacts. Collect a dated snapshot, regenerate the register and workbook, and update interpretation only when the evidence changes. The hourly monitor follows this rule and reports meaningful completion, failure or access loss. Unknown fields remain unknown; repeated snapshots do not become new independent samples.

## **Parallel expansion — 11 September UTC**

Submitted: 75 independent fresh set-covering CG cases, array 810454, with up to 50 running concurrently. Together with the nine existing covering cases, this completes the six-chain k=2–15 comparison. Inputs, physics, CG code and budgets match the earlier partitioning runs.

Also submitted: 36 inherited-column CG jobs across chains 1, 2, 4 and 6, k=2–10. Each chain progresses independently; only successive k values within that chain must wait for their predecessor. These are queued experiments, not new numerical results.

Scaglione MIP scheduling is constrained by reserved RAM despite idle CPUs. Comparable k≤10 MIPs peaked at about 4.1 GiB; new requests of 16 GiB provide headroom and permit more simultaneous work. The GPU host compute-01 stays excluded from CPU-only jobs. Exact manifests, hashes, job IDs and dependencies are recorded in outputs/parallel\_research\_20260911 and linked from the experiment register.

### **First completed result from the parallel expansion**

The terminal-energy repair completed all three tariff cases (job 810459). Both methods require aggregate return energy of at least 280.7833 kWh. All joint solutions use five buses; optimality is for the saved, validated finite column pool, not the full routing model. The objective below is electricity plus charging-start fees, on the model grid.

| Tariff peak | Fixed-duty optimized | Joint optimized | Reduction |
| ----- | ----- | ----- | ----- |
| 08:00 | 279.8999 | 261.5698 | 6.55% |
| 12:00 | 332.0255 | 332.0255 | 0% |
| 18:00 | 232.2499 | 217.5051 | 6.35% |

Source: terminal\_energy\_fair\_mip\_retry\_5cdb813\_20260910, execution commit 5cdb813. The original failed attempt is retained. Actual return energy and continuous physical charging costs are recorded separately; this is a common minimum requirement, not identical realized terminal energy.

### **Queue verification**

At 11:22 p.m. EDT, 50 fresh CG cases were running simultaneously and 25 were pending. A sampled running case reached Gurobi LP optimization without a startup error. The 75 corresponding MIPs and all 36 warm-chain MIPs are queued with per-case dependencies.

Six capacity-test reruns are also submitted: CG array 811181 and dependent MIP array 811182\. The original failures were scheduler timeouts, with no saved resumable pools. These are fresh runs with the same code and physics, an eight-hour CG budget and nine-hour scheduler allocation. The original failures remain recorded. CPU-only jobs exclude scaglione-compute-01; held historical jobs are untouched.

### **Overnight default-partition MIP trial**

The 75 fresh-case MIPs have moved to default\_partition: arrays 812766 and 812767, with a combined limit of 50 running tasks. Each retains 3600 seconds of solver time, stage 1 capped at 1800 seconds, and a two-hour scheduler allocation for preprocessing and validation. Dependencies on the corresponding frozen pools are preserved. No running MIP was cancelled; warm-chain and capacity MIPs remain on Scaglione.

The preemption study records every attempt’s submit/start/end times, node, partition, resources, priority and scheduler outcome, plus result hashes. Preemption, other failures, administrative cancellations, pending jobs and still-running attempts are counted separately. Retried MIPs use new attempt paths; Gurobi’s search tree is not resumed. We will compare time to a validated result, including queue wait and retries, and compute lost to interruptions. No default trial MIP had started at the initial accounting check, so its preemption rate is not yet estimable. The study is indexed at outputs/research\_register/preemption\_study/README.md.
