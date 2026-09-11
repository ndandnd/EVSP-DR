# **Experiment register**

**Purpose:** one place to find each experiment, its settings, results, logs and limitations. Figures and interpretation remain in the other tabs; this index points to the underlying evidence.

## **Where to look**

| Question | Initialization / evidence source |
| ----- | ----- |
| Which campaigns have we run? | outputs/research\_register/REGISTER.md — campaign index with source directories and report links. |
| What happened in each case? | EVSP\_DR\_Experiment\_Register.xlsx — editable, filterable tabs for CG/LP, integer solves, charging comparisons and execution records. |
| What supports a number? | register.json and register.csv — original artifact paths, hashes, code revision, inputs and recorded settings. |
| How does this connect to April and summer? | HISTORICAL\_EVIDENCE.md and historical\_inventory.csv — 39 evidence families, with comparability warnings and missing-file notes. |
| Which failures need action? | EXECUTION\_ISSUES\_20260910.md — exact errors, solver versus scheduler status, and recovery evidence. |

Local project folder: /Users/nadan/Documents/projects/demandresponse. The register lives in outputs/research\_register. The workbook lives in outputs/01a07ecc-9b77-79b3-9782-e4308a80ba07.

## **Evidence captured**

The 11 September, 14:53 EDT snapshot contains 1511 artifact/stage records across 31 campaign/source groups. These are not 1511 independent experiments: CG, MIP, workflow and audit observations can refer to the same input. Counts change as live queue records disappear; dated snapshots are retained. The historical inventory adds 39 evidence families.

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

Completed: 75 independent fresh set-covering CG cases, array 810454, with up to 50 running concurrently. Together with the nine existing covering cases, this completes the six-chain k=2–15 comparison. Inputs, physics, CG code and budgets match the earlier partitioning runs.

Inherited-column campaign: six chains at k=2–10, including the original chains 3 and 5 and 36 additional cases across chains 1, 2, 4 and 6\. This is now a mixture of completed and active experiments, not an entirely queued campaign. The current overview and latest timestamped register identify completed results.

At the launch snapshot, reserved RAM constrained Scaglione MIP scheduling despite idle CPUs. Comparable k≤10 MIPs peaked at about 4.1 GiB; new requests of 16 GiB provide headroom and permit more simultaneous work. The GPU host compute-01 stays excluded from CPU-only jobs. Exact manifests, hashes, job IDs and dependencies are recorded in outputs/parallel\_research\_20260911 and linked from the experiment register.

### **First completed result from the parallel expansion**

The terminal-energy repair completed all three tariff cases (job 810459). Both methods require aggregate return energy of at least 280.7833 kWh. All joint solutions use five buses; optimality is for the saved, validated finite column pool, not the full routing model. The objective below is electricity plus charging-start fees, on the model grid.

| Tariff peak | Fixed-duty optimized | Joint optimized | Reduction |
| ----- | ----- | ----- | ----- |
| 08:00 | 279.8999 | 261.5698 | 6.55% |
| 12:00 | 332.0255 | 332.0255 | 0% |
| 18:00 | 232.2499 | 217.5051 | 6.35% |

Source: terminal\_energy\_fair\_mip\_retry\_5cdb813\_20260910, execution commit 5cdb813. The original failed attempt is retained. Actual return energy and continuous physical charging costs are recorded separately; this is a common minimum requirement, not identical realized terminal energy.

### **Queue verification**

Historical launch observation: 50 fresh CG tasks ran simultaneously. The 75-case fresh expansion and its MIPs have since completed, filling the 84-case covering table with the nine earlier controls.

Capacity retry history: arrays 811181/811182 were the first retry. Five remaining CG cases subsequently required a deadline/checkpoint fix and are now running in array 872397, with dependent MIPs 872398–872402. One additional retry already produced a validated one-bus result without a CG pricing certificate. The current result count is 11 of 16 pilot cells; original failures and superseded attempts remain in the register.

### **Overnight default-partition MIP trial**

Completed default-partition trial: all 75 fresh-case MIPs completed, with no observed preemptions in this cohort. Each had a 3,600-second solver budget and a two-hour scheduler allocation. This is evidence for this workload and observation period, not a guarantee against future preemption.

The preemption study records every attempt’s submit/start/end times, node, partition, resources, priority and scheduler outcome, plus result hashes. Preemption, other failures, administrative cancellations, pending jobs and still-running attempts are counted separately. Retried MIPs use new attempt paths; Gurobi’s search tree is not resumed. We will compare time to a validated result, including queue wait and retries, and compute lost to interruptions. The completed 75-case trial observed zero preemptions; retain subsequent attempts to measure how this changes with workload and cluster usage. The study is indexed at outputs/research\_register/preemption\_study/README.md.
