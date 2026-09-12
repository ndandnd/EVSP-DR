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

The 10 September, 10:08 p.m. EDT snapshot contains 989 artifact/stage records across 26 campaign/source groups. These are not 989 independent experiments: CG, MIP, workflow and audit observations can describe the same input. Counts change when live queue observations disappear; earlier dated snapshots are retained. The historical inventory adds 39 evidence families and explicitly lists its coverage limits.

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