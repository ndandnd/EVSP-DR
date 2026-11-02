# Experiment status

**2 November 2026.** The five-bus corrected-transfer sensitivity and fixed-duty eight-bus witness are verified locally. The local augmented-pool MIP was run on the authenticated original 39,390-column file. See the adjacent reports and hashes. The blind search has a separate Slurm allocation.

| Work | Slurm job | Release condition |
| --- | ---: | --- |
| C2 k8 fresh repair-pricing pilot | 905307 | Queued on default |
| Independent repeat of fixed-duty witness and augmented MIP | 905509 | Queued on default |
| Verify unseeded pilot output | 905510 | After terminal 905307 |
| Fresh simple-dive C2 paired control | 905843 | Queued on default |
| Decide whether to replicate at 8 | 905844 | Inspect actual receipts after 905509 and 905510 |
| Five remaining chains × two fresh arms at 8 | 905845 | After successful eight-bus gate |
| Decide whether to scale to 10 | 905846 | After eight-bus comparisons and pilot control terminate |
| Six chains × two fresh arms at 10 | 905847 | After successful ten-bus gate |

The pilot, its independent repeat and the control were waiting for scheduler priority when checked. The downstream dependencies are intentional. Slurm readback confirmed the default partition, niceness 10000 and exclusion of the reserved GPU node. Fresh arms request 8 CPU cores, 48 GiB and at most 20 hours. No new solution is attributed to queued jobs.

Two sparse-prefix arrays are registered on the same release gates. Job **906017** uses k5 → k8, job **906018** uses k5 → k8 → k10. Each array has six independent chain tasks. Each task pays for its own predecessors inside the same 20-hour/8-CPU ceiling. It regenerates predecessor-selected trip sequences on the receiver graph, with new singleton routes and new physical replay. This tests sparse-prefix initialization, distinct from importing every old column at every k. See the [design, code, tests and Slurm receipts](sparse_prefix/README.md).

**Original-pool diagnostic.** Four distinct peaks each require eight simultaneous trips. Every route in an eight-bus solution must cover one trip at each peak. This removes 12,108 of 39,390 routes without excluding any feasible eight-bus solution, leaving 27,282. At a 60-second limit, neither the exact-once nor the covering feasibility model found a selection or proved infeasibility. [Exact-once receipt](diagnostic/original_pool/partition_60s.json) · [Covering receipt](diagnostic/original_pool/cover_60s.json). Two longer, 2-hour tests are registered as job **906021**, tasks 0 and 1. Their outcomes, including a timeout, will be reported independently of augmented-pool success.

**Superseding the exact-once timeout.** Two complete searches independently proved the original pool has no eight-bus partition even after relaxing away the plug rows. The exact-once repeat (906021_0) was therefore cancelled while pending. The separate eight-bus covering search (906021_1) remains registered. [Proof and reproducibility](diagnostic/exact_cover/README.md).
