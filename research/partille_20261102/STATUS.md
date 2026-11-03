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

**Fractional versus integer at eight.** A separate LP checked all 173 exact trip equalities and 3,319 occupied finite station-minute limits at fleet weight eight. It succeeds with 152 fractional columns and zero plug overload. The integer search is impossible in the same archived pool. This comparison controls both the column pool and the trip/plug rules.

## Operational amendment · 2 November Pacific / 3 November UTC

At Nathan's direction, the existing 11 scoped jobs were updated in place with `scontrol ... Nice=0`: 905307, 905509, 905510, 905843, 905844, 905845, 905846, 905847, 906017, 906018 and **906021_1 only**. This supersedes the historical niceness 10000 above. The cancelled partition task 906021_0 was not changed. Ready-job priority rose from 4300–4301 to 14300–14301. Job IDs, scientific settings, phase budgets, resource allocations, dependencies and exclusion of scaglione-compute-01 were preserved; no duplicate jobs were submitted.

Live scheduler and accounting readback confirmed that blind repair 905307, witness repeat 905509, simple control 905843 and original-pool covering test 906021_1 all started at **2 November 20:03 PST / 3 November 04:03 UTC**. At 20:04 PST / 04:04 UTC, the other seven scoped jobs remained pending on their intended dependencies. Running status establishes no scientific result.

The current `run_pair.sbatch`, `gate.sbatch` and `sparse_prefix/sparse.sbatch` templates now use nice 0 for future recovery consistency and carry explicit amendment comments. The original as-submitted niceness remains in historical readbacks, deployment receipts, Git history and `sparse_prefix/SHA256SUMS`; that frozen manifest describes the pre-amendment template. Amended template SHA-256 values are:

- `run_pair.sbatch`: `a71bbd55cdde77695ae60aa4c772b0d43a1df9649e082f7687601c64502af23a`
- `gate.sbatch`: `3fa013db88db91d53da0be77a217e5a3168d288db574d2a389e054d694338eea`
- `sparse_prefix/sparse.sbatch`: `9b6613040694e8736ddfcf33076084111a66efa6bbf5fac81f33acfa0d1f47f0`

## Eight-bus repair and cluster repeat · evening of 2 November Pacific

The local repair recombined adjacent passenger trips from the original columns. It recovered eight buses serving all 173 trips exactly once with zero overloaded plug minutes, without using GIRO assignments. After root CG alone it took 4.79 s to construct, retime and check the duties. After the route dive it took 7.70 s. The graph and root CG took 4.56 h upstream. A two-minute fixed-fleet cost MIP on the augmented pool kept all eight new duties at 361.752. Its finite-pool cost bound was 238.369. This one-case retrospective test is separate from the prospective Slurm comparison. [Methods and validation](https://github.com/ndandnd/EVSP-DR/tree/3611e544b23034cfb10d2d7b1d41184461ba5c49/research/partille_20261102/diagnostic/link_recombination).

Cluster repeat 905509 completed in 91 s. The native GIRO-duty witness and augmented selection both passed replay at eight buses, 173 trips exactly once and zero overloaded plug minutes. Gurobi accepted the supplied start in 19.58 s at its objective limit. Blind repair 905307, simple control 905843 and original covering test 906021_1 were running at that check. Only the registered prospective result may release the six-chain comparison.
