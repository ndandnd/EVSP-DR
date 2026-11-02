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
