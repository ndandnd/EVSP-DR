# Launch status

- Remote campaign root: `/home/nc437/ladder-lite/capacity_speed_pilot_20260910_v2_7d38efd`
- Exact executable commit: `7d38efdd39857438c4a6e30b43b09e973ce51086`
- CG array: `772080`, `default_partition`, tasks `0-15%2`, 1 CPU, 24 GB, 90 minutes
- MIP array: `772082`, `scaglione`, tasks `0-15%2`, 4 CPUs, 16 GB, 30 minutes
- MIP dependency: `aftercorr:772080`
- MIP excluded node: `scaglione-compute-01`
- Submission time: `2026-09-10T20:19:16.837563+00:00`
- Initial verified state: CG tasks 0 and 1 running; remaining CG tasks throttled; MIP tasks dependency-pending

The failed first staging root
`/home/nc437/ladder-lite/capacity_speed_pilot_20260910_7d38efd` contains no
submitted jobs. It was superseded after the submission helper initially ran
without the login-shell Slurm path. The active `v2` root above is the only
campaign to collect.
