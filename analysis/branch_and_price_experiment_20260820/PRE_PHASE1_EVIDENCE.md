# Pre-Phase-I branch-and-price evidence (invalidated bounds)

Date: 2026-08-20.

An independent review found that the first branch-and-price driver could prune a
node when finite-penalty artificial variables remained positive.  A finite
Big-M optimum is not an infeasibility certificate.  Therefore:

- every lower bound, gap, `proven_optimal` value, and artificial-based prune
  produced by that driver is invalid and must not be cited;
- physically replayed integer partitions remain valid feasible upper bounds;
- completed-run counters and wall times remain performance observations only.

## Completed local artifacts

The raw files remain ignored local artifacts under
`results/branch_and_price/runs/`. Their hashes bind this inventory.

| cell | grid | valid feasible fleet | valid feasible cost | nodes | pricing solves | wall s | JSON sha256 |
|---|---|---:|---:|---:|---:|---:|---|
| k02_s2 | 15/10 | 3 | 300070.5920 | 301 | 56611 | 711.820 | `f40d41828034f59dcb1fd18270c76436a7e3e1e691cd5d0c20c1ea61520fdb91` |
| k02_s1 | 1/5 | 2 | 200041.6448 | 11 | 1991 | 4023.305 | `1118c497b4ae6e853e68130e43c228924dc3a05a4351fbda15dbbf33a5b6fec1` |
| k02_s2 | 1/5 | 2 | 200129.4400 | 1 | 88 | 48.798 | `151a9a7aacb791fdc19d14e38632519df813caefdec70de8867cec6b415e72ba` |
| k02_s3 | 1/5 | 2 | 200067.2000 | 1 | 274 | 420.438 | `0b22916d2a1a3ffa3e6e78e0e7daba26a1f4bee7d893d9a1eea7f86857231ddd` |
| k03_s3 | 15/10 | 3 | 300105.3520 | 175 | 102168 | 3013.511 | `5284ff8a6a6b0776536b91801538a7194aefe4766c2573ae8e1eba7c69b0bd95` |
| k03_s1 | 1/5 | 3 | 300088.8848 | 3 | 955 | 2281.189 | `25e5ec1c84e02cf971cf3e4b3751df02f67723dc4540b4077c85a193167790aa` |

The selected routes embedded in those JSON files passed
`validate_final_selected_routes` before publication.  The `nodes`,
`pricing solves`, and `wall s` columns are retained as raw performance
measurements; the old driver did not durably record pure pricing time per node.

## Interrupted-run upper-bound observations

These incumbents also passed physical replay in-process, but interruption
occurred before their route payloads were published.  They are retained as
non-reconstructable numerical upper-bound observations, not durable witnesses.

| cell | grid | feasible fleet | feasible cost | source |
|---|---|---:|---:|---|
| k02_s1 | 15/10 | 4 | 400090.4720 | root-pool SciPy MIP |
| k02_s3 | 15/10 | 7 | 700085.4720 | root-pool SciPy MIP |
| k03_s1 | 15/10 | 5 | 500155.6480 | root-pool SciPy MIP |
| k03_s2 | 15/10 | 10 | 1000148.0880 | root-pool SciPy MIP |
| k03_s2 | 1/5 | 3 | 300151.8400 | root-pool SciPy MIP |
| k05_s1 | 15/10 | 70 | 7000128.7440 | root-pool SciPy MIP |
| k05_s2 | 15/10 | 5 | 500255.8800 | integral Ryan--Foster node |
| k05_s3 | 15/10 | 113 | 11300042.7360 | root-pool SciPy MIP |

No scientific conclusion is drawn from this file. Corrected searches require a
certified artificial-mass Phase I, an artificial-free fleet Phase II, durable
node ledgers, and resumable checkpoints.
