# Pinned cleanup recovery: MIX-1 two-price split

Post498962 failed at the per-route10-duplicate enumeration guard. The selected
source is `ROOT2/results/mix1_two_price_split/fresh_cg/job499021_r0/out/summary.json`.
Its per-route duplicate counts are10,1,0,2,11: at most3,079 subsequences in total.
This recovery admits11 only for that exact source and selected-route hash. It
changes no enumeration, charging frontier, physics, objective, MIP budget or
solver setting. No new CG, GIRO columns or subset heuristic is introduced.

`cleanup_case11.py` verifies the original module, summary, selected routes and
all input hashes, then executes the original module with one byte changed:
`len(duplicated)<=10` becomes `len(duplicated)<=11`. Imports come from the frozen
checkout at4a8b497e. The checkout is never edited. The exact diff is in
`guard_only.diff`; `verification.json` records compilation and hashes.

`recovery_worker.py` preserves the collector layout
`ROOT2/results/mix1_two_price_split/cleanup/job<J>_r<R>/attempt.json` and `out/`.
It locks this case, refuses a finished cleanup, preserves every old attempt,
and records source/input/code/output hashes and restart. A successful process
remains separate from existence of a selection, exact-once proof and shared
capacity validation. Shared capacity remains unmodeled.

Resources stay8 CPUs,48G,5h on default_partition, with requeue and exclusion of
scaglione-compute-01. Failed498962 used2,637,800K MaxRSS in16:58; successful same
cohort posts498951/498960 used15,076,124K/18,249,744K in2h17/2h14. Those measurements
support preserving the original allocation. Frontier generation is serial and
not covered by the final MIP's3,600-second limit; the Slurm5h limit still applies.

Deploy these files together under `ROOT2/recovery_20260926_case11/`. The sbatch
template does not submit itself. Before submission, the owner checks live queue,
remote policy, absence of replacements, and records dependencies/source receipts.
Both original predecessors are terminal. The worker's filesystem lock supplements
the queue check; it is not a pending-job deduplication ledger.

Verification: `python3 -m unittest discover -s <recovery-directory> -p test_case11.py -v`.
The tests build no optimization model. Remote input preflight is
`python cleanup_case11.py --source <pinned-summary> --source-sha256 <pinned-hash> --out <unused-output-path> --check-only`;
it writes nothing and imports no solver. Source hashes are constants in the wrapper.
