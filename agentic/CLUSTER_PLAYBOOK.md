# Cluster playbook

Nathan pastes these. Prompt must read `nc437@unicorn-login-01`; if it reads
`nadan@Nathans-MacBook-Pro` he is on the laptop and must `ssh` first.

## Layout on the cluster

    ~/ladder-lite/
      repo        22G  campaign ll_20260820c outputs; commit 339db0ab (local-only)
      repo-wp          detached 6d02f91   } separate checkouts so running jobs
      repo-n6          detached ff047b9   } never read a tree that is changing
      repo-g           detached 780a75a   }
      repo-af          arc-flow branch    }
      finetime/ factfill/ bridge300/ warmpool/ big240c/ big240g/ cpi_sweep/
      arcflow/ chain/ finegrid_mip/ diversify/ dv_mip/ tfeas/ b5_check/
      chain.sh tfonly.sh poolmip.sh af.sh g.sh
      handoff_*/       squeue + sacct snapshots
    ~/attic/     14G  staged ghosts; contains CITED EVIDENCE, see cleanup handoff
    ~/evsp_env        Python 3.12.13 — every job's interpreter. NEVER touch.

Batch prefixes: `ft_` `fx_` fine/factorial resolution · `br_` physics bridge ·
`wp_` `b4_` `g_` warm pools · `af_` arc-flow · `cpi_` column throughput ·
`ch_` `tf_` chained analysis.

## Check state — never decide from squeue alone

    squeue --me -o '%.12i %.30j %.16P %.2t %.10M %.10L %R'
    squeue --me -h -o '%j' | sed 's/_k[0-9].*//' | sort | uniq -c
    sacct -u "$(id -un)" -S now-2days -X -P \
      -o JobID,JobName%44,State,Elapsed,MaxRSS,ExitCode | tail -40

## Harvest everything

    bash ~/ladder-lite/repo-n6/agentic/harvest_all.sh

(or copy `harvest_all.sh` from this folder to `~/ladder-lite/` and run it there)

## Interpreting results — the rules that matter

- `stop_reason=certified` and `min_rc=0` ⇒ the LP is the **model** LP optimum.
  Anything else ⇒ **no bound in either direction**; route weight is only an upper
  bound on the LP value.
- MIP `OPTIMAL` ⇒ optimal **over the columns received**. Check
  `source_cg_iterations` and `source_cg_stop_reason`. Two rows in the existing
  data read "63 and 81 buses, proven" from pools truncated at 140 and 80
  iterations — meaningless as model results.
- `⌈certified L⌉ == validated incumbent` ⇒ **discrete-model optimum proven**
  (sandwich). Use before any expensive solve.
- Integer results are **non-monotone in grid resolution**. Never present them as
  a monotone function of the approximation knob.

## Resubmit patterns

Chain analysis onto a finished CG cell (feasibility + pool MIP):

    sbatch --requeue -J ch_X -c 2 --mem=32G -t 3:00:00 -p default_partition \
      ~/ladder-lite/chain.sh <cg_status.json> <target_k> ~/ladder-lite/chain

Feasibility only, no MIP rerun:

    sbatch --requeue -J tf_X -c 2 --mem=32G -t 1:00:00 -p default_partition \
      ~/ladder-lite/tfonly.sh <cg_status.json> <target_k> ~/ladder-lite/chain

Chain a not-yet-finished CG job at submit time:

    --dependency=afterany:<cg_jobid> --kill-on-invalid-dep=yes

## Partition choice

Default `default_partition` (14,538 CPUs, 226 nodes; no preemption observed over
20+ hours). Use `scaglione` (Nathan's own nodes, 304 CPUs, 6 nodes) **only** for
jobs longer than ~2 h that cannot resume. `--progress-dir` was audited and is
**observational only** — it restores neither incumbent nor tree, so a relaunched
MIP repeats from root. That is why large pool MIPs still belong on `scaglione`.

**Always pass `--mem`.** `DefMemPerCPU=1000` means an omitted `--mem` gives
threads × 1 GB. Keep `--mem ≤ 24G` for wide scheduling; most nodes are ~32 GB.

## Disk

Was 383 GB, now ~190 GB. Remaining bulk is primary column journals, which gzip at
**18.6×**. See `../CLUSTER_CLEANUP_HANDOFF_20260821.md`. **Gate compression on
chain completion** — `run_exact_pool_mip.py` and `target_pool_feasibility.py`
read the raw path from the status JSON and will not open a `.gz`.

## Known failure modes and their fixes

| symptom | cause | fix |
|---|---|---|
| all tasks exit 2 in ~8 s, empty logs | `BASH_SOURCE` resolves to the Slurm spool copy, not the repo | pass the repo root explicitly |
| `commit mismatch` on every task | plan pins a commit that HEAD no longer matches | rebuild the plan, or check out the pinned commit |
| `plan or matrix already exists` | `plan.sh` writes to `$LL_ROOT/campaign` regardless of `LL_CAMPAIGN` | `mv` the old campaign dir aside first |
| `Socket timed out on send/recv` from sbatch | **reply** timeout, not submission failure | verify by counting queued jobs, never by exit code; add `sleep 0.2` |
| `unsupported fixed-duty diagnostic grid` | `--initial-pool greedy` needs a commensurate grid | use `warm_pool_fixed_duty_optimizer` path, or a whitelisted grid |
| chain writes `.skipped` | CG output or journal missing | expected for failed CG; check the CG `.err` |
