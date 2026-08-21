# Cluster operating rules (Unicorn)

**Date:** 2026-08-21. Operator: Nathan (`nc437@unicorn-login-01`).
Audience: any agent or collaborator submitting cluster work for EVSP-DR.
Also recorded in the manager's persistent memory as `cluster-operating-rules`.

Operating rules for the Unicorn cluster (`nc437@unicorn-login-01`), derived from
the 2026-08-19..21 EVSP-DR campaign. Nathan asked for these to be recorded as a
handoff so any agent works the same way.

## Partition choice

**Default to `default_partition` for everything.** 14,538 CPUs / 226 nodes
versus `scaglione`'s 304 / 6 — a 48x throughput difference. Gurobi works on
both (verified). CG jobs ran 20+ hours on default_partition with **no observed
preemption**.

**Use `scaglione` only when all three hold:** runtime > ~2 h; the job cannot
checkpoint or resume so preemption loses real work; and it is on the critical
path. In practice that is only the largest pool MIPs (k30/k40).

**Anti-pattern:** short jobs on `scaglione`. At 32-48 GB each only ~4 run
concurrently there, so a 30-minute job displaces a 4-hour job that needs the
protection. This turned a 3-hour integer analysis into a 30-hour queue.

`scaglione` is Nathan's account's own nodes (`Account=scaglione`), which is why
it is not preemptible for him; `default_partition` is aggregated/borrowed.

## Mandatory sbatch flags

- **Always pass `--mem`.** `DefMemPerCPU=1000` with `CR_CORE_MEMORY` means an
  omitted `--mem` gives threads x 1 GB — a `-c 2` job silently gets 2 GB.
- **Keep `--mem <= 24G` for wide scheduling.** Most nodes are ~32 GB; only 36
  have 257 GB+.
- **`--requeue`** on anything safely restartable; give MIPs a `--progress-dir`
  so a killed job still yields its incumbent.
- `MaxTime` is UNLIMITED on both partitions (`DefaultTime` 4 h), `MaxArraySize`
  1001, `MaxJobCount` 40000, no per-user submit limit.

## Session-start check, until the pattern is known

    sinfo -o '%20P %6a %8t %6D %12m' | grep -E 'default_partition|scaglione'
    squeue --me -h -t R,PD -o '%P %t' | sort | uniq -c
    squeue -h -p scaglione -o '%u' | sort | uniq -c

If scaglione has >=4 idle nodes AND there is long unprotected work, use it for
that work only. Otherwise everything goes to `default_partition`. Let the
measurement drive the rule, not the reverse.

## Habits that repeatedly cost time

1. **Never assume a git branch name.** Run `git ls-remote --heads origin` first.
   Cursor appends suffixes (`-3a99`, `-1451`, `-b22e`) and honoured the exact
   requested name only once. Assuming cost three failed pastes.
2. **Prefer `cat > file <<'EOF'` then run the file** over long
   `bash <<'BASH'` blocks containing nested `--wrap "..."` strings. The latter
   was mangled in paste four separate times; the former never failed.
3. **One paste block per request.** Nathan asked for this explicitly, more than
   once.
4. **Throttle `sbatch` with `sleep 0.2`.** `Socket timed out on send/recv` is a
   *reply* timeout, not a submission failure — the job usually landed. Verify by
   counting queued jobs, never by the exit code.
5. **Chain dependent work at submit time** with
   `--dependency=afterany:<jobid> --kill-on-invalid-dep=yes`, so nothing needs a
   human later. Guard the dependent script so a missing input writes a
   `.skipped` marker and exits 0.
6. **Never change a checkout that running jobs read from.** Make a new one
   (`repo-wp`, `repo-n6`, `repo-g`, `repo-af`).
7. **`--wrap` batches carry no provenance.** Parameters live only in job names.
   Prefer plan-declared campaigns; if you must use `--wrap`, snapshot
   `sacct -X -P` before walking away.
8. **`bash -x` on the real invocation** is the fastest diagnosis for a silent
   Slurm failure. A bare `|| return 2` with `2>/dev/null` hides everything.

See also `STATUS_20260820.md` and `records/DECISION_LOG.csv`.
