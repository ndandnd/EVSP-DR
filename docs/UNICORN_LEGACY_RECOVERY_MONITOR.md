# Unicorn legacy-recovery monitor

`monitor_legacy_recovery.py` is the read-only, one-command monitor for a
preemption-damaged `cg-bigtar` task and its isolated exact-CG continuation.
It replaces hand-written `sacct | awk | while` snippets.

The important Slurm detail is that `sacct`'s `JobIDRaw` is **not** the array
identity on Unicorn. For example, old array task `867334_22` may have a raw
internal id such as `867359`. The monitor requests expanded records with
`sacct --array`, queries both fields, and extracts the task only from
`JobID=867334_22`. It prints every unsuccessful original `JobID`, task number,
state, raw id, elapsed time, and exit code rather than showing only a count.

## Normal use

Run the monitor from a **separate monitor checkout**, and point `--root` at the
active recovery checkout. While recovery job `904367` is running, never switch,
pull, update, or otherwise modify that active root; treat it as read-only.

```bash
cd "$HOME/EVSP-DR-monitor"
bash src/monitor_legacy_recovery.sh \
  --root "$HOME/EVSP-DR-legacy-recovery-bab7bfe" \
  --array-job 867334 \
  --task 22
```

The monitor automatically discovers the newest result beneath
`src/results/legacy_recovery/job867334/c*/task22/` and the matching active
`R22-*` allocation. Pin both when comparing archived observations:

```bash
bash src/monitor_legacy_recovery.sh \
  --root "$HOME/EVSP-DR-legacy-recovery-bab7bfe" \
  --array-job 867334 \
  --task 22 \
  --recovery-job 904367 \
  --result "$HOME/EVSP-DR-legacy-recovery-bab7bfe/src/results/legacy_recovery/job867334/cf31513f44007/task22/Practice_Custom_DutyUnion_k30_r2_peak18.json"
```

The default `--attestation-mode prefix` checks the migration schema and ids,
raw manifest, and SHA-256 of the immutable migrated prefixes of the current
journal and iteration CSV. This is suitable for an occasional login-node
check. For a rapid `watch`, avoid repeatedly hashing the prefix:

```bash
watch -n 300 'bash src/monitor_legacy_recovery.sh --root "$HOME/EVSP-DR-legacy-recovery-bab7bfe" --task 22 --attestation-mode structural'
```

Before publishing or copying a result, verify the complete raw migration
archive as well:

```bash
bash src/monitor_legacy_recovery.sh \
  --root "$HOME/EVSP-DR-legacy-recovery-bab7bfe" \
  --task 22 \
  --attestation-mode full
```

Use `--json` for a machine-readable campaign record. The command exits 2 only
for a `FAIL` verdict; `HEALTHY` and `WARN` exit 0 so a scientific warning does
not make an interactive monitoring loop abort.

## What it reports

- aggregate states for the original array and the exact selected array task;
- the separately submitted recovery allocation and duplicate-job detection;
- migration/attestation identity and the recorded repair class, keeping the
  immutable migration-tool commit distinct from the current continuation-code
  commit (which may legitimately be newer);
- status, journal, iteration-log, and attestation size, timestamp, and age;
- the latest immutable snapshot and its LP fields;
- the newest iteration's objective, fleet weight, artificials, reduced cost,
  and column count;
- recent iteration, objective, fleet-weight, and pool-growth rates; and
- one `HEALTHY`, `WARN`, or `FAIL` verdict with its concrete reasons.

`HEALTHY` is an operational integrity verdict, not an optimality claim. A
completed but uncertified pricing run is deliberately `WARN` even when all
files are valid.
