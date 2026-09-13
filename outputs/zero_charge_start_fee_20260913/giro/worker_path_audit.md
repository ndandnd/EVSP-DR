# GIRO worker path audit — 2026-09-13

The first GIRO submission (frontier jobs `113589`–`113594`, with dependent MIP jobs `113595`–`113600`) failed before the Python worker started. Slurm executes a staged copy of the submitted shell worker under a path such as `/var/spool/slurmd/job113589/...`; the worker used `dirname "$0"` to derive `CODE_ROOT`, so it looked for `giro_zero_fee_campaign.py` below the Slurm spool directory. The six frontier jobs therefore failed at startup and the six MIPs remained unsatisfied by their dependencies. Preserve those job records and failure logs as the initial failed attempt; replacements must use unique output/job records.

The execution path contract for replacements is explicit and must not depend on the staged worker pathname:

- `EVSP_EXECUTION_REPO=/home/nc437/ladder-lite/code_pins/giro_zero_start_fee_941ed75`
- `EVSP_CAMPAIGN_ROOT=/home/nc437/ladder-lite/giro_zero_start_fee_20260913`
- the worker invokes `$EVSP_EXECUTION_REPO/scripts/event_uniform_envelope/giro_zero_fee_campaign.py`
- the worker passes `--root "$EVSP_CAMPAIGN_ROOT"`
- the detached checkout at the execution path must remain clean at commit `f895c61a525c51d32ab09276f12074217c1788ce`

A local regression test should stage or copy `worker.sh` under a spool-like temporary directory while exporting the two explicit paths, then assert that the invoked Python script is resolved from `EVSP_EXECUTION_REPO`. A worker that falls back to `dirname "$0"` is not launch-ready. This audit did not modify jobs or solver code.

Actual recovery: `worker_recovery.sh` assigns the immutable absolute code path directly to `CODE_ROOT`; the launcher exports `EVSP_CAMPAIGN_ROOT`. This implements the explicit-path contract without relying on Slurm’s staged script location. Numerical source and plan remain at f895c61.
