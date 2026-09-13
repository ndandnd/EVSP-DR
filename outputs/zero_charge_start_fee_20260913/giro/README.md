# GIRO zero charge-start fee deployment

The detached execution pin is `f895c61a525c51d32ab09276f12074217c1788ce` at `/home/nc437/ladder-lite/code_pins/giro_zero_start_fee_941ed75`. The prepared campaign root is `/home/nc437/ladder-lite/giro_zero_start_fee_20260913`; its six-cell `plan.json` is copied to `remote_plan.json` and hash recorded in `deployment.json`. The audit prepared the campaign; the manager then submitted and repaired its Slurm wrapper. Current jobs are `114100`–`114111`, recorded in `jobs.json`; `jobs.initial_failed.json` retains the initial six pre-solver failures and six MIPs cancelled before starting. The replacement worker is outside the immutable code checkout and uses its absolute path.

The campaign has fresh fee-0 and fee-5 fixed-duty frontiers for peak08, peak12, and peak18. Each joint MIP uses the same tariff's two frontiers plus the saved pool repriced to its destination fee, with the aggregate `expanded_grid_terminal_soc_kwh` target `280.7833253` kWh and continuous replay validation. MIPs use `default_partition`, 8 CPUs, 48 GiB, a two-hour allocation, and a 3600-second two-stage optimization budget; only `scaglione-compute-01` is excluded.

Run the compact collector after results arrive:

```sh
/home/nc437/evsp_env/bin/python -u /home/nc437/ladder-lite/code_pins/giro_zero_start_fee_941ed75/scripts/event_uniform_envelope/giro_zero_fee_campaign.py collect --root /home/nc437/ladder-lite/giro_zero_start_fee_20260913
```

`commands.json` contains the six frontier and six dependent MIP worker commands. `slurm_commands.json` is the remote no-submit launcher dry run; it verifies the six frontier and six MIP `sbatch` commands use `default_partition`, exclude only `scaglione-compute-01`, and request two-hour allocations. `summary.json` from the collector will omit selected route arrays and any large LP payloads.
