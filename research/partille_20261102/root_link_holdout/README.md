# Testing whole duties against recombined trip connections

Start with the [registered protocol](PROTOCOL.md) and [case manifest](cases.json). Chain 2 provided the development example. The other five chain instances contribute the test outcomes.

## Reproduce

Each manifest binds a dated owner summary and a byte-counted root prefix to the audited execution source, physical inputs and station powers. Obtain the licensed original sources through the private experiment register. The public repository contains scientific code and aggregates. Owner summaries do not independently record every historical executable hash. Run-time checks authenticate the pinned replay source used in this comparison.

Place the existing [independent selection checker](../diagnostic/validate_selection.py) beside the drivers, as in the cluster deployment. Each invocation gets a new output directory. Both wrappers store raw schedules and Gurobi logs in private storage. The repair receipt records an expected scientific miss independently from the Slurm exit code.

```sh
python run_recombination.py --manifest case.json --output new-repair-attempt --retime-seconds 180 --start-fee 5
python root_control.py --manifest case.json --output new-control-attempt --start-fee 5
```

The development regression recovered the original root-only result, eight buses for 173 trips, no duplicate trips or plug overload, and modeled nonfleet cost 374.688. A corrupted summary fingerprint was rejected before matching. Two small MIP controls verified feasibility with two available plugs and infeasibility with one plug for simultaneous mandatory charges.
