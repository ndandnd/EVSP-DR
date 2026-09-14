# Frozen chain continuation k26–28

Six existing chains continue their already frozen duty order; no new random selection. The18 inputs have564–674 trips. Exact input hashes, stable trip-ID nesting and unchanged previous-trip attributes are checked in `input_validation.json`. The unchanged full generation manifest remains under `inputs/manifest.json` (its original launch-through25 metadata is historical; this campaign stages only26–28).

Remote root: `/home/nc437/ladder-lite/chain_extension_20260914`; storage `/share/scaglione/nc437/evsp-dr/chain_extension_20260914`. Schema `evsp-chain-extension-launch-v1`; sourceCGa0e0bb7681c8451e3cbbbfa06aef390026d9af4b, MIP871d057e1067411f09581e37d78f7c1ca43f68bb. Source, static data, tooling and parent hashes are recorded in `manifest.json`.

Native validation187907 passed actual CLI parsing, unrestricted Gurobi optimization, instrumented/plain graph hash equality, cache-required inherited CG and two-stage MIP physical replay. The independent frozen previous-input gate was reviewed before production; the earlier prelaunch manifest is preserved with an explicit amendment trail. `campaign.diff` compares the prior wrapper.

Launched graph array187967 (18 tasks, throttle18),18 sequential CG jobs and18 independent-after-CG saved-pool MIPs. `case_jobs.json` is the exact map. The first CGs forC1/C4/C5/C6 retain old pending k25CG dependencies;C2/C3 use authenticated already-published k25 artifacts. Every subsequent CG depends on its own graph and previous CG; no MIP feeds CG. Published result/journal gates validate hashes, producer commit and frozen previous input identity.

All CPU jobs use default_partition and exclude scaglione-compute-01. Requests: graph2CPU/64G/12.5h, CG8CPU/96G/5h (4h nativeCG budget), MIP8CPU/24G/2h (1h nativeMIP). Graph watchdog12h retains the existing allowance below the scheduler12.5h ceiling. A prior750-trip graph hit12.5h, so graph completion remains uncertain. Progress logs are retained; algorithmic/watchdog failures are not blindly requeued. Scheduler preemption creates unique attempts; CG resumes supported checkpoints, MIP tree restarts, incomplete graph rebuilds.

`launch_verification.json` records all54 effective scheduler states, exclusions, memory and dependencies at verification. All54 were pending at that sample:18 graph tasks eligible,36 downstream jobs dependency-waiting. Native validation is not a research result. `mip_registry_additions.json` contains18 cohort records for the root agent’s locked shared-registry update; this worker did not edit the global registry.
