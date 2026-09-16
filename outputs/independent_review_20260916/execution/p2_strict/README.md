# F4 / P2 item11 — stricter chain5

**Submitted:31 CG jobs +31 dependent MIPs. Scientific outcome: UNRESOLVED.**

Question: does chain5 still obtain an LP fleet near30 at target31 when depot charging is slower, every bus retains15% battery energy, and vehicle groups stay separate?

| Setting | This experiment |
|---|---|
| Route21 buses (18E1) |236.44-kWh battery;35.466-kWh minimum|
| Local buses (18E2) |239.01-kWh battery;35.8515-kWh minimum|
| Starting / ending energy |Start full; end above the same15%minimum|
| Charging power |PARX60kW; other stations240kW|
| Objective |100000 per bus + flat electricity +5 per charging start|
| Service coverage |Set covering|
| Initialization |Previous same-group column pool + current singleton routes|

We keep the existing chain5 duty order. At each k, only the group receiving the new duty is solved again; its other group's result is reused. The two group schedules can be combined because they serve disjoint trips and this experiment has no shared-station constraints. At k31 their union contains716 trips. No original GIRO schedules are injected.

**This is a combined sensitivity test, not full GIRO feasibility.** It omits shared charger capacity, nonlinear charging curves, setup time, minimum charge duration, idle draw and charger-to-group compatibility. The existing capacity-speed driver adds one column per iteration, whereas the earlier chain driver adds30; runtime differences therefore cannot be attributed solely to physics. The18E2 battery follows workbook/profile239.01; the PDF rounds it to239.

## Running and queued

First independent CG jobs: **341259** (18E1) and **341263** (18E2). Both were running at the first inspection, without startup errors. Other CGs wait only for their previous same-group CG. Each MIP waits only for its own CG, allowing MIP and subsequent CG work to overlap.

All62 jobs use the default partition,8 CPUs and exclude `scaglione-compute-01`. CG receives48GB/6h, with a4h internal CG limit; MIP receives24GB/75min, with30min fleet search followed by charging optimization for the remaining one-hour budget, constraining fleet≤the best stage-one incumbent.

## Evidence

- `manifest.json`:31 frozen inputs, physics, hashes and component combinations for every k.
- `jobs.json`, `submission_verification.json`: accepted jobs and verified resources/dependencies.
- `collection_after_submission.json`: first compact result collection; absence of a result is not a scientific failure.
- `smoke/native/verification.json`: successful licensed cluster test of inherited CG, job341140.
- `IMPLEMENTATION.md`: code, validation and operational details.

Execution commit: `50ceb6c095a580f79f87b53bef536cac31f81963`. Six new inheritance tests, nine existing driver tests, and local/native inherited-CG smoke tests pass. Remote campaign: `/home/nc437/ladder-lite/review_strict_c5_20260916`; `collect_remote.py` returns hashed stage results and LP/pricing times. Keep CG certificates, finite-pool MIP proofs and covering/physical validation separate.
