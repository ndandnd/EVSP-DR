# Fixed-pool MIP pilot: all 25 endpoints complete

One scoped scheduler snapshot at **2026-09-22 07:56:15 UTC** found all 25 active trials COMPLETED. All five preparations and 25 endpoint receipts are valid. **738 checks pass**: original ordered-pool/matrix hashes, source and manifest pins, recorded physical admission, intended starts and selected-route identities, exact model dimensions, settings, full native-log hashes, objective/bound, termination and timing checks. Fifteen trials prove their finite-pool fleet optimum; ten reach their 1,800-second optimizer allowance normally. No recovery or new submission is required.

[All 25 editable results and provenance](results.csv) · [Verification and source hashes](verification.json) · [Independent source/start audit](source_identity_audit.json) · [Scheduler snapshot](scheduler_snapshot.json) · [Full solver logs and completed artifacts](../../mip_structure/collections/20260922T075615Z/receipt.json).

## Fleet outcomes: incumbent / bound

A ✓ marks a finite-pool proof. Every k8 optimum is 9, above target 8. All sequential C1 k15 arms attain and prove target 15. Neither fresh k15 pool reaches target 15 in this pilot.

| Pool | Default | MIPFocus1 | MIPFocus2 | PreSparsify1 | Offline saved start |
|---|---:|---:|---:|---:|---:|
|C1 k8 fresh|9 / 9 ✓|9 / 9 ✓|9 / 9 ✓|9 / 9 ✓|9 / 9 ✓|
|C4 k8 fresh|9 / 9 ✓|9 / 9 ✓|9 / 9 ✓|9 / 9 ✓|9 / 9 ✓|
|C1 k15 fresh|18 / 15 open|18 / 15 open|18 / 15 open|19 / 15 open|18 / 15 open|
|C3 k15 fresh|18 / 15 open|18 / 15 open|18 / 15 open|17 / 15 open|17 / 15 open|
|C1 k15 sequential|15 / 15 ✓|15 / 15 ✓|15 / 15 ✓|15 / 15 ✓|15 / 15 ✓|

## Actual optimizer wall seconds

Times exclude physical preparation, artifact loading, model construction and original saved-incumbent acquisition. † denotes time-limit termination, rather than time to proof. Source CSV retains full precision, final Gurobi Runtime, native log explored time/work, and all loading/building times separately.

| Pool | Default | MIPFocus1 | MIPFocus2 | PreSparsify1 | Offline saved start |
|---|---:|---:|---:|---:|---:|
|C1 k8 fresh|1232.992|1472.540|849.588|984.828|1388.307|
|C4 k8 fresh|461.957|700.653|511.809|756.059|601.274|
|C1 k15 fresh|1800.072†|1800.126†|1800.357†|1800.065†|1800.086†|
|C3 k15 fresh|1800.032†|1800.045†|1800.035†|1800.031†|1800.023†|
|C1 k15 sequential|104.168|105.088|80.752|1779.487|11.150|

## Interpretation and limits

- **Solver settings have mixed effects.** MIPFocus2 proves C1 k8 in 849.588s versus default 1232.992s (31.1% less time), but on C4 k8 default is fastest (461.957s). All five arms prove 9 on both pools; no setting creates an 8-bus route cover in those frozen pools.
- **PreSparsify1 is not a general improvement.** It improves the C3 k15 incumbent from 18 to 17 at the same allowance, but worsens C1 k15 fresh from 18 to 19. Sequential C1 k15 still proves 15, taking 1779.487s versus default 104.168s (17.08×). These are one-seed, different-node observations.
- **The saved start has limited scope.** Sequential C1 k15's saved 15-bus incumbent proves in 11.150s versus ordinary greedy-start default 104.168s. The ordinary-start MIPFocus2 run proves 15 in 80.752s. Saved-start acquisition occurred in an earlier run and is excluded here. C3's saved 18-bus start later improves to 17; C1 fresh's saved 18-bus start remains 18. A precomputed incumbent does not resolve either fresh 15-bus target.
- **Finite-pool scope is unchanged.** Binary set covering, unit fleet objective, no new route generation, no applied dominance/screening reductions, no charging MIP or added capacity constraints. Physics remains 240 kWh battery/initial SOC, constant 240 kW charging, zero reserve, no terminal floor/shared capacity, original flat tariff/start fee 5. Duplicate passenger coverage can occur. Native physical admission/ordered-pool evidence was authenticated; this monitor does not independently replay every physical route.
- **Timing revisions are explicit.** The 15 original trials retain their slow NPZ loader; the 10 replacement trials use the validated once-only loader. Report optimizer times for this comparison, not lifecycle totals across revisions. Final Gurobi Runtime exceeds the native log's rounded “Explored…seconds” line by at most 0.277 s; both are retained, and measured optimize wall agrees with final Runtime within 0.01 s. Time-limit wall overshoot is at most 0.357 s.

Original failed preparations 729457/729775 and canceled loading/pending attempts remain preserved in the campaign. No historical holds were changed, no broad queue check was made, and no new jobs were launched. All five case scans previously completed with zero duplicate-incidence columns, zero redundant covering rows and one connected component; those diagnostics did not alter these trials.
