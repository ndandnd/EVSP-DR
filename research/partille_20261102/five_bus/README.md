# Five-bus corrected-transfer comparison

## Primary result

Restoring the directed charger transfer makes the original passenger assignment infeasible in the frozen single-hour, exact-terminal-floor model. The recorded schedule reprices to 230.644788 units under its historical reconstruction, which tolerates rounded energy records. The alternative assignment has a replayed 164.910539-unit schedule and a restricted MIP gap of 0.00530%. A three-stage percentage under these exact rules is undefined.

The raw movements establish 1 minute and 0.400000006 kWh inbound in 73 observations, with zero time and energy outbound in 81 observations. Both inherited assignments traverse this directed connector 28 times. Regeneration uses only the passenger sequence and candidate station path. Each optimization starts cold.

Allowing continuous charging across tariff boundaries while retaining exact nominal terminal floors still leaves the original assignment infeasible. A single-duty upper bound shows why. Give each bus every available charging minute, ignore other buses, remove idle consumption, and apply the physical taper. One duty still falls 0.00000863014 kWh below its nominal terminal floor. Confining each session to one hour raises that shortfall to at least 11.119265 kWh. The alternative assignment costs 163.819028 with crossings and the nominal floors. These are restricted-model conclusions.

## Sensitivity to reconstructing rounded recorded energy

For this separate comparison, a recorded charge delivers the lesser of the reported kWh and the amount physically deliverable within its recorded timestamps, at the documented power curve. Each duty's resulting terminal energy sets its exact floor. Both optimized arms permit one continuous session per opportunity across hourly price changes and pay the price applicable to each portion of delivery. All three use the same directed charger movements, battery physics, tariff, per-duty floor set and cohort plug limits.

| Charging and assignment | Cost in illustrative tariff units | Starts | Minimum SOC, kWh | Reduction from previous stage |
| --- | ---: | ---: | ---: | ---: |
| Recorded activities, reconstructed delivery | 230.644778 | 52 | 39.824607 | — |
| Recorded passenger assignments, reoptimized charging | 178.335809 | 47 | 35.466000 | 22.67945% |
| Saved alternative assignments, reoptimized charging | 163.819022 | 42 | 35.466000 | 8.14014% |

Total reduction from the reconstructed recording is 28.97345%. Nineteen of 52 recorded sessions required a cap larger than 0.00000001 kWh. The total adjustment was 0.0000548119 kWh, and the largest was 0.0000120684 kWh. Individual terminal floors declined by 0.00000400 to 0.00001327 kWh. The two restricted optimality gaps were 0.00372% and 0.00730%.

Independent replay checked every original service ID, service endpoints and time, connector timing and energy, charge duration and SOC-dependent power, hour-by-hour costs, minimum reserve, capacity, individual terminal requirements and shared charger intervals. Every sensitivity arm covers 62 distinct source trips exactly once. Each uses at most one plug at each opportunity site. All three return 250.718270 kWh across their five buses. Largest adverse individual terminal residual is below 0.000000000007 kWh.

Battery capacity is 236.44 kWh, reserve 15%, idle draw 0.1 kW, minimum charging 3 minutes, and setup time zero. Both opportunity sites use the 18E1 taper. PARX is modelled at 60 kW with unlimited plugs and receives no charge in these schedules. Energy prices follow the same illustrative 08:00 peak with zero charging-start fee. The recording preserves its actual movements and parking. Reoptimized buses wait along the frozen recovered station paths, with static reference depot travel. Results apply to this cohort with these candidate visits. They provide no fresh assignment optimization, full-network certificate, fleet-wide occupancy or driver/platform feasibility result.

## Evidence and reproduction

The three audit JSONs retain the primary result, crossing-only control and reconstructed-floor sensitivity separately. The execution receipt pins frozen inputs and code, model exports, solver logs, private witnesses and independent replay files. Each model had two local CPU threads, a 600-second limit, seed zero and relative MIP gap 0.0001. All finished before the limit. No cluster submission or dependency was needed.

The frozen factorial producer is commit `b931772dadf1e5930fd87bf4a89edd7dcaf6c51a`. The local correction code commits were `ba9cbb50f66047400ea0aa657f8c2ec5c56e3cae` (literal connector), `8f2ddae41b091ec3362fdeab850c9c5d9c97e9ef` (cross-hour), `c52752d81b498ee7cc86d3cedc915750d14fe1ca` (clock-capped run and final physical verifier) and `224edc118b00b182e2973fdb9e984c48e70f80de` (aggregate receipt). Hashes identify the exact standalone scripts and distinguish them from the historical solver.

Reproduction requires authorized access to the frozen factorial bundle and October physical checker in the research checkout, Python 3.12, NumPy and Gurobi. The release contains code, aggregate metrics and hashes. Licensed records, per-duty schedules and solver models belong in a private directory. Set ROOT to the absolute path of the source research checkout and PRIVATE and RELEASE to new output directories, then run the following from the code directory.

```sh
mkdir -p "$PRIVATE/multihour" "$PRIVATE/clock_capped/multihour" "$RELEASE"
python3 solve_five.py "$ROOT" "$PRIVATE" recorded
python3 solve_five.py "$ROOT" "$PRIVATE/clock_capped" recorded --clock-capped-targets --across-hours
for arm in original saved_joint_fee0; do
  python3 solve_five.py "$ROOT" "$PRIVATE" "$arm"
  python3 solve_five.py "$ROOT" "$PRIVATE/multihour" "$arm" --across-hours
  python3 solve_five.py "$ROOT" "$PRIVATE/clock_capped/multihour" "$arm" --across-hours --clock-capped-targets
done
python3 verify_five.py "$ROOT" "$PRIVATE" "$RELEASE" ''
python3 verify_five.py "$ROOT" "$PRIVATE" "$RELEASE" multihour
python3 verify_five.py "$ROOT" "$PRIVATE/clock_capped" "$RELEASE" multihour
python3 build_receipt.py "$ROOT" "$PRIVATE" "$RELEASE"
```
