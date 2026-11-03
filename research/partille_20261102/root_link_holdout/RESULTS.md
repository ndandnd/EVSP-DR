# Five held-out eight-bus dispatch tests

Recombining consecutive trips and retiming the resulting duties produced two independently verified eight-bus schedules. Choosing only whole duties from the same root columns produced none within the registered budget. Eight passenger trips overlap on every case. Each successful schedule serves every trip once, respects all charger limits, and attains the minimum fleet for its timetable under the tested model.

| Chain | Recombination and independent replay | Repair elapsed | Whole-duty MIP at fleet ≤8 | MIP solve |
|---|---|---:|---|---:|
| C1 | One fixed duty failed during retiming | 10.86 s | Infeasible in the root pool | 100.69 s |
| C3 | 8 buses | 21.64 s | Time limit, no incumbent | 180.16 s |
| C4 | One fixed duty failed during retiming | 12.80 s | Infeasible in the root pool | 7.42 s |
| C5 | One fixed duty failed during retiming | 9.92 s | Infeasible in the root pool | 138.26 s |
| C6 | 8 buses | 10.33 s | Time limit, no incumbent | 180.02 s |

The three failure rows identify the first duty where the repair stopped. For C3 and C6 the whole-duty control remains unresolved. Neither of these hits demonstrates that the original pool lacked an eight-bus solution. C2, the separate development instance, already had such a proof. None of the five held-out cases provides a matched-fleet charging-cost comparison between the two methods.

The repair clock includes authentication, reading the root prefix, matching, retiming and checking the candidate. The control column gives solver time. The [native aggregate](paired_results.json) reports model build and control driver time separately. Historical graph construction and root CG took 1.99–5.05 hours on the selected attempts (median 3.99 hours). C4 also incurred an earlier interrupted attempt with 10,805 s of recorded root CG (3.00 h). Its complete elapsed time is unknown. Historical LP and pricing components are included in root CG. The study measures incremental repair on existing archives. Three roots stopped at their CG time limit. C3 and C6 were the two roots with a pricing certificate. This five-case association leaves the effect of additional pricing untested.

## What happened to the three unsuccessful covers?

Every case had an eight-path cover when only the order and timing of trips were considered. We kept the exact eight proposed duties in each unsuccessful case and tried each duty once in isolation, with no competing bus and all original charging and travel rules. Of those 24 duties, 20 produced individually replayed schedules. The fixed-sequence event graph had no reachable terminal for two duties in C1, one in C4 and one in C5. Giving those duties exclusive access to the chargers did not make them schedulable. The check does not establish which physical parameter removed the final reachable state or establish continuous-time infeasibility. [Isolated-duty results](isolated_duties/aggregate.json) · [Diagnostic code](isolated_duties/diagnose_singles.py)

The preregistered three-of-five threshold for considering k10 was missed. The next experiment at k8 should keep the eight-bus target, reject a complete duty when independent fixed-sequence pricing certifies it cannot finish, and rematch the remaining connections. Count how many alternatives need to be tested and every second spent. A charger conflict among otherwise schedulable duties requires a separate joint-retiming check.

[Registration](PROTOCOL.md) · [Editable case table](paired_table.csv) · [Input hashes](cases.json)
