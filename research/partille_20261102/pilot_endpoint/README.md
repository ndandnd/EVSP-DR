# C2 k8 prospective pilot endpoint · 3 November 2026

Both runs selected nine buses in their covering masters. The repair arm served 173 distinct trips 194 times, including 21 surplus services. The simple-dive control served the same trips 184 times, including 11 surplus services. Both schedules passed the recorded route, energy, cost and minute-plug replay checks. Neither delivered an exactly-once eight-bus schedule.

| Measure | Integer-directed repair | Simple dive |
|---|---:|---:|
| Slurm elapsed time | 8 h 32 min 5 s | 6 h 42 min 18 s |
| Root CG time | 10,810 s | 7,060 s |
| Root columns | 26,085 | 28,253 |
| Full-graph weighted LP certificate | None | 800,236.677061 |
| Columns after the integer treatment | 26,220 | 41,340 |
| Final selected routes added after the root | 3 of 9 | 6 of 9 |
| Final pool fleet incumbent / bound | 9 / 8 | 9 / 8 |
| Selected grid charging cost, including start fees | 254.432 | 239.512 |
| Overloaded station-minutes | 0 | 0 |

Repair pricing made 89 calls across 16 neighborhoods and added 135 columns. Fifteen repair MIPs produced three accepted improvements. All three retained nine buses. The last neighborhood reached the time limit. The simple dive fixed three routes before its time limit. Both final fleet and charging MIPs also reached their time limits.

The two arms used the same data, physics, fresh initialization and nominal stage limits. The CPU models differed, as did the root pools that reached the integer stage. The observed cost and elapsed-time differences therefore do not isolate the effect of repair pricing.

The solver jobs finished normally. Native verification rejected the blind selection for duplicate service, reporting exit 3. The registered k8 and k10 gates both returned scientific hold, exit 42. All 34 conditional follow-up tasks were cancelled before starting. The existing eight-duty diagnostic witness remains valid. The prospective comparison supplies no basis to release or rerun those tasks.

The accompanying JSON contains the source-digest checks, exact wall times, native stage outcomes and endpoint chronology. Its scheduler timestamps use America/New_York local time.
