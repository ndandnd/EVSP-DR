# Completed capacity deadline pilot

## Capacity deadline recovery — all five saved pools and MIPs completed

Snapshot 20260911T211448Z: all five CG tasks 872397 and dependent MIPs872398–872402 completed at scheduler level. CG stopped at the cooperative pricing_deadline after 28,800 seconds; none has a pricing certificate. Atomic pool checkpoints are saved, unlike the earlier lost runs. All five MIPs prove fleet and charging optima only within their tiny saved pools and pass the shared-station capacity sweep; route feasibility is by exact-event construction, not an independently claimed full continuous replay.

| Case | Final RMP route weight | Saved columns | Integer buses | CG completed iterations |
|---|---:|---:|---:|---:|
| duty13406, capacity | 1 | 34 | 1 | 20 |
| k2, capacity / combined | 2.8 | 36 each | 3 each | 13 each |
| k3, capacity / combined | 16 | 38 each | 16 each | 3 each |

The k3 final weighted RMP objective is 1,600,127.456, not a certified full-model lower bound. In its third completed iteration, capacity-only pricing consumed 25,722.65 seconds (7.15 hours), while that iteration's LP took 0.00563 seconds. Combined-arm pricing took 24,797.47 seconds. The following pricing call hit the deadline. This is concrete evidence that pricing—not the tiny saved-pool MIP or LP solve—is the computational bottleneck in these pilot cases. It does not prove that capacity requires sixteen buses. The k3 selected solution has three duplicate-covered trips; removal is not separately validated.

All 16 original pilot cells now have MIP outcomes across original and recovery attempts. Ten original CG completions were certified; six recovered cases remain uncertified. No new campaign or unchanged retry was submitted. Next recovery decision: profile capacity-aware pricing before spending another eight hours on the same search; keep the identity-bound saved pools. Evidence and full hashes: outputs/parallel_research_20260911/capacity_deadline5_completed/.
