# F6 — three charging comparisons, starts and battery levels

**P0 item 5 completed.** All figures below are for the same five GIRO duties (62 trips), five buses, zero charge-start fee, 240-kWh batteries and uniform 350-kW charging. The model imposes no minimum SOC reserve, 3-minute minimum charge, setup time or shared-station capacity. This differs from the 240-kW chain campaign.

| Tariff peak | Original GIRO schedule: cost interval | Fixed GIRO trips: charging optimized | Fresh CG + exact-once cleanup | Starts: original / fixed / CG | Windows under 3 minutes: original / fixed / CG |
|---|---:|---:|---:|---|---|
| 08:00 | 230.287–230.981 | 128.293 | 124.692 | 52 / 46 / 44 | 0 / 6 / 3 |
| 12:00 | 289.594–290.597 | 164.231 | 160.605 | 52 / 43 / 46 | 0 / 3 / 6 |
| 18:00 | 223.447–223.723 | 95.285 | 88.296 | 52 / 42 / 44 | 0 / 2 / 6 |

Costs are tariff cost units. Original GIRO supplies an energy amount and a charging window, not its within-window power trace; the original invoice is therefore an interval, not a uniquely observed number. The optimized costs use continuous physical replay. All 45 route records (three tariffs × three arms × five buses) pass the pinned physical validator and cover their input trips exactly once. Detailed values are in `comparison.csv`; every bus's minimum and ending SOC is in `per_bus_soc.csv`, with reconstructed trajectories in `soc_traces.json`.

**The comparisons still use relaxed operating rules.** Minimum SOC across original buses is 45.515 kWh (18.96%); it is approximately zero for the fixed arm at peaks 08/12 and for the CG arm at all three peaks. Fixed peak18 reaches 1.540 kWh (0.64%). Some optimized charge windows last only **25.714 seconds**, while the shortest original window is 240 seconds. These results do not establish feasibility with a 15% reserve or 3-minute minimum.

Aggregate ending energy is **280.7833 kWh original**, versus **281.1700 kWh in both optimized arms** at peaks 08/12 and **282.9000 kWh in both optimized arms** at peak18. Thus fixed-versus-CG comparisons match ending energy, while original-versus-optimized comparisons share an aggregate minimum but have slightly different realized ending energy. There is no per-bus terminal target, and battery energy can be distributed very differently among the five buses. Source-activity SOC and production-deadhead SOC differ slightly; both are retained for original GIRO rather than silently mixed.

The CG column is the **fresh zero-fee full CG followed by duplicate-trip deletion and charging reoptimization**, from `terminal_duplicate_cleanup_20260916`. It is not the older `joint_pool_optimized` entry in the source comparison JSON. That older entry repriced saved columns and equalled fixed-duty performance; using it here would substitute the wrong experiment.

## Finding verdicts

| F6 claim tested | Verdict | Scope |
|---|---|---|
| The charging result is one pre-screened 5-duty instance tested with three synthetic tariffs | **VERIFIED** | Three tariffs are not three independent instances. The eligibility screen is 240/350. |
| A fixed-duty result has 46 starts, including a 2.5-kWh/26-second event | **VERIFIED for peak08** | Peak12 has43 starts; peak18 has42. Any claim of46 at all three peaks is **REFUTED**. |
| Short charges are a concern in the CG arm too | **VERIFIED** | CG has3,6,6 windows below3min at peaks08,12,18 respectively. |
| Original GIRO costs were absent from the prior headline but exist in source data | **VERIFIED** | They are surfaced above with their original interval semantics. |
| The relaxed five-duty model has a feasible CG-derived schedule cheaper than the tested fixed-duty schedule | **VERIFIED** | 2.807%,2.207%,7.335% less under matching fleet, input, tariff, and realized aggregate ending energy. This is a feasible improvement, not global charging optimality. |
| Those gains persist under GIRO's charging/setup/reserve rules or at larger k | **UNRESOLVED** | Requires the planned matched constrained reruns. |

The fixed and cleanup optimizers certify **grid-objective** outcomes in their stated route pools/route families, not global continuous charging optimality. The peak18 cleanup's grid charging MIP stops at a 0.1363% gap. Continuous replay prices are separately reported and not substituted into that grid proof.

## Evidence and reproduction

`audit.py` replays every selected route at350kW with zero timing/power grace, reconstructs SOC using the production graph's trip/deadhead energy, checks saved ending SOC, checks exact-once service, and counts positive stored charging windows. It consumes the frozen eligible-instance hash `b386f8a16958d25c857297ac4643bf6c73ae2114557c585446725cbd51c8b64d`. No optimization was run for this F6 audit. Source file identities are in `source_hashes.json` and `results.json`.

Remote original and fixed sources: `/home/nc437/ladder-lite/giro_zero_start_fee_20260913/{results/peakXX/fee0/frontier.json,results/peakXX/fee0/joint/comparison.json,repriced_sources/peakXX/fee0/original.json}`. Fresh CG cleanup sources remain in local `outputs/terminal_duplicate_cleanup_20260916/results/peakXX_fresh/`; their selected-route hashes are in `results.json`. The local `sources/` copies retain the fetched originals; publication can use the compact audited CSV/JSON plus those hashes and remote paths.
