# Independent launch audit

Recommendation: eight independent same-allocation paired jobs: d00_g0 and d00_g1 fresh (2CPU/32G,7200s per arm,4h30 allocation); w1_k08,w4_k11,w6_k12 and reverse-order repeat w1_k08 warm (8CPU/96G,7200s per arm,4h30 allocation); k1_duty13406 capacity and e1_short_k2 capacity (1CPU/24G,10800s per arm,6h30 allocation). Alternate arm order; use fresh processes. All eight eligible concurrently on default_partition; exclude scaglione-compute-01. Requests preserve prior successful allocations; warm parent RSS excludes eight replay workers, so do not use it as aggregate memory.

Warm bound512/900s/workers8 must remain exact. All three historical child imports accepted512 without deadline (470,378,403s). Include cache read, index setup, worker creation, pool parsing and replay in total timing, with phase timing separately. True previous-k sources below; parent freezes and verifies immutable source/cache copies.

Capacity baseline both uses fixed accounting; only prefix flag varies. e1_short_k2 first binding-dual call took8598s versus14s with no dual. 10800s may censor full CG. Compare same completed pricing calls only, otherwise show bounded progress and censoring; no RMP lower-bound claim. No new MIPs needed.

Historical fresh fullCG1663/3129s, warm fullCG2969/3565/3877s justify7200s perarm. Shorter1800/3600s may censor and must be labeled.

Exact identities and original arguments are in source_evidence.json and capacity_evidence.json. Source CSV paths below relative to overnight code checkout.

d00_g0
CSV: overnight_decomposition_20260912/d00_g0.csv
Input SHA256: 0ee1ff2b04d5ab75e40dfc841dbf91f4383a6b612e2a7db017e4b5d80f099f89

d00_g1
CSV: overnight_decomposition_20260912/d00_g1.csv
Input SHA256: 0c8cec257e17dec782f82fc3d078bf489ff09eedaa27ee03f6e80c36dbe91696

w1_k08
CSV: scale_ladder/instances/nested_probability_k2_15_20260908/Practice_Custom_DutyUnion_k08_p01_20260908.csv
Input SHA256: cca5b6ca0c9af9eeb20adb414e300a130471e9e802ef6543a85b9d3ba2860f9d
Parent status: /home/nc437/ladder-lite/overnight_extension_20260912/cases/w1_k07/cg.json
Status SHA256: 851553ece66ddb9b55b1cddb6cceb55c3af660d32d3df6a491347f5b239aa641
Pool: /home/nc437/ladder-lite/overnight_extension_20260912/cases/w1_k07/cg.json.columns.jsonl
Pool SHA256: 00839ab01f8b3f8cff51a26289942e81f8b19a6cc2169ca5d9ba6dbf4c9fa3d5

w4_k11
CSV: scale_ladder/instances/nested_probability_k2_15_20260908/Practice_Custom_DutyUnion_k11_p04_20260908.csv
Input SHA256: 837037c026cf3971309c96c719f594565fdc5521dfe2d43241e2a344b31f1a92
Parent status: /home/nc437/ladder-lite/overnight_extension_20260912/cases/w4_k10/cg.json
Status SHA256: afad352b6510b5bd83d0e6d9d7bd3964ede47298feaa7892742d67b689e5a170
Pool: /home/nc437/ladder-lite/overnight_extension_20260912/cases/w4_k10/cg.json.columns.jsonl
Pool SHA256: 651f9710aa82b04e099c71f0c604baa708a3ea1f0329b6a10dda57db6f3ca7fd

w6_k12
CSV: scale_ladder/instances/nested_probability_k2_15_20260908/Practice_Custom_DutyUnion_k12_p06_20260908.csv
Input SHA256: 8b8ec38fc2303f2aad71358fae7583c1c4fe009efeb1a147eb9eeebd93abacc6
Parent status: /home/nc437/ladder-lite/overnight_extension_20260912/cases/w6_k11/cg.json
Status SHA256: 9e107f40a43346f372e8bec5a351f24b3c25252a612cec465ee93d98af9068cf
Pool: /home/nc437/ladder-lite/overnight_extension_20260912/cases/w6_k11/cg.json.columns.jsonl
Pool SHA256: 52eca27e1ba5313f85fe030c64ab810c3a564957002a4dea04078c23cc30b826

## Final deployment amendment

The final campaign has **nine paired allocations**: add `k1_duty13406` with the combined capacity/PARX60 arm and existing `data/tariff_response/peak12_h26.csv` tariff (SHA-256 `8b231a2574fd4e3b4dc94873ad2d6515bfaba09e07afefbb1df43d5f775a8381`). Both selector treatments use corrected accounting. All nine independent jobs are eligible concurrently on default_partition and exclude scaglione-compute-01.

Warm allocations request **08:00:00**, superseding the earlier4h30 recommendation: at most3h to build a new source-authenticated network cache, then2h per arm plus overhead. Cache preparation is recorded separately; both measured arms require the same new cache. Source-hashed historical cache identities are never rebound. Fresh allocations remain4h30; three capacity allocations remain6h30. The repeated w1_k08 uses reversed order. Tooling commit `dd16c9f0` under `scripts/efficiency_validation_20260912/` provides preparation, worker, explicit-submit launcher and read-only collection. This is tooling/readiness evidence, not a submission or result.
