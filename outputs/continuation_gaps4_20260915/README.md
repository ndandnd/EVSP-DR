# Chain 1 target28: longer search on the original saved pool

Question: can the same pool that produced31buses/bound28 in the original one-hour MIP recover28 with more integer search? No new columns, GIRO seed or physics change. Keep this separate from the original one-hour comparison.

Use the previously native-validated871d057 solver and identical worker/submission code, greedy initializer policy and defaultSeed0. Change only the time allowance in the design:12600seconds total,10800fleet; remaining time minimizes charging with fleet no greater than the first-stage incumbent. A new tree/hardware means this is not an isolated deterministic time-only comparison.

Prepare validates original result/source/journal/input/static hashes, physical replay and exact native non-time settings, and rejects duplicate longer searches or an already-excluded/matched target. Baselinecover240kWh/240kW,100000buscost+electricity+5perchargingstart; no reserve/sharedcapacity/endfloor. Native attempts check the full-size license and replay columns.

Resources:8CPU24GB4.5hours, default_partition, requeue with unique attempt directories, exclude scaglione-compute-01. No dependencies because source artifacts are complete. Policy read15September before submission. Preserve held historical and EVSPV2G work. Submission receipt and immutable manifest are authoritative.

Launched18:22:47UTC, job236276, verified running onjingjie-cpu-04. Native full-size license check passed; one unique allocation registered. Manifest SHA5359719887ab02c702ee723045d44ae047eaf838663ed38e9722040391598638. Focused collection and merged normalization passed. Main collection began before this root was registered; next full scan includes it.
