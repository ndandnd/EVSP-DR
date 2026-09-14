# Retrospective iteration-prefix controls

These are reconstructed historical cold pools, not new CG runs or exact240-minute snapshots. No full-model LP certificate or final LP objective is carried into the constructed MIP inputs. Historical iteration logs are written after pricing and before insertion of that iteration’s batch; each prefix retains `found_iter < cutoff iteration`.

| Case | Logged boundary (min) | Excluded cutoff iteration | Raw/unique columns | Build job | MIP job |
|---|---:|---:|---:|---|---|
| c1_k15 | 239.881 | 2232 | 67,294 | 189905 | 189908 |
| c2_k15 | 239.917 | 2129 | 64,225 | 189906 | 189909 |
| c4_k15 | 239.990 | 2384 | 71,847 | 189907 | 189910 |

Every full source journal was streamed and SHA-256 checked. All found_iter values were valid and monotonic; reconstructed native cheapest-tripset counts match the frozen historical log. Original records/costs/order are unchanged. Construction took15–16seconds per case and peaked below212MiB; no large journal work ran on a login node.

Each own-dependent MIP uses frozen871d057 code, baseline covering physics,8threads,12600seconds total and10800seconds fleet search, matching current seed-arm MIPs. Native fullpool cost/replay and zero rejected/repaired columns are required before publication. Parent sourcecode is e091a4d; inputs and static data are hash-bound. All jobs use default_partition, exclude scaglione-compute-01 and preserve unique restart attempts. The unchanged reviewed registry helper records each MIP attempt path.

Native validation189902 passed on the full364-trip C1k15 singletonpool: feasible364buscover, physicalreplaytrue,0rejected/0repaired. This verifies format/model/replay compatibility, not target efficacy of the historical prefixes. Two focused constructor tests passed. Final native MIPs are still running/pending at the saved construction verification sample.

Final manifest SHA-256: `9db78a24cec6c0bdc8e731aa9ddffe79d122d234821500067e846ed42f9f4897`. The registry-only prelaunch amendment and original native-smoke manifest are preserved. Canonical mip_registry_additions.json rows are review aids; do not append duplicates because runtime helper registers attempt paths.
