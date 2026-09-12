# Warm preparation recovery

The initial four warm allocations stopped before graph construction because the generated cache-only command incorrectly included `--out`. The CLI rejects this combination. The initial known-option tests did not exercise parser cross-option restrictions; they were insufficient for this launch path. This was a launcher defect, not evidence of a CG/pricing result. All failed attempts and their diagnostics remain in the original campaign and in `warm_startup_failure.json`.

Commit `11afd8372dd14fdb2a11cd5eaddbee1d02644b1f` omits `--out` only for preparation. Solver files are byte-identical to the original execution pin `89c5ba3e8a66fd77ad8397e5ce63eae647a0a49a`. New actual-parser tests cover valid generated commands and the rejected original combination. An unmocked local run of the generated command built a cache for the real eight-trip fixture (696 nodes,15,238 arcs); see `cache_smoke.json` and `cache_smoke/`.

Four replacements use a separate checkout and manifest at `/home/nc437/ladder-lite/efficiency_validation_warm_retry_20260912`. Input bytes, frozen prior-k parents, physics, treatment order, resources and limits match the original warm design. Only the launcher preparation command and associated execution pin/path changed. The five original fresh/capacity pairs continue untouched.

| Pair | Failed startup job | Replacement job |
|---|---|---|
| w1_k08 | 964196 | 966398 |
| w4_k11 | 964197 | 966399 |
| w6_k12 | 964200 | 966400 |
| w1_k08_repeat | 964201 | 966401 |

Manifest SHA-256: `b28a1a01aea6d4320b18db1fa66d12fa6a5e80aa7448921dd7997fd34508978b`. Both original and recovery roots are collected hourly using the v2 collector. Retain original-root inputs and borrowed Git object stores until all dependent runs and archives are secured. Job submission is not a correctness certificate or a timing result.

Startup verification: all four replacements remained RUNNING beyond two minutes, with preparation execution records, the corrected command, successful Gurobi preflight output and empty stderr. This confirms progress past the original parser failure; cache completion and paired solve results remain pending. See `startup_verified.json` and `startup_collection.json`.
