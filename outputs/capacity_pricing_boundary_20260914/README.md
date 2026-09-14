# Capacity pricing boundary campaign

This root contains eight matched strict-capacity k1 pricing diagnostics: duties
13405–13408 crossed with the exact reference and prefix-memo selectors. Every
cell uses flat prices, 240 kWh initial energy, reserve 0, a 13,200-second CG
limit, a 600-second diagnostic finite-pool MIP, and a four-hour allocation.

The immutable execution record starts with [`manifest.json`](manifest.json),
[`freeze.json`](freeze.json), and [`jobs.json`](jobs.json). The launch and native
license evidence is summarized in [`LAUNCH_REPORT.md`](LAUNCH_REPORT.md) and
[`launch_verification.json`](launch_verification.json). The remote root is
`/home/nc437/ladder-lite/capacity_pricing_boundary_20260914`; large results are
stored under `/share/scaglione/nc437/evsp-dr/capacity_pricing_boundary_20260914/results`.

Collect completed endpoints on Unicorn with:

```bash
/home/nc437/evsp_env/bin/python \
  /home/nc437/ladder-lite/capacity_pricing_boundary_20260914/collect_verified.py \
  --root /home/nc437/ladder-lite/capacity_pricing_boundary_20260914 \
  --adapter /home/nc437/ladder-lite/capacity_pricing_boundary_20260914/strict_capacity_adapter.py
```

The collector reuses `strict_capacity_adapter` with `kind=pilot`. Running
attempts appear only in `workflow.attempt_progress`; a CG or MIP enters
`records` only after the matching worker stage returns zero and the manifest,
allocation, commands, inputs, and artifact hashes verify. A pricing deadline is
an uncertified CG endpoint. A completed MIP proves only its saved finite pool.

The 15-second cluster-native smoke is preserved in `native_smoke_189147/`.
Fresh, separately verified evidence from the preceding strict k2 pilot is in
`strict_capacity_k2_audit_20260914T1659Z.json` and `.md`; those physics cells
had unequal CG limits and are not runtime-causal estimates.

| index | duty | selector | job |
|---:|---:|---|---:|
| 0 | 13405 | reference | 189164 |
| 1 | 13405 | prefix-memo | 189165 |
| 2 | 13406 | reference | 189166 |
| 3 | 13406 | prefix-memo | 189167 |
| 4 | 13407 | reference | 189168 |
| 5 | 13407 | prefix-memo | 189169 |
| 6 | 13408 | reference | 189170 |
| 7 | 13408 | prefix-memo | 189171 |

`normalizer_validation.json` records the four completed duty-13408 production
endpoints through the registered `capacity_pricing_boundary_20260914` branch.
It preserves two certified CG rows and two separate finite-pool MIP rows.
