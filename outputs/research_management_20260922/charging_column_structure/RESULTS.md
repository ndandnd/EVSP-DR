# Fixed-pool charging-capacity pilot: completed

All three equivalent formulations prove a minimum fleet of **3** on the same 321-route, 35-trip saved pool. Each LP relaxation also has objective 3, within numerical tolerance. All selected solutions pass the original trip-cover and one-minute charger-occupancy constraints. This is a finite-pool proof; it is not a full-model CG certificate or a new physical replay.

The pilot preserves the saved strict 18E1 charging schedules, taper assumptions, binary route bounds, trip-cover sense, fleet objective, and conservative one-minute occupancy. Endpoint differences are constructed from the rounded occupied minute runs, not from unrounded physical intervals. Only the matrix representation changes.

| Formulation | Input rows / variables / nonzeros | Presolved rows / variables / nonzeros | Build s | LP wall s | MIP wall s | LP / MIP / bound |
|---|---|---|---:|---:|---:|---|
| Original minute rows | 1789 / 321 / 44844 | 82 / 290 / 5009 | 0.2072 | 0.0495 | 0.3392 | 3 / 3 / 3 |
| Merge identical rows | 303 / 321 / 10232 | 108 / 292 / 5596 | 0.0283 | 0.0151 | 0.0587 | 3 / 3 / 3 |
| Endpoint differences + occupancy variables | 307 / 593 / 7714 | 131 / 403 / 6779 | 0.0424 | 0.0152 | 0.1002 | 3 / 3 / 3 |

All models retain 321 binary variables; the endpoint model adds 272 continuous occupancy variables. Every MIP used one search node. Editable exact values: [results.csv](pilot/results.csv).

Merging identical rows cuts input nonzeros by **77.18%**. Full-matrix zero percentages are 92.191116%, 89.480070%, and 95.762726%, respectively. The merged model has fewer nonzeros despite a lower percentage of zeros. Native presolve leaves the original formulation with the fewest nonzeros, so input sparsity alone does not predict solver work.

The measured times favor merged rows on this tiny case. This is one run in a fixed original/merged/endpoint order, with possible first-model initialization and cache effects; it does not establish a causal or general speedup. A larger fixed-pool comparison with repeated or balanced ordering would be needed before changing production defaults. Safe use is currently limited to the frozen pool: new CG columns may invalidate row equivalence and require updated rows and correct capacity duals in pricing.

## Proof and reproducibility

- Original minute model: [optimal incumbent and bound](pilot/collections/20260922T053756Z/attempts/729675_r0/original_minute/mip.log#L46), [presolved dimensions](pilot/collections/20260922T053756Z/attempts/729675_r0/original_minute/mip.log#L29), [LP optimum](pilot/collections/20260922T053756Z/attempts/729675_r0/original_minute/lp.log#L30).
- Identical-row model: [optimal incumbent and bound](pilot/collections/20260922T053756Z/attempts/729675_r0/identical_rows/mip.log#L46), [presolved dimensions](pilot/collections/20260922T053756Z/attempts/729675_r0/identical_rows/mip.log#L29), [LP optimum](pilot/collections/20260922T053756Z/attempts/729675_r0/identical_rows/lp.log#L30).
- Endpoint model: [optimal incumbent and bound](pilot/collections/20260922T053756Z/attempts/729675_r0/endpoint_difference/mip.log#L44), [presolved dimensions](pilot/collections/20260922T053756Z/attempts/729675_r0/endpoint_difference/mip.log#L29), [LP optimum](pilot/collections/20260922T053756Z/attempts/729675_r0/endpoint_difference/lp.log#L30).
- [Execution manifest](pilot/collections/20260922T053756Z/attempts/729675_r0/manifest.json) pins pool, runner and worker hashes, settings, route model and proof scope. Each variant directory contains its copied log, MPS model and result JSON with the original-matrix validation and selected route payloads.
- [Collection receipt](pilot/collections/20260922T053756Z/collection.json) hashes the copied evidence and records Slurm job **729675**, COMPLETED, exit 0:0, six seconds on `snavely-cpu-02`.

Gurobi 12.0.3; Seed 0; Threads 4; no explicit MIP start; separate 30-second LP diagnostics and 300-second MIP limits. The full three-model process took 1.8683 seconds. The LP solution was not transplanted into the MIP. Process memory readings are cumulative and do not support a per-variant memory comparison.

Execution revision **f37c6ce4c6e0a7ccffd3719a633bacd40b9a351c** is preserved at `codex/charging-column-structure-20260922`. Its parent pilot commit `3c3cdd2aae16d66d7337058dd5048389cc4f9fcc` adds only `pilot/run_pilot.py` and `pilot/worker.sub`; f37 changes only `pilot/worker.sub` to pin the license environment. Neither commit changes production source.

The deterministic publication input is [pool.jsonl.gz](pilot/inputs/pool.jsonl.gz), with gzip mtime 0. The uncompressed SHA256 is `714cec263633e3fa8ac178052b3be595097c1990769503afcac86e3d5d11a845`. From the pilot directory, restore it with:

```sh
python3 -c "import gzip,pathlib; p=pathlib.Path('inputs'); (p/'pool.jsonl').write_bytes(gzip.decompress((p/'pool.jsonl.gz').read_bytes()))"
```

The original pool-generation commit is unknown and explicitly null in the manifest; the input content is pinned by SHA256. Source model inspection, mathematical equivalence, no-license tests, and official Gurobi references are in [README.md](README.md).
