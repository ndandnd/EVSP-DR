# Cross-generation evidence collection and rebuild

The collector is read-only with respect to computational artifacts. It hashes
matched files and writes one new manifest. It does not repair tails, copy
results, invoke solvers, or submit jobs.

## Paste-ready Unicorn collection command

```bash
set -euo pipefail
cd "$HOME/EVSP-DR-cross-generation-evidence"
OUT="$HOME/evsp-evidence-manifests/cross-generation-20260816.json"
mkdir -p "$(dirname "$OUT")"
test ! -e "$OUT"

python -u src/collect_cross_generation_inputs.py \
  --template CROSS_GENERATION_EVIDENCE_INPUT_MANIFEST_20260816.json \
  --root "current_heuristic=$HOME/EVSP-DR-current/src/results" \
  --root "repool_small=$HOME/EVSP-DR-k40mip-f40b120/results/repool_small" \
  --root "exact_big=$HOME/EVSP-DR-k40mip-f40b120/results/exact_big" \
  --root "k40_factorial=$HOME/EVSP-DR-k40mip-f40b120/src/results/k40_factorial" \
  --root "exact_telemetry=$HOME/EVSP-DR-k40mip-f40b120/src/results/exact_cg_telemetry" \
  --root "mip_campaign=$HOME/EVSP-DR-k40mip-f40b120/src/results/mip_statistics" \
  --root "releases=$HOME/EVSP-DR-releases" \
  --out-manifest "$OUT"

sha256sum "$OUT"
```

Review `collection_report` and enrich null run metadata from immutable
campaign/release manifests before analysis. Null hashes or scale identity must
not be guessed from filenames.

## Rebuild

Run `src/build_cross_generation_evidence.py --input-manifest <reviewed.json>
--out-dir <new-empty-directory>`. Output publication is create-only and includes
`completion.json`, normalized CSVs, provenance, compatibility documentation,
coverage audit, dry-run rerun plan, and deterministic figures.

## Inputs still expected from Unicorn/releases

- Current instrumented heuristic-DP pricing CSVs and endpoint/checkpoint JSONs.
- Exact `repool_small`, `exact_big`, and k40 factorial `.iters.csv`, status,
  journal and optional phase telemetry.
- MIP campaign checkpoint/final artifacts with RAW/GIRO treatment metadata.
- Verified single-duty, exact-pair and small-union archive manifests.
- Instance, trip-set and tariff hashes from immutable status/release manifests.
- Git/worktree identity, solver/backend versions, resources and stopping rules.

The tracked historical CSVs are already enumerated in the template manifest.
They provide real trajectories but incomplete provenance; the output must retain
those provenance fields as null.
