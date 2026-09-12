# Planned paired efficiency validation on 12 September

Historical planning record. See [launched status and initial failure evidence](efficiency_validation_20260912/README.md) for the current state.

Status at creation: registered before launch. No submission IDs or completed campaign results have been supplied to this register. The algorithm implementation task owns source integration, validation and submission. The research manager owns result collection and document updates. This entry is provisional until the frozen manifest arrives.

Proposed cluster root: `/home/nc437/ladder-lite/efficiency_validation_20260912`.

| Paired workload | Allocations proposed | Comparison |
|---|---:|---|
| Fresh decomposition CG | 2 | Same input and code, optimization switches changed |
| Three actual warm successors plus a reverse-order repeat | 4 | Same frozen previous-k pools, 512-route / 900-second import limit |
| Flat-price capacity cases | 2 | Corrected accounting in both arms, capacity selector switch changed |
| Combined station-power constraints with noon-peaking price | 1 | Corrected accounting in both arms, capacity selector switch changed |

Each pair shares an allocation and runs its treatments in separate processes. Proposed treatment limits are two hours per baseline CG arm and three hours per capacity arm. Current-source warm caches are rebuilt before paired timing, with up to three hours for preparation; record preparation separately from treatment and total allocation time. All nine independent allocations are intended to be eligible concurrently, with `scaglione-compute-01` excluded. No MIPs are planned in this first validation stage. These are reported plans, not verified scheduler settings.

## Source status and evidence

The implementation task reports focused checks passing for accounting-only fix `550bc795b18f801dca3a07b8becc1dc4c5527abb` and opt-in capacity selector `309d98d266ebaf6b7e99543a67f8f2be5736874a`. Its baseline integration was reported as `a850ceb2`; the final execution pin is pending.

The local [independent launch recommendation](../efficiency_validation_20260912/audit/RECOMMENDATION.md) currently describes eight pairs, before the proposed ninth noon-price case and the added cache-preparation allocation. Reconcile this with the final manifest rather than silently treating the earlier recommendation as the launch specification.

The [local smoke record](../efficiency_validation_20260912/local_smoke/validation.json) tests an eight-trip subset at commit `9a957e9a1a317988660396c5d06e4ccfc5dbd679`: paired fields and column journals agree, with two CG iterations and 23 columns. This is a small correctness check, not a performance estimate or validation of every final integration switch. The register maintainer read these artifacts but did not independently rerun the tests.

## Required collection metadata

Before wiring result collection, obtain the frozen manifest, full execution commit, exact output paths/schema and job IDs. Retain source/input/pool/cache hashes; resource requests and actual arm order; switch values; cache-preparation and treatment times; completion/error markers; and pricing certificates or explicit stopping reasons. No guessed glob or fabricated submission record is added here.

A computational acceleration should reduce runtime under unchanged objective/physics. Correcting station-power tariff accounting is a separate model-cost repair and can alter reconstructed costs; it is not evidence of demand-response savings. Preserve old plots as dated controls until comparable completed runs justify new figures. Keep scheduler completion, pricing certification, route validity and eventual integer proofs separate. Do not infer full-CG speedups from local component benchmarks.


## Pre-submission staging update

The implementation task reports that the cluster root and isolated code clones are staged, with no submissions yet. Baseline source is now `dd16c9f079e4e3e6e8689da2d414034feb6cc182`, subject to a final collector-related pin; capacity remains `309d98d266ebaf6b7e99543a67f8f2be5736874a`. The earlier allocation discrepancy is reported resolved:

| Workload | CPUs | Memory | Allocation | Treatment budget |
|---|---:|---:|---|---|
| Fresh CG | 2 | 32 GB | 4 h 30 min | 2 h per arm |
| Warm CG | 8 | 96 GB | 8 h | Up to 3 h cache preparation outside paired timing, then 2 h per arm |
| Capacity CG | 1 | 24 GB | 6 h 30 min | 3 h per arm |

Warm inputs are reported frozen from completed previous-k cases only. The implementation task reports eight whole-route reduced-cost checks passing for corrected accounting (`local_smoke/fixed_accounting.json`), 34 baseline integration tests passing, and identical status and pools in the eight-trip CG smoke comparison. These are reported validation outcomes; the research manager has not independently rerun them. The amended audit, final manifest, output schema and scheduler IDs will establish the launched configuration. No measured cluster speedup is claimed.
