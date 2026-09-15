# Larger compact previous-k initialization comparison

Fixed panel: six chains × target k20/k25 × mandatory core / core filled to512, with24 independent CGs and24 own-CG dependent MIPs. The mandatory core preserves the previous-k integer witness and every positive LP-support trip set with its exact native cheapest ordered path. Fillers are deterministic absent incidences ranked by length descending, cost/trip, then sequence. Never drop core above512. No current-k optimized or GIRO routes are seeded.

Both arms use native a0e0 CG code and its unchanged graph-cache identities. The smaller cohort used e091; across-size results are not a single-code comparison. Only within-pair initialization comparisons are controlled. CG budget14400seconds; MIP871 budget12600seconds total,10800fleet stage. Both use8CPUs; CG96GiB/5h allocation, MIP24GiB/4h30 allocation. Default partition, compute-01 excluded, no arbitrary independent-CG throttle. MIPs depend only on their own CG.

All k19/k24 parents and target caches were ready at freeze, so no upstream pending jobs block these CGs. Parent CG and MIP cost plus extension CG history are retained. The history list is not a total of all intermediate MIP compute; frozen upstream manifest and source pointers preserve that ancestry. Target graph original build time is recorded separately as common preexisting preparation. Results live under /share/scaglione/nc437/evsp-dr/compact_large_seed_20260914 via home links; target caches are reused readonly without13GiB per-arm copies.

Selected sequences, actual child replay/acceptance/additions, CG pricing certificate, finite-pool MIP proof, physical replay, and GIRO target attainment are separate evidence. Derived seed statuses make no optimization or pricing-certificate claim. Existing restart-safe worker preserves private attempts and execution accounting; preempted native CG restarts and MIP tree restarts. A completed output is hash checked before reuse. Submission intent ledger forbids blind resubmission after uncertain scheduler response.

Native validation uses largest C5k25 (624trips), both arms with one CG iteration and short dependent MIP. This is compatibility validation, not a production scientific outcome. Production manifest and job map are immutable evidence; no shared collector/register files modified by this campaign preparation.

## Native validation

Frozen manifest `ad27ba3208ef72f1dce76a3452652542b6ee2fd7612fae797bd6b801d464e50f`. Both largest-C5k25 arms passed: core452/452 accepted/added (25.73s replay), filled512/512 (29.40s); zero rejected routes. Native cache load and CG driver1iteration produced usable artificial-free endpoints; peak RSS27.37GiB. Each own native871 MIP returned26buses with successful physical replay, zero rejected/repaired columns, and exactsource result/journal hashes. These are compatibility fixtures, not production conclusions.

## Production launch

All48production jobs submitted after independent review and healthy scheduler gate. CG: 207647–207670 (24 jobs). MIP: 207671–207695 (24 jobs). Exactcase mappings and dependencies are in jobs.json/case_jobs.json. Batchedscontrol verified every job, CPU/memory/time/requeue, defaultpartition, and compute-01 exclusion. Initialstates and reasons are in scheduler_verification.json.
