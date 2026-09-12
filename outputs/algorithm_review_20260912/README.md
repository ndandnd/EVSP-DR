# EVSP–DR code and literature review

Start with [the review and ranked recommendations](REVIEW.md). It connects the current execution code to measured bottlenecks, relevant literature, and the two-month paper schedule.

| File | Purpose |
|---|---|
| [REVIEW.md](REVIEW.md) | Integrated conclusions, priority order, experiments and paper positioning |
| [WORK_ORDERS.md](WORK_ORDERS.md) | Six bounded implementation tasks with acceptance gates |
| [PRICING_REVIEW.md](PRICING_REVIEW.md) | Pricing recurrence, capacity-window work and exactness boundaries |
| [ENGINEERING_REVIEW.md](ENGINEERING_REVIEW.md) | Code-line findings for import, graph, master and I/O efficiency |
| [MATHEMATICAL_NOTES.md](MATHEMATICAL_NOTES.md) | Fleet-only bound derivation, model contracts and safe column dominance |
| [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) | Primary papers and reusable code resources |
| [RUNTIME_EVIDENCE.md](RUNTIME_EVIDENCE.md) | Timings for 84 certified fresh-covering cases and a cold-build example |
| [review_provenance.json](review_provenance.json) | Immutable source revisions/file hashes and inspected public-code pin |

The integrated priority order takes precedence over the optional schedules in individual subreviews. Implementing the entire engineering inventory is not a prerequisite for a conference paper.

Production source and cluster jobs were unchanged by this review. Proposed speedups are unmeasured until their paired benchmarks are completed. The runtime extraction is verified against 98 selected cases in the frozen collector snapshot, including 14 certified overnight cases. To repeat the read-only reconciliation:

```sh
python3 outputs/algorithm_review_20260912/extract_runtime_evidence.py \
  outputs/post_meeting_20260910/monitor/20260912T052658Z.json
```

The two existing event-pricer suites passed (26 tests total); see [verification.json](verification.json) and [runtime_validation.json](runtime_validation.json). Pricing-batch and shortest-path timing counters overlap in the source; the review explicitly accounts for that. An RMP objective, pricing certificate, finite-pool MIP result and physical validation remain distinct.
