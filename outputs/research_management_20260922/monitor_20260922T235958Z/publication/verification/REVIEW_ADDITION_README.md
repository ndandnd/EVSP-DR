# Supplemental figure-guide and review-link QA

**PASS:21/21 checks plus visual inspection of Docpages2/3.** This is an additive stage; the original38-check receipt and its source verifiers remain unchanged. Final current export hashes are recorded in `review_addition_verification.json`.

Removing exactly the two approved linked paragraphs reconstructs the previous final Doc byte-for-byte. All42 visible slides, their images and resolved image references are unchanged; only slide42notes receive an appendix, and its entire previous body is preserved as an exact prefix. The retained prior finals match the hashes in the original38-check receipts. The Slides PDF is byte-identical to the prior verified visible deck.

The figure guide supports the saved-route MIP versus route-generation explanation, heuristic upper bounds,512 inherited-route limit and packed-graph explanation. The checked review and long-wait audit support the57-minute direct-trip gap cap, positive-charge inter-trip station bridges and unresolved production fleet/cost impact. They do not establish a new full-model certificate or measured fleet loss. Exact source-file hashes and both link targets are recorded in the supplemental JSON.

Rendered Docpages2/3 are clear, with no clipped text or overlap. The unchanged k33 table now continues across the page boundary between complete rows; all cell values remain intact. PNGs under `rendered/` remain local QA intermediates and should be excluded from Git publication.

Reproduce using bundled Python with pypdf:

```sh
/Users/nadan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 /Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260922/monitor_20260922T235958Z/publication/verification/verify_review_addition.py
```

Use this supplemental verifier for the final additions. The original Doc/Slides scripts describe the prior38-check stage and intentionally retain their original predicates. No UI edit, cluster call or source mutation was performed during this review.
