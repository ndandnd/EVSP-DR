# Independent publication verification

**PASS: 32/32 automated checks, plus visual review.** Final export hashes and the two audited source hashes are recorded in `verification.json`. No live artifact edits or cluster access were performed.

The Doc changes are confined to the existing chain-extension section and strict graph paragraph. All prior Doc source/history/figure links remain. New six-chain k33 values, larger endpoints, omitted-constraint qualifications, uncertified CG scope and native graph-reload measurements match the423-check operations audit and strict graph receipt audit. Job824877 is recorded as submitted, not as a completed CG result.

All42 slides remain. Only slides10/42 and their notes change semantically. Export-generated table-style GUIDs and media filenames changed; definitions, all embedded image bytes and every slide image reference were independently matched. Slide42 remains a native editable7×6 table with all six chains and exact displayed values. Source notes point to the latest evidence.

Rendered slides10/42 and Doc pages2/3/6/7 were inspected. The expanded table and lower caption do not overlap; text remains readable without clipping. Final Doc page7 was re-rendered after the824877 submission addition and spacing correction. The unchanged visible Slides PDF was retained while notes-only PPTX updates were checked. Local PNGs under `rendered/` are QA intermediates and should be excluded from evidence Git publication.

Reproduce the content/hash checks from any working directory:

```sh
/Users/nadan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 /Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260922/monitor_20260922T195842Z/publication/verification/verify_exports.py
```

The verifier uses ZIP/XML inspection and pypdf; it neither modifies nor exports the Doc or deck. Visual findings are the independent reviewer's observations, not automatic proof of layout.
