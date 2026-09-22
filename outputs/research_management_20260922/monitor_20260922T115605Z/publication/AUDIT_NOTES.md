# Independent final-publication audit

**PASS: 27 checks.** Reproduce with `python3 verify_publication.py`. Source hashes and exact comparisons are retained in `verification.json`. This audit made no UI, cluster or optimizer calls and edited no publication exports or shared results.

- The deck grows from 41 to 42 slides. Only original slide 10 changes semantically, and its prior factual content is preserved through the changed final sentence. Slide 2 and every other original slide remain unchanged after normalizing generated table-style IDs. All original style definitions, embedded media bytes and slide image bindings remain intact.
- Slide 42 has one editable native 4-row × 6-column table. All 24 cells, including headers, match the source endpoints. The Doc's new 4-row × 5-column table likewise matches all 20 cells. CG minutes use reported CG wall seconds / 60 rounded to one decimal; route weights round only numerical noise around 33, 33 and 32.
- The Doc changes exactly the previous extension paragraph into the new status/table/scope block, plus the intended cache-reload sentence region in the strict paragraph. All original tables and figure references are preserved, and strict historical facts and links around that insertion are unchanged.
- C1/C4 MIPs are explicitly running at the 22 September 07:57 EDT snapshot. C3 is 34 buses / finite-pool bound 33, remains unproved and misses target 33. The no-pricing-certificate, restricted-master, duplicate-service and unvalidated-capacity limitations remain explicit. Separate graph time 9.56–12.31 hours is correctly rounded and distinguished from CG time.
- The cache statement matches local test evidence and native job 741034: 13 tests passed, verified source files, matching native test-log hash, no solver started and no full k19 graph/CG. The publication states that full-size graph and inherited-pool checks remain next.
- Final PDF slides 10/42 and Doc pages 2/3/6/7/8 were visually inspected. The new table, captions and edited scope statements are readable without clipped or overlapping text. The long strict paragraph continues normally across pages 6–7.

The audit uses standard-library ZIP/XML parsing, not python-pptx. Markdown exports verify Doc content and figure references but do not expose live Doc native-object internals. Temporary images under `audit_renders/` need not be published.
