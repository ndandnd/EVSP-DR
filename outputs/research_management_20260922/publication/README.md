# Document update receipt — 22 September

- New three-page tab: [Route columns & tests](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ikbgt85cdszz). Four native editable tables: column definitions, toy overlap, exact matrix compression and completed Gurobi pilot. Existing timing/sparsity tab preserved.
- Current journal: one navigation line plus two targeted endpoint updates. `current_before.md` matches the previous verified export. All other lines and existing figure references remain unchanged.
- `route_columns_tab.md` is the actual live-tab export, with source links. `route_columns_explained.md` is the initial working explanation, superseded by the live export and completed pilot; it does not represent final scheduler status.
- `doc_verification.json`: 32 checks passed, including matrix sizes, direct native-runtime comparisons, original-matrix validation, finite-pool scope, completed preparation diagnostics, early fleet endpoints and existing figure preservation. `verify_doc.py` reproduces these checks.
- Visual QA: three pages, with section breaks before compression and tests. All tables are editable, unclipped and legible. The pilot timing column uses Gurobi Runtime; the CSV additionally reports wrapper wall time. No general speedup is claimed.
- Original PDF exports and page PNGs remain local. Their hashes are retained in `artifact_hashes.json`; redundant graphics are excluded from Git. Google revision history preserves native formatting and prior content.
- No Slides changes during this task. Current standing permissions take precedence over the earlier dated automatic-Slides policy.
