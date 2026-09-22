# Verified Doc and weekly Slides update — 22 September 2026

The live [Google Doc](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ikbgt85cdszz) and [weekly Slides](https://docs.google.com/presentation/d/1F6udjgkiPH51vMUcku7ZT3PhZPAxsQwUCi3TK9vFRDM/edit?slide=id.g3fbd02b91a9_8_388) were updated and independently verified. Historical decks were not edited.

## Published changes

- Doc **Route columns & tests**: completed five-pool, five-setting fleet table; model dimensions and sparsity; exact capacity-row compression and endpoint equations; Gurobi Runtime distinguished from elapsed wrapper time. The first three tables are unchanged. The fourth only clarifies its runtime header. New completed results are on page 4.
- Doc current front: three targeted lines update the evidence link, queue snapshot and reserve/depot-power endpoint. **Fleet times & matrix**: one obsolete proposed-test paragraph replaced with completed outcomes. Existing timing tables, source links and figure/history tabs remain intact.
- Slides **10** and **39**: corrected reserve/depot-power endpoint and completed structural diagnostics. New **40**: editable 25-result fleet table. New **41**: editable exact capacity-representation comparison. All original media contents and their slide relationships remain unchanged.

## Verification and records

[verification.json](verification.json) records **32 passing checks**, hashes, table comparisons, normalized slide changes and rendering checks. [AUDIT_NOTES.md](AUDIT_NOTES.md) explains the verification scope. Run `python3 verify_publication.py` from this directory with the saved local exports available. Slides were edited natively; PPTX/PDF exports were used only to inspect the result.

Before/after Markdown exports are retained in Git. Native before/after PPTX, final PDF and inspection PNGs remain locally in this directory; their hashes are recorded, but the binary exports are omitted from Git. Reproducing the native export audit therefore requires those local files. Re-exporting a live document can regenerate XML identifiers and is not a replacement for its hashed snapshot.

Scientific sources: [completed MIP trials](../mip_structure/README.md), [scoped queue and strict continuation](../operations/README.md), [capacity pilot](../../charging_column_structure/RESULTS.md). Source logs, execution identities, solver statuses and proof scopes remain separate. The tiny capacity pilot does not establish a general speedup; the 25 MIP trials use one seed, and saved-start acquisition costs are excluded from reported optimizer times.
