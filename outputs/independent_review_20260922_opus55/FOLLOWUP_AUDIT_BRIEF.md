# Focused follow-up evidence audit

Please take option 1: check the completed Opus 5.5 review against primary evidence and fill consequential access gaps. This is an evidence audit, not another independent review: disclose where you designed the pilot or wrote the earlier recommendations being assessed. Challenge both the original review and Astra's corrections where the evidence warrants it.

Project: /Users/nadan/Documents/projects/demandresponse. Start with this directory's REVIEW.md, ASSESSMENT.md, audit_counts.json, audit_bounds.md and audit_long_wait.md. The original review is frozen. Avoid repeating counts or code checks already supported by those audits unless you find a conflict. Use Opus-F1 through Opus-F8 to distinguish these findings from September 16 F-numbers.

## Priorities

1. **Opus-F4: target-informed fleet cap.** Inspect the actual pilot and later matched replication, including 7/8 versus 0/8 at k8 and the mixed k15 outcomes. Identify how each cap was selected, what information the method used, and what conclusions this supports. Separate a legitimate target-feasibility test from a general target-free algorithm. Do not substitute ceil(weighted-LP route weight) for a certified fleet bound. Assess a cap escalation rule and the smallest decisive control; do not run it.
2. **Opus-F3: primary operating assumptions.** Read relevant raw GIRO/Transdev material under outputs/meeting_20260910/giro_email_sources/, not only summaries. Trace battery/reserve, station power/counts, terminal energy, deadhead timing and waiting rules to file/page/email references. Distinguish explicit operating requirements, interpretations and deliberate research simplifications. Map each material difference to the actual baseline, strict and charging-comparison campaigns; do not treat experiments not read by the first reviewer as absent. Keep raw correspondence private.
3. **Published claims and execution provenance.** Read the current Doc and weekly Slides plus GitHub history/branches/PRs relevant to the flagged code, if access works. Check whether current text overstates the underlying results or has already corrected the review's concerns. Resolve only material dirty-source/missing-artifact questions through scoped read-only Unicorn checks and saved manifests; no broad data download. Record access failures and timestamps rather than inferring contents.
4. **Opus-F1/F2/F7: proof and attribution.** Use the existing bound/long-wait audits as starting points. Distinguish full pricing certificates, numerical fleet floors, restricted-pool proofs, observed feasible schedules and broader physical feasibility. The 57-minute connection restriction is verified; its real-instance fleet/cost effect is not. Do not adopt assumed Q values or missing reduced costs as certificates. Separate a larger sequential target miss from evidence that its relative advantage over fresh disappears.
5. **Narrow literature check.** Only after factual gaps: identify a few closest primary papers on integer-directed column generation/diving and price-and-branch. State which mechanism is established and what specific new contribution our evidence might support. Cite actual papers and the passages supporting the comparison. Do not infer either novelty or lack of novelty from general familiarity alone.

## Deliverable and limits

Write FOLLOWUP_AUDIT.md beside this brief. Start with a short decision summary, then a table: Opus finding; verified/refuted/qualified/open; exact primary evidence (source revision/hash, file/line, log fields or document location); effect on our claim; smallest next action. Explicitly identify your prior involvement. Add proposed wording corrections and at most three ranked next experiments. Preserve disagreements and unresolved points rather than forcing agreement with either reviewer.

Do not modify production code, submit or cancel cluster jobs, edit live Docs/Slides, publish, or contact anyone. Read-only access is sufficient; list indispensable unavailable evidence. Avoid a broad historical rereview, mass exports or repeated queue checks.

Current Doc: https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit
Weekly Slides: https://docs.google.com/presentation/d/1F6udjgkiPH51vMUcku7ZT3PhZPAxsQwUCi3TK9vFRDM/edit
GitHub: https://github.com/ndandnd/EVSP-DR
Unicorn: ssh nc437@unicorn-login-01.coecis.cornell.edu; artifacts under /home/nc437/ladder-lite/, mapped by local manifests.
Later matched pricing evidence: outputs/research_management_20260921/monitor_20260921T235438Z/README.md and monitor_20260922T035403Z/README.md.
Latest collected endpoints at handoff: outputs/research_management_20260922/monitor_20260922T235958Z/README.md. Treat this as a dated snapshot, not a live status claim.
