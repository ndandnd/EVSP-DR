# EVSP–DR working instructions

## Cluster resources

Before cluster submissions, read `/home/nc437/ladder-lite/SCAGLIONE_RESOURCE_POLICY.md` on Unicorn (local reference: `outputs/meeting_20260910/SCAGLIONE_RESOURCE_POLICY.md`). Independent default-partition CG arrays use a default concurrency of **50**, or all cases if fewer exist. Do not impose smaller arbitrary throttles. Record a concrete cluster-policy or measured resource reason for any lower limit. Preserve true previous-k data dependencies.

Exclude `scaglione-compute-01` from every CPU-only job, regardless of partition. Other Scaglione CPU nodes remain usable for MIPs. Leave held historical jobs untouched unless explicitly authorized. Notify the user promptly if Unicorn access is lost.

## Experiment records

Use `outputs/research_register/README.md` as the experiment entry point. Record source input hashes, execution commit, physics, objective, master sense, initialization, resource requests, job dependencies, output paths and hashes. Keep scheduler status, CG certificate, finite-pool MIP proof, physical validation and GIRO target attainment separate. Do not call an RMP objective a full-model lower bound without the corresponding pricing certificate. Distinguish fractional route weight from the weighted objective and from a fleet-only lower bound.

Update the experiment register, current Google Doc (https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit) and current weekly Google Slides (https://docs.google.com/presentation/d/1F6udjgkiPH51vMUcku7ZT3PhZPAxsQwUCi3TK9vFRDM/edit) automatically when meaningful verified results or audited corrections arrive, without waiting for another reminder. This user instruction of 22 September 2026 supersedes the older no-Slides policy for the current weekly deck. Preserve source artifacts, superseded-result flags, historical decks, existing figures and historical/figure/CG Doc tabs. Keep summaries, tables and captions editable and verify publication. Avoid unchanged rewrites, exports and repeated notices; retain the existing four-hour meaningful-change heartbeat. If publication access is blocked, retain prepared changes and report the specific blocker.

When the live queue has zero running EVSP–DR jobs, prioritize diagnosing and recovering useful already-authorized work immediately. Repair failed predecessors and true dependencies without waiting for another user prompt; preserve scientific settings, record resource reasons, and do not release held historical jobs. Scheduler resource/priority waits are distinct from broken dependencies.

Low utilization, not only zero utilization, is a research-management trigger: when only one or a few EVSP–DR jobs run, check for ready independent authorized experiments and unblock review/preparation work. Preserve genuine previous-k dependencies; do not launch redundant runs merely to inflate job count. Maintain a parallel experiment backlog tied to unanswered research questions.
