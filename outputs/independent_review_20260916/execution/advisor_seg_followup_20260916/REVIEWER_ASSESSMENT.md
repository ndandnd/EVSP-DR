**17 September external audit received:** the reviewer independently reproduces36 group/mixed values on12 inputs and checks all102 input identities. Its corrected proof scope agrees with our structural result. [Acceptance, remaining wording issues and reproducibility request](../advisor_audit_acceptance_20260917/README.md). The original assessment below is historical.

# Assessment of the independent reviewer

**Update after the new computation:** the exact time-only bounds now establish that group separation rules out k−1 in all nine cases. This supplies the lower-bound proof that the reviewer’s saved-pool experiment alone did not provide. The review below assesses the original experiment on its own evidence. See ../../time_only_vsp_20260916/independent_review.md for the new result and its continuous/event-grid scope.

The reviewer contributed a useful, inexpensive controlled experiment: remove mixed-group columns, keep all other saved column costs and coverage unchanged, and re-solve. The 12 archived JSON files support the reported numerical pattern: all nine k−1 route weights increase to k; the three controls stay at k. All solves report OPTIMAL, all unmixed supports have zero mixed weight, and the reproduced full-pool route weights agree with the recorded endpoints within 1e−5. These checks inspect the saved results; they do not rerun the LPs or independently validate each column.

This is strong evidence **within those saved pools**, and a good reason to run group-separated pricing. It is not yet a proof that group separation raises the full-model fleet bound by one.

| Claim | Assessment |
|---|---|
| Removing mixed columns changes the nine tested saved-pool LP route weights by +1 | Verified against all nine result files |
| Three saved-pool controls are unchanged | Verified against all three result files |
| GIRO is LP-optimal with segregation in all 102 instances and every group | Not established by these 12 restricted-pool solves |
| Integer per-group sums imply integer schedules | False in general; e.g. w5_k31 has 480 positive unmixed route variables summing to 31 |
| Ceil of each group's route weight in the weighted-cost LP is a fleet lower bound, even within the pool | Not established by this code: its objective is bus cost + charging, not fleet alone |
| Mixed compatibility is worth exactly one bus in the full LP model | Still awaiting unrestricted group-specific pricing or matching lower and upper bounds |
| The original service-overlap gap measures the electrification premium | Not yet: it also ignores deadheading; the requested time-only VSP isolates more of that gap |

For minimization, a restricted-column LP objective is an **upper** bound on its full-column LP objective. This relation applies to the optimized objective, not automatically to the auxiliary sum of route weights. A fleet certificate needs a fleet objective or a separately justified conversion from the weighted objective. A coefficient of 100,000 makes fleet dominant in these observed solutions but is not, by itself, a proof of lexicographic optimization.

The earlier audit already contains independently derived numerical fleet bounds. Those are separate evidence and are not replaced by the new weighted-pool route sums. The frozen 128/102/67/35 headline counts are unchanged by this follow-up.

## Delegation

Yes: keep using the reviewer for bounded independent checks, counterexamples, and cheap discriminating experiments. Retain a second check on proof scope and headline claims. The clear source paths, fast experiment and predictions are valuable; the concluding generalizations need correction before use in a paper.

The Claude reviewer is external to the available Codex subagents. A concrete follow-up request is saved in REVIEWER_HANDOFF.md for the user to relay. No claim is made that a message was delivered to Claude. The local computation and cluster scheduling are delegated to existing Codex agents in parallel under the user's authorization.

Source README, script and all 12 results are preserved verbatim under reviewer_source/ with hashes in source_receipt.json; review findings do not silently alter the reviewer's original report.
