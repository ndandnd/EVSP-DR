# Acceptance of the external time-only audit — 17 September

**Accepted: the audit corroborates the existing F2/F4 structural result. No numerical headline changes.** The reviewer reports independent matching calculations on 12 instances (36 group/mixed comparisons), plus consistency checks on all 102 CSV rows. This is useful external corroboration, not a second independent recomputation of all 102 instances or all 612 certificate contents. The reviewer's executable recomputation is only in its chat transcript; it has not been supplied here as a reproducible script.

Our new read-only check confirms all 102 input identities and group-bound counts against the audited chain table. Both supplied READMEs are archived verbatim with hashes in receipt.json. Existing internal certificate verification and physical-witness evidence remain the primary reproducible sources.

| Proposition | Finding | Status |
|---|---|---|
| Separating groups forces at least k buses, even without energy constraints, in all 102 cases | F2/F4 | VERIFIED by existing time-only certificates; external audit agrees |
| With baseline continuous charging, k feasible GIRO-duty witnesses attain that bound | F4 | VERIFIED using existing F4 witness audit; exact segregated fleet optimum k |
| Mixing is necessary for any k−1 solution in the nine cases | F2/F4 | VERIFIED; separated lower bound excludes it |
| A mixed, energy-feasible integer k−1 fleet has been found | F2/F4 | UNRESOLVED; time-only paths and fractional route weights do not supply it |
| All102 segregated event-lattice integer optima are k | F4 | UNRESOLVED; continuous witnesses need not lie on the event grid |
| Local duties require an extra 1–3 buses because of electrification at baseline | F4 | REFUTED for the stated separated continuous baseline; deadhead-aware lower and feasible upper bounds both equal k |

## Remaining wording corrections in the supplied reports

The reviewer files are preserved as supplied. These qualifications govern our use of them:

1. **Restricted LP terminology.** The minimized weighted restricted-pool objective is an upper bound on the corresponding full-column LP optimum. Its route-weight sum is a separate quantity, not automatically a bound on either weighted optimum or fleet optimum. The corrected README still has two imprecise sentences about route weights being upper bounds on the pool's LP.
2. **Energy and the paper claim.** Say: “Energy does not increase the minimum fleet in these 102 group-separated baseline continuous instances.” Do not say the entire EV problem reduces to a VSP: charging cost and feasibility still require energy decisions, and an arbitrary minimum time-only path cover need not be energy feasible. Mixed integer k−1 feasibility also remains open.
3. **Old premium prediction.** The superseded overlap section retains its old interpretation, and the final “Suggested next cheap experiment” still predicts a local 1–3-bus premium. That baseline experiment is already complete and found zero in the separated continuous model. Those paragraphs are historical, not current work orders.
4. **Sensitivity logic.** A stricter power/reserve arm moving the LP would show an additional constraint matters under changed physics. It would not refute the proved necessity of mixing for baseline k−1 solutions. Check the frozen arm predictions separately from the structural theorem.
5. **Practical saving.** “One bus” is the time-only/fractional opportunity in nine cases, not an achieved integer operator saving. It remains a research question until a feasible electric schedule is validated.

## Reviewer usefulness and next handoff

The audit is valuable: it checks source identity, reproduces graph/matching values independently, and now distinguishes continuous feasibility from event-grid representability. Continue delegating bounded verification work to the reviewer, with explicit proof scope and saved code/results. Next cheap handoff: export the independent recomputation script and its 36-cell results from the transcript, and remove or clearly retire the stale paragraphs identified above. This note is a handoff for the user; no message was sent to the external reviewer.

No new jobs, retries, solver calls or experiment-setting changes. The six authorized arms continue under the existing monitor. Current Doc scientific claims already use the required continuous/event distinction; its numerical claims, figures, history and Slides are unchanged.
