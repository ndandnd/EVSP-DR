# Vehicle groups and time-only fleet bounds

**F4/F2 VERIFIED: group separation rules out k−1 in all nine cases.** The new exact time-only relaxation needs k buses when groups are separated, in all 102 instances and in each group individually. Allowing mixing gives k in 93 cases and k−1 in exactly the same nine as the saved LP endpoints.

| Question | Result | Scope |
|---|---|---|
| Does group separation exclude the nine k−1 solutions? | Yes, exact time-only lower bound k | Holds without needing an energy argument |
| Does time-only scheduling explain the 1–3 local-route gap above service overlap? | Yes; deadhead compatibility raises the local minimum to its GIRO count | Not an identified electrification premium |
| Is a mixed integer electric-bus solution at k−1 known? | No | Time-only paths need not satisfy energy constraints |
| Is GIRO k optimal with groups separated? | Continuous baseline physics: yes, with the independently linked F4 duty witnesses | Event-grid integer optimality needs an event-feasible k-bus schedule |

[All 102 results and exact proof](../../time_only_vsp_20260916/README.md) · [Independent certificate audit](../../time_only_vsp_20260916/independent_review.md). The implementation uses production travel sources, permits shortest deadhead paths through reference locations, and ignores energy and depot restrictions. This enlargement makes the lower bound conservative. All 612 matching certificates and 306 reachability antichains pass; all 1,040,239 saved LP route connections fit the relaxation.

## Six arms and full Partille

The six action3 arms continue unchanged. Their predictions are frozen in `predictions.json`, tied to the reviewer's original README hash. [Read-only comparison](CHECKING.md) separates partial replay, final CG weight/certificate, integer fleet/proof and physical validation. At the 01:19 UTC check all six final predictions remained pending; completed control shards had 6,144 feasible sequences and no anomalies. Partial losses in stricter arms are not fleet outcomes.

[Full-Partille CG343119](../full40_12h_scaglione_20260916/README.md) replaces held CG341405: Scaglione, 120 GiB, 8 CPUs, exact 12-hour allocation, 11h45m CG. It waits for unchanged graph341404_0; downstream MIP341406 remains held and points to the replacement. Only this authorized replacement was submitted. Resource estimates and checkpoint/shutdown evidence are recorded separately from scientific success.

## Reviewer and document

The reviewer's cheap controlled pool experiment was valuable and pointed to the correct structural mechanism. Its global proof claims exceeded what restricted weighted-pool LPs alone establish. The new independent time-only certificates supply the missing lower-bound argument. [Assessment](REVIEWER_ASSESSMENT.md) preserves that distinction; [bounded Claude follow-up](REVIEWER_HANDOFF.md) is ready for the user to relay. No message was sent to the external Claude session.

The current Google Doc has the new result, continuous-versus-event distinction, source links and full-Partille job status. `doc/doc_verification.json` passes: all intended replacements, four editable tables, original six footer links and byte-identical figure/curve/history exports. Five PDF pages were visually inspected. Baseline 128/102/67/35 counts and the separate 70 target matches are retained; no Slides edits.
