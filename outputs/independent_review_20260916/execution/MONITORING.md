## Latest authorization — time-only VSP and full-Partille follow-up

Before failure notifications, check `monitor/notified_failures.json`; jobs341179 and341187 OOM failures have been reported. Do not repeat unchanged alerts.

The user requested the per-group time-only VSP for all 102 audited instances and authorized reducing the full-Partille CG request to 12 hours on Scaglione with justified memory. This supersedes the earlier conditional full-CG hold, but not the held downstream MIP or unrelated submission restrictions. The existing six full action3 arms continue; do not duplicate them.

Compare those arms against the frozen reviewer predictions with `python3 outputs/independent_review_20260916/execution/advisor_seg_followup_20260916/check_predictions.py --refresh` (read-only). This replaces a separate call to the action3 collector for that polling cycle. Missing or uncertified endpoints are not refutations. Keep partial replay counts distinct from completed CG/MIP results. See `execution/full40_12h_scaglione_20260916/` for the effective job, exact budget and checkpoint wrapper; never release old held commands blindly.

The time-only audit is `outputs/independent_review_20260916/time_only_vsp_20260916/`. Its model permits shortest deadhead paths and ignores energy; distinguish time-only fleet certificates, continuous GIRO-duty feasibility, event-grid representability and weighted-CG certificates. Preserve all baseline counts and model labels.

Latest full-arm campaign: also run `python3 outputs/independent_review_20260916/execution/action3_full_20260916/collect.py` each hourly check. Read action3_full_20260916/MONITORING.md once. All six arms and downstream jobs are already submitted; do not submit duplicates. Preserve the new concise Doc rewrite and read its before/after verification record.

## 16 September — approved sequencing (supersedes the earlier blanket submission hold)

Only these new submissions are authorized: (a) one action3 timing pilot; after it passes source/physical/runtime gates, the remaining five timing-pilot arms. No full replay or downstream CG/MIP rollout is implied. (b) Action2 remains UNsubmitted: plan seed0 with afterok on each item10 CG; seeds1/2 only on valid seed0 target misses. (c) Hold p3_full CG/MIP; implement/test checkpoint fallback. Resubmit fullCG on Scaglione only if evidence supports memory request<=120G; otherwise keep held and report. Preserve graph work already running and all historical held/V2G jobs. (d) Keep Doc labels:67/102 numerical fleet-bound matches and35/102 open fleet gaps. (e) No other submissions until P1 items7/8/9 and action1 have endpoints; then notify with four results: fresh-k15 hits/18; C5k31 12h outcome; constrainedk5 three-arm costs; k32 seed outcomes/variance. Endpoints failing validation are reported separately, never counted as hits. This endpoint gate does not itself authorize action2 submission or an unspecified new campaign.

Details: outputs/independent_review_20260916/execution/advisor_sequence_20260916. Hourly collection remains read-only except for the explicitly gated actions above. No repeat queue-only notices.

Current action3 status: control342321 passed; all five authorized follow-ups342380–342384 are already submitted. Each hourly check also runs `python3 outputs/independent_review_20260916/execution/advisor_sequence_20260916/single_factor_pilot/monitor_pilot.py`. Collect only; do not resubmit or expand. The tested fallback is staged in `review_full40_20260916/checkpoint_v2/`, but held old jobs retain v1 commands and must not be released blindly.

## Historical — earlier submission hold, superseded by the sequencing above

Only the six prepared k=5 F6 solver jobs (three tariffs × fixed-duty/fresh CG) are authorized for new submission. Existing campaigns may continue. Do not submit the item10 MIP follow-ups, item11 single-factor arms, other new jobs, replacement attempts, or partition changes until the user confirms after seeing current cluster load. Read-only collection, auditing, planning, documentation and code preparation may continue. This restriction supersedes older instructions to keep launching ready work. Preserve held jobs and EVSPV2G experiments. See execution/advisor_followup_20260916 for plans and receipts.

# Focused review monitoring

Run once each hour while the review campaigns are active:

```sh
python3 /Users/nadan/Documents/projects/demandresponse/outputs/independent_review_20260916/execution/collect_review.py
```

The validated run at **2026-09-16 22:18 UTC** completed in **2.99 seconds**. All six campaign collectors and both scheduler queries succeeded. `monitor/latest_path.txt` points to the latest dated snapshot; that directory contains `snapshot.json`, `receipt.json`, `squeue.txt` and `sacct.txt`. Collection failures produce `failure.json`; partial collector/query failures are marked `partial_collection`.

| Campaign | Review question |
|---|---|
| P1 MIP repeats | F5 fresh pools; F2/F4 warm-pool misses |
| Minimum-duration tariff comparison | F6/F7 charging response |
| Strict chain 5 | F4 depot power, SOC reserve and vehicle groups |
| Random trip groups | F5 warm starts without GIRO duty grouping |
| Full Partille 40 | F8 full-instance feasibility and bound |
| Frölunda ladder | Generalization (review item 14) |

The collector queries only registered review jobs and their recorded attempts. It invokes each campaign's small native collector; it does not use the old full-project collector, launch jobs, cancel jobs, or resubmit failed jobs. Scheduler state and attempt receipts are kept separate from pricing certificates, finite-pool fleet proofs, physical replay and duplicate-trip removal checks. A successful collection is **not** a scientific verification. A running attempt has incomplete elapsed time; retain completed and interrupted attempts when calculating total effort.

Only the synthetic peak08/12/18 tariff rows are transferred from the demand-response campaign. All SE3 prices, scientific outputs, costs and private attempt details remain on Unicorn. The native internal collection is saved under `/home/nc437/ladder-lite/review_monitor_20260916/<UTC stamp>/dr/private_internal/`. Do not copy that directory into Git, the document, or local reports. Scheduler metadata may identify those jobs but contains no prices or outcomes.

For the hourly monitor: stay quiet when status is unchanged; notify on meaningful completion, worker failure, lost SSH access, or a blocked dependency requiring action. If access fails, say so promptly. Inspect the relevant native evidence before updating the research register or Google Doc. Do not infer a full-model fleet proof from a finite-pool result. Do not restart jobs from this runbook; investigate the recorded attempt first. Preserve held historical jobs and other projects.

An hour is sufficient for the multi-hour CG and MIP budgets. Extra checks are useful after a reported failure or completion, not to fill idle time.

## How to interpret the next results

- **F5 / item 7:** the review's trigger is a matching seed in at least four of six fresh k15 pools. Report every seed and the old 30-minute versus new three-hour fleet budget. Do not call this an isolated causal effect of inheritance.
- **F2/F4 / item 8:** 30 buses in the LP does not imply a 30-bus integer solution. A 12-hour timeout also does not exclude one.
- **F2/F5 / item 9:** include the four reused seed-zero cells once. Verify ordered-pool and initializer identity before attributing differences to seed or search allowance.
- **F6/F7 / item 10:** distinguish original invoice estimates from feasible schedules. Check active charging durations, per-bus SOC, attained ending energy and trip assignment. A phase-I stop is not combined-objective pricing convergence. Joint CG may miss columns present in fixed-duty alternatives; no savings are guaranteed.
- **F4 / item 11:** each prefix needs both vehicle-group components. This changes several physical assumptions and the pricing driver, so it does not isolate a runtime cause. Shared capacity, nonlinear power, minimum-duration/setup rules and terminal SOC are still omitted.
- **F5 / item 12:** intermediate stage numbers are not GIRO fleet targets. Account for graph, initialization/CG, MIP and interrupted attempts separately. Only the final trip set equals C1 k15.
- **F8 / item 13:** reconstruct any F3 bound from this input's paired LP/pricing records and cost envelope. The 48-hour CG clock follows separate graph preparation.
- **F8 / item 14:** preserve the declared conservative Frölunda deadhead conversion. Early small-case success does not establish full generalization.

Record a tested proposition as **VERIFIED**, **REFUTED**, or **UNRESOLVED** in `ledger.json` and the execution README. Missing proof is not proof of the opposite. New native flags are first recorded as such; verify source hashes and scientific validation before promoting a headline.

The current Doc's source is `doc/current.html`. The dashboard builder now reads it; do not regenerate superseded claims from historical prose. Update the current tab in place, retain figure/history tabs, and do not edit Slides. The existing workbook predates the added audit fields; use the audited CSV, then rebuild the workbook when a useful batch of endpoints is verified.

Publish through the worktree named in `outputs/research_register/publication_workspace_path.txt`, preserving the dirty main checkout. Regenerate and inspect `../build_publication_allowlist.py`; never copy restricted real-price outputs or whole result trees blindly. Verify the Unicorn research-register mirror after publication. The original F1 equality-pool and shared-capacity flags remain distinct from the newly validated passenger assignment with empty driving.

## Four-result endpoint gate — read-only

After each normal hourly collection, run `advisor_sequence_20260916/report_gate/four_numbers.py` against the snapshot named by `monitor/latest_path.txt`; write into `advisor_sequence_20260916/report_gate/snapshots/<UTC stamp>`. See that directory's README for the command. The collector now covers seven campaigns, including constrained k5 F6, and explicitly tracks all four reused P1 seed-zero jobs.

Do not announce partial target-hit totals as the requested four-result briefing. Wait for all37 required solver cells to have endpoints or explicit terminal failures. The gate emits `items: null` while pending or if scientific/control checks fail. Terminal failures remain censored and are listed separately; no automatic retry is authorized. When ready, report the four requested results with validation limits; readiness does not authorize any new submission. This report gate itself performs no network access or scheduler mutation and does not broaden the separately authorized actions at the top of this runbook.
