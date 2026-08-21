# Curator notes — 2026-08-21

Curator branch: `cursor/records-curator-2969-8502`, based on
`records/ladder-lite-20260819-2969@739d8dd` with the freeze commit `0d408af`
cherry-picked. Work order: `CURATOR_WORKORDER_20260821.md`.

## 1. Freeze confirmation

**The shared ledgers `records/DECISION_LOG.csv` and `records/BUG_LOG.csv` are
frozen** at their `739d8dd` state: `B0001`–`B0031` (with `B0010`–`B0012`
allocated on the ladder-lite branch, see §3) and `D0001`–`D0033`. They are
unfrozen only through the curator. Feature-branch agents write findings to
`records/inbox/<branch>.md` with provisional `LOCAL-N` labels; authoritative
IDs are assigned only via `records/ID_REGISTRY.csv`.

## 2. Reservation

All future allocations come from **`B0100+` and `D0100+`**.

The ranges `B0032`–`B0099` and `D0034`–`D0099` are **retired and must never be
assigned**: feature branches already used `B0032`/`B0033` (twice each, for
different findings) and `D0034`/`D0035`, so any assignment inside the gap
could collide with an existing branch citation.

## 3. `B0010`–`B0012` are honored, not colliding

The authoritative `BUG_LOG.csv` jumps from `B0009` to `B0013`. The gap IDs
exist only on `cursor/ladder-lite-20260819-2969` and mean exactly one thing
each anywhere in the project, so they are honored as allocated and recorded in
the registry under their original IDs. They are not remapped.

## 4. Collision adjudication (deliverable 2)

`records/ID_MAPPING.csv` maps every colliding `(branch, old_id)` pair to its
canonical ID — 23 rows. Two corrections to the work-order §2 table, from an
exhaustive row-by-row comparison of every branch ledger (tips and pre-removal
states) against the authoritative ledgers:

- the **event branch also squatted `B0032` and `B0033`** (sink-batching
  heuristic; G3 correction). They do not collide with the authoritative ledger
  but do collide with the branch-and-price branch's `B0032`/`B0033`, which
  name unrelated findings. Both sides are remapped.
- branch-and-price `D0034`/`D0035` collide with nothing today but sit
  immediately above the authoritative high-water mark `D0033`; the work order
  lists them as colliding and they are remapped with the rest.

Titles in the registry and mapping are **verbatim from the source branch**.
ID references inside a title (e.g. "Correction to B0024") are branch-local
and resolve through `ID_MAPPING.csv` (branch B0024 → canonical `B0101`;
branch-and-price B0031 → canonical `B0108`). The authoritative `B0022`
referenced by canonical `B0100` is the records-branch `B0022` (unchanged).

## 5. Inbox adjudication

Every inbox `LOCAL-N` entry was either given a canonical ID in the registry or
recorded here as already covered by an authoritative row. No entry was
silently dropped.

| branch | local label | disposition |
|---|---|---|
| ladder-lite | LOCAL-1 | canonical `B0100` |
| ladder-lite | LOCAL-2 | canonical `B0102` |
| ladder-lite | LOCAL-3 | canonical `B0103` |
| ladder-lite | LOCAL-4 | canonical `D0100` |
| ladder-lite | LOCAL-5 | already covered by authoritative `D0028` (cross-resolution union retired); no new ID |
| ladder-lite | LOCAL-6 | canonical `D0112` (producer claim; independent validation is deliverable 6) |
| fixed-duty | LOCAL-1 | already covered by authoritative `D0021`/`D0022` (G1–G5 pass, duty 13411 admitted); no new ID |
| fixed-duty | LOCAL-2 | canonical `B0104` |
| fixed-duty | LOCAL-3 | canonical `D0102` (instrument half) |
| fixed-duty | LOCAL-4 | canonical `D0102` (event-cap half) |
| fixed-duty | LOCAL-5 | canonical `D0113` |
| event | LOCAL-1 | canonical `D0114` |
| event | LOCAL-2 | canonical `D0115` |
| event | LOCAL-3 | canonical `D0116` |
| event | LOCAL-4 | canonical `B0106` (same finding as branch `B0032`) |
| event | LOCAL-5 | canonical `B0111` |
| branch-and-price | LOCAL-1 | canonical `D0117` |
| branch-and-price | LOCAL-2 | canonical `D0118` |
| branch-and-price | LOCAL-3 | canonical `D0119` |
| arc-flow | LOCAL-1 | already covered by authoritative `D0025` (3/3/3, excess 1/1/4); no new ID |
| arc-flow | LOCAL-2 | already covered by authoritative `D0026`/`D0030` (LP equality + audit); no new ID |
| arc-flow | LOCAL-3 | already covered by authoritative `D0030` caveat (witness acceptance rule); no new ID |
| arc-flow | LOCAL-4 | canonical `D0120` |
| arc-flow | LOCAL-5 | canonical `D0121` |
| arc-flow | LOCAL-6 | canonical `D0122` |
| arc-flow | LOCAL-7 | canonical `D0123` |

Registered findings remain **branch-local claims at their producing commits**;
registration assigns an identity, not a validation verdict.

## 6. Flags for the operator (no ledger edit; ledgers frozen)

1. **`D0117`/`D0118` answer STATUS §6 open question 7**: `run_exact_pool_mip.py
   --progress-dir` does **not** resume and does not reload an incumbent; it is
   observational only. This weakens the `D0033` rationale clause "MIPs
   checkpoint to `--progress-dir` with a recovery script" as a reason to keep
   large pool MIPs off scaglione; `D0119` records the branch's own conclusion
   that the largest MIPs still warrant scaglione. The `D0033` rule of thumb
   itself already carries the right exception (long + unresumable + critical).
2. **`EVSP_DR_HANDOFF_20260819.md` does not exist in this repository's git
   history on any branch** (checked all refs). STATUS §9 item 8 and §2 of the
   status preamble cite it; the citation is annotated in the corrected STATUS
   rather than silently dropped. If a copy exists outside git, it should be
   committed and hashed.
3. The authoritative `B0031` correction ("commit the CSVs") is executed:
   ladder commit `72c7bf4` committed the 18 duty-union instances. Independent
   validation is recorded in `analysis/duty_union_validation_20260821/`.

## 7. Full-suite run and one superseded import

First full-suite run on this branch (pinned stack, Python 3.12.3, Linux VM):
**518 passed, 126 subtests passed, 1 failed** —
`test_fixed_duty_continuous_optimizer.py::RealDutyContinuousGates::
test_all_k2_k3_k5_duties_replay_under_both_tariffs`, `ValueError: replay SOC
bound violation after direct deadhead`, with the fixed-duty branch's version
of `src/fixed_duty_continuous_optimizer.py` (producer `70c88f4` lineage).

This is a known, fixed defect: the event branch's committed report
(`analysis/event_based_pricer_20260821/REPORT.md@92dd6c1`) states the
continuous fixed-duty tests pass "after retaining solver-tolerance
micro-energy in contiguous replay segments", and event commit `554fef32`
carries exactly that fix (a segment counts as an active charge event when the
binary indicator is on **or** the solved energy exceeds tolerance) plus a
23-line regression test. The failure is solver-tolerance-sensitive, which is
why the fixed-duty branch's own run reported green on its machine.

The curator therefore superseded the import of
`src/fixed_duty_continuous_optimizer.py` and
`tests/test_fixed_duty_continuous_optimizer.py` with the event-branch versions
(producing commit `554fef32fee7977f3fc6ee42aafe0c35939e39d1`);
`records/EVIDENCE_IMPORT_20260821.csv` records the final identities and this
branch's git history preserves the initial ones. Final full-suite run:
**520 passed, 126 subtests passed, 0 failed** (153.8 s).

Caveat for the operator: the fixed-duty factorial evidence
(`analysis/fixed_duty_continuous_20260820/`, `D0113`) was produced with the
pre-fix extractor on the producer's machine, where its own replay gates
passed. The fix changes replayed-schedule extraction only, not the MILP
objective; the factorial cost numbers are solver-objective quantities and are
unaffected, but any future re-run should use the fixed extractor.
