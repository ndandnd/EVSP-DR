# Git audit and conservative cleanup — 25 September 2026

Completed metadata cleanup and removed seven redundant local branch names; created one isolated maintenance branch. No source merge, force branch deletion, remote deletion, push, checkout change in the primary workspace, or deletion of worktree directories/files was performed.

## Verified changes

| Item | Before | After cleanup | After maintenance checkout |
|---|---:|---:|---:|
| Local branches | 88 | 81 original branches remain | 82 including new maintenance branch |
| Registered worktrees | 81 | 37 | 38 |
| Prunable worktree entries | 44 | 0 | 0 |
| Remote-tracking refs | 67 | 67 | 67 |

First deleted `codex/capacity-representation-pilot-20260922` and `codex/charging-column-structure-20260922` with `git branch -d`. Both were exact aliases of primary `main` at `f37c6ce4c6e0a7ccffd3719a633bacd40b9a351c`, had no worktrees or unique commits, and remain retained by `main` plus `origin/codex/charging-column-structure-20260922`.

Pruned only 44 metadata directories whose `gitdir` paths no longer existed. The dry-run list was checked exactly against the audited invalid metadata entries. Eight corresponding filesystem directories still existed without `.git`; all eight remain intact. Archived all 44 metadata directories in `prunable_worktree_metadata_before.tar.gz` (2,035,413 bytes, SHA256 `69c145ad83070e0fb93fae2b021c69fe5f1849f21022663c2a2f27f51ffa3b18`).

Before pruning, four detached tips and 15 additional commits present only in worktree reflogs were retained under `refs/tags/codex/archive-worktree-20260925-<full-sha>`. All 19 references were individually verified. This prevents loss of research history after future garbage collection. The metadata archive preserves the former index and reflog records as well. The tags were later backed up to origin; the successful upload receipt and saved remote-ref check are included in this publication packet.

## Primary workspace and source preservation

The primary workspace remains on `main` at `f37c6ce4c6e0a7ccffd3719a633bacd40b9a351c`. It is four commits ahead of `origin/main`, whose tip is still the February commit `c4002c58f1beb50d1b793154e12bfdd9412217d2`. Root reported a successful origin fetch; the audit verified that its 67 observed remote ref SHAs were unchanged after that fetch.

Main had two tracked edits and 632 collapsed untracked status entries at the initial snapshot. Both tracked file hashes were verified unchanged after cleanup:

- `outputs/meeting_20260910/collect_remote.py`: `8e8d6b556cda0519ea1d050bf651bd131657e66ede545cf7f8b54c6f2de6db11`
- `outputs/research_register/build_register.py`: `62b1e93e01d62a7cf1a287fe1dcf9c48037a359662bf4c7b52f20b2739e13f08`

Their original working-tree diff is preserved in `tracked_main.patch`; the staged diff is empty. The other tracked dirty worktrees were `capacity-shortcircuit-20260917` (one research README) and `/Users/nadan/Documents/projects/evsp-dr-nested-auditor` (one README, two untracked entries). Neither was modified. Active experiment branches and all unique branch tips remain available. The initial inventory identifies 26 branch tips with commits not reachable from any origin ref; absence of an upstream alone was never treated as permission to delete.

## Maintenance checkout

Created `.codex-work/research-maintenance-20260925` on branch `codex/research-maintenance-20260925` from `origin/codex/week-evidence-20260921` at `d04b11876cb38866b524c3793c8aff996545e3b3`. Initial checkout was clean. Exact noncone sparse patterns are:

```text
/outputs/research_management_20260925/
/outputs/research_register/entries/
/outputs/research_register/README.md
```

Only nine preexisting tracked files, totaling 97,152 bytes, were populated. This isolates evidence publication from primary workspace changes. No unrelated primary-workspace files were copied. Root is responsible for any subsequent evidence commit or push.

## Source consolidation assessment

There is no nontrivial fast-forward from `main` to the current solver source branches. Modern solver lineages differ from main by ten main-only commits and roughly 380–400 branch-only commits, with substantial historic file retirement and experiment-specific settings. A bulk merge into main is not a safe housekeeping change.

Retain these distinct current source lineages:

| Purpose | Tip | Relationship and preservation |
|---|---|---|
| Spatial tariff / terminal CG | `4a8b497e668be9962bca5906a8b69116fa634882` | `codex/spatial-tariff-recipe-20260925`; two local commits after `codex/zero-fee-terminal-cg` (`2c7445ac`). An isolated checkout from zero-fee could fast-forward here; no merge or scientific validation is implied. |
| Strict graph validation | `3bb32c1a84af73c97689dfc9f5ad43136cfcead9` | Retained by matching local and origin `codex/strict-graph-validation-20260922`; contains the strict reuse/lineage ancestry. |
| Dive cap controls | `d8fbf40b922c158d53959a12ac950371e7c7c6fe` | `codex/dive-cap-escalation-20260923`; one local commit after `codex/integer-columns-20260921` (`c50e5f20`). |
| Research evidence | `d04b11876cb38866b524c3793c8aff996545e3b3` | Matching local and origin `codex/week-evidence-20260921`; chosen maintenance base. |

A future functional integration should use an isolated source checkout, preserve all execution SHAs, inspect exact file overlap, and validate relevant parser/configuration defaults, saved-pool input lineage, pricing certificate behavior, physical replay, and experiment-specific tests. Housekeeping alone does not justify solver runs or merging branches that represent different physics. No new tests were needed for this metadata-only cleanup; verification used ancestry, reference reachability, exact dry-run matching, hashes, status, and counts.

## Additional reviewed cleanup

The initial shortlist (`ADDITIONAL_CLEANUP_CANDIDATES.md`, `additional_cleanup_shortlist.json`) identified 15 older codex branches with no active worktree and exact ancestry into a retained local+origin modern branch pair. After review, five eligible branches were archived under named tags and removed with `git branch -d`: `event-uniform-cluster-tools-20260824`, `gurobi-primary-20260906`, `highs-baseline-20260906`, `issue18-clean`, and `small-exact-threshold-20260903` (all with `codex/` prefix). Each tip remains in local and origin `codex/strict-graph-validation-20260922` at `3bb32c1a84af73c97689dfc9f5ad43136cfcead9`. Exact former branch/tag/SHA mappings are in `recovery_refs.tsv`; every nonforce deletion is recorded in `additional_cleanup_receipt.json`.

The remaining ten shortlist branches were not deleted. Their ordinary deletion guard is main or a divergent upstream, so they require separate review; no force deletion is proposed. All 24 newly created archive tags were subsequently backed up to origin; `archive_tags_to_back_up.txt` lists those exact references. `archive_push_receipt.json` records the successful upload, and `../archive_ref_verification.json` verifies equality between the saved remote refs, recovery list and current local tags. No solver branch was pushed by this maintenance preparation.

Generic issue branches, active worktrees, all remote branches, archived experiment branches, unique commits and main’s uncommitted work remain intact.

## Recovery

The two deleted local aliases can be recreated without touching files:

```sh
git branch codex/capacity-representation-pilot-20260922 f37c6ce4c6e0a7ccffd3719a633bacd40b9a351c
git branch codex/charging-column-structure-20260922 f37c6ce4c6e0a7ccffd3719a633bacd40b9a351c
```

Recover any protected detached commit in a new location with its retained tag, replacing the placeholders with a tag SHA and an unused path:

```sh
git worktree add --detach /private/tmp/recovered-worktree refs/tags/codex/archive-worktree-20260925-<full-sha>
```

For index/reflog inspection, extract the metadata archive into a separate review directory first. Do not blindly restore stale `.git/worktrees` registrations over current metadata. The archive records are intentionally available even though the corresponding gitdir paths were invalid.

## Evidence files

- `inventory.json`, `branches.tsv`, `refs.txt`, `worktrees.txt`: full pre-cleanup inventory and per-branch ancestry/upstream/source preservation assessment.
- `cleanup_receipt.json`: every executed Git command and result, backup tags, directory preservation, before/after counts and dirty-file hashes.
- `prunable_metadata_audit.json`: invalid gitdir paths, reflog preservation analysis and byte sizes.
- `prune_dry_run.txt`, `prune_executed.txt`, `worktrees_post_cleanup.txt`, `refs_post_cleanup.txt`: exact cleanup evidence.
- `maintenance_worktree_receipt.json`: sparse-checkout creation, selected bytes, source SHA and initial clean status.
- `post_fetch_ref_check.json`: verification after root’s fetch.
- `additional_cleanup_shortlist.json`: initial recommendation and retaining references; `additional_cleanup_receipt.json` records the five subsequently approved deletions.
- `recovery_refs.tsv`, `archive_tags_to_back_up.txt`: all 24 new archive tags with exact SHAs and former branch mappings.
- The metadata tarball and full unabridged maintenance receipt are local recovery artifacts; the latter is retained at `/private/tmp/demandresponse-maintenance_worktree_receipt_full_20260925.json`. The publication receipt replaces its full-tree stdout with a SHA256, byte count and line count.


## Publication scope note

The metadata archive, main-workspace patches and complete-local-tree checksum manifest are intentionally local-only. They are retained in the primary workspace at `outputs/research_management_20260925/git_audit/`; the packet contains their recorded hashes and recovery descriptions, not their bytes. The final archive-tag upload and exact remote SHA verification are recorded in `archive_push_receipt.json`, `archive_remote_refs.txt` and `../archive_ref_verification.json`. The compact receipt is sufficient for inspecting maintenance decisions; recovering preexisting main edits requires the local patch.
