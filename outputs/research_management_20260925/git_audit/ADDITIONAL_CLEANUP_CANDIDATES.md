# Additional local branch cleanup candidates

Historical shortlist snapshot: five `eligible` candidates were subsequently approved, archived by named tags, and removed with nonforce deletion. See `additional_cleanup_receipt.json`. The ten remaining candidates were not deleted. Each tip is an exact ancestor of the listed local and remote branch at the same SHA, has no active worktree, and has no commits unique relative to that retained branch. Preserve branch-name provenance with the proposed archive tags in the JSON before any approved deletion. `git branch -d` eligibility is predicted from its configured upstream (if it exists), otherwise main; no force deletion is proposed.

| Candidate | Retained local and origin branch | Retained SHA | Same-name remote | git branch -d guard |
|---|---|---|---|---|
| codex/capacity-efficiency-20260912 | codex/strict-graph-validation-20260922 | 3bb32c1a84 | True | not merged into refs/heads/main |
| codex/event-native-warm-chain-20260909 | codex/zero-fee-terminal-cg | 2c7445ac64 | False | not merged into refs/heads/main |
| codex/event-uniform-cluster-tools-20260824 | codex/strict-graph-validation-20260922 | 3bb32c1a84 | True | eligible |
| codex/giro-k1-recovery-20260909 | codex/strict-graph-validation-20260922 | 3bb32c1a84 | False | not merged into refs/heads/main |
| codex/gurobi-primary-20260906 | codex/strict-graph-validation-20260922 | 3bb32c1a84 | True | eligible |
| codex/highs-baseline-20260906 | codex/strict-graph-validation-20260922 | 3bb32c1a84 | False | eligible |
| codex/issue18-clean | codex/strict-graph-validation-20260922 | 3bb32c1a84 | False | eligible |
| codex/k5-raw-mip36h-status-fix-20260905 | codex/strict-graph-validation-20260922 | 3bb32c1a84 | False | not merged into refs/heads/main |
| codex/nested-threshold-k2-15-20260908 | codex/strict-graph-validation-20260922 | 3bb32c1a84 | True | not merged into refs/heads/main |
| codex/overnight-extension-20260912 | codex/zero-fee-terminal-cg | 2c7445ac64 | True | not merged into refs/heads/main |
| codex/overnight-pool-mip-20260908 | codex/strict-graph-validation-20260922 | 3bb32c1a84 | True | not merged into refs/heads/main |
| codex/queue-graph-recovery-20260912 | codex/zero-fee-terminal-cg | 2c7445ac64 | True | not merged into refs/heads/main |
| codex/queue-recovery-code-20260912 | codex/zero-fee-terminal-cg | 2c7445ac64 | True | not merged into refs/heads/main |
| codex/research-register-20260910 | codex/week-evidence-20260921 | d04b11876c | True | not merged into refs/remotes/origin/codex/research-register-20260910 |
| codex/small-exact-threshold-20260903 | codex/strict-graph-validation-20260922 | 3bb32c1a84 | False | eligible |
