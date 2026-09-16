# Repeated-trip inventory in selected target-32 fleets

Read-only audit, 16 September 2026. Counts below are computed directly from selected route trip lists. Source result and input CSV hashes are recorded in [results.json](results.json); [screen.py](screen.py) reproduces the inventory on Unicorn. This is not a duplicate-removal algorithm, a new physical replay, or an infeasibility proof.

| Chain | Selected buses | Distinct input trips | Trip IDs on multiple buses | Extra service occurrences | Most repeated-trip IDs on one route |
|---|---:|---:|---:|---:|---:|
| 1 | 35 | 768 | 206 | 306 | 36 |
| 2 | 32 | 751 | 161 | 212 | 26 |
| 3 | 33 | 753 | 120 | 147 | 22 |
| 4 | 32 | 749 | 126 | 151 | 23 |
| 5 | 32 | 731 | 114 | 131 | 23 |
| 6 | 32 | 779 | 116 | 142 | 23 |

Each selected route contains no repeated trip ID internally. Counts of distinct covered IDs equal the input-row counts; input hashes match the campaign manifest and saved physical-pool audit. This count check is not a new independent identity-by-identity physical validation. Existing individual-route replay and duplicate/shared-capacity validation flags remain separate.

The existing `terminal_duplicate_cleanup.py` enumerates deletion subsets and explicitly stops when a route has more than ten repeated-trip IDs. All six selections exceed that limit. For chains 2, 4, 5 and 6, the simple sum of 2^d over routes is approximately 157.6 million, 10.8 million, 11.2 million and 16.4 million subsets before deduplication, feasibility pruning or charging alternatives. These are counts for that enumeration strategy, not lower bounds on the work required by a better method.

Next design to evaluate: a bounded assignment/deletion heuristic followed by fixed-sequence physical and charging replay, with an exact repair formulation on failures. Preserve the selected fleet cap, source physics/objective, exact-once trip assignment and source hashes. A successful replay can validate that schedule; a failed heuristic cannot prove that the fleet is impossible. No such repair was executed by this screen. Simply raising the existing enumeration limit is not an approved production design.

[Collected-source verification](collection_verification.json) matches all six native selections to the collection. Chain 1’s canonical and attempt files have different byte serialization but identical parsed JSON; the other five match byte-for-byte.
