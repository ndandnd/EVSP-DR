# Finding inbox

Feature-branch agents write findings here instead of appending to the shared
ledgers, because branches reused authoritative IDs for unrelated findings.

One file per branch: `records/inbox/<branch-name>.md`. One entry per finding:
a provisional local label (`LOCAL-1`, `LOCAL-2`, ...), the claim, the evidence
path, and the producing commit SHA. Never an authoritative `B####`/`D####` —
the curator assigns those via `records/ID_REGISTRY.csv`.
