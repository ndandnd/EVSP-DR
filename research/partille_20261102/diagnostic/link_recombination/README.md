# Passenger-link recombination diagnostics

## Post hoc GIRO-witness membership

The authenticated 39,390-route Chain 2 k8 archive contains **161 of 165 distinct adjacent passenger-trip links** from the independently regenerated eight GIRO witness sequences. Four links are missing. **Seven of eight full witness sequences are absent**, reproducing the earlier membership result; **three of those seven** nevertheless have every adjacent passenger link available somewhere in the archive. The single present full sequence occurs in four route records.

| Witness ordinal | Passenger trips | Present links | Missing links | Full-sequence records |
|---|---:|---:|---:|---:|
| 1 | 50 | 49 | 0 | 0 |
| 2 | 14 | 13 | 0 | 4 |
| 3 | 15 | 13 | 1 | 0 |
| 4 | 14 | 12 | 1 | 0 |
| 5 | 24 | 23 | 0 | 0 |
| 6 | 16 | 14 | 1 | 0 |
| 7 | 22 | 20 | 1 | 0 |
| 8 | 18 | 17 | 0 | 0 |

Witness ordinals follow first occurrence of their duty in instance row order; actual trip and duty IDs are withheld. The eight sequences partition all 173 passenger trips exactly once and match the saved independently replayed witness selection. The pool has 1,876 distinct adjacent passenger links.

[Checker](posthoc_check_links.py) · [ID-free aggregate and input hashes](posthoc_aggregate.json).

The checker maps archive local row indices through the unique `Ordered_Trip_ID` field, regenerates passenger sequences from original `Regular` duty rows, and authenticates the decompressed archive, root prefix, summary, instance, duty source and saved witness. Source/sink and charging nodes are excluded. An adjacent passenger pair counts even when its route charges between those passengers.

Run from a private working checkout with the original input layout:

```sh
python3 outputs/partille_validation_20261102/diagnostic/link_recombination/posthoc_check_links.py \
  --repo . --out /private/tmp/evsp-nov02-posthoc-links.json
```

The public research-package path is `research/partille_20261102/diagnostic/link_recombination/`; the corresponding private working-checkout path is `outputs/partille_validation_20261102/diagnostic/link_recombination/`. Source CSVs, the archived journal and saved witness selection are intentionally withheld. Reproduction requires these hash-matched private inputs; the public aggregate contains their SHA-256 hashes, no raw IDs or schedules.

The measured **1.330 s** starts before input authentication and includes hashing, CSV reconstruction, selected-sequence comparison, journal parsing and membership counts. It excludes exploratory file discovery, writing the checker and publication. No solver, graph construction, pricing or cluster job was used.

This is a post hoc diagnostic using the GIRO witness. Union-of-links availability does not establish that a full sequence can be concatenated in one archived route, that charging schedules are compatible, or a new physical/optimization certificate.

## Independent blind retrospective pilot

Using only the original priced-pool passenger sequences and canonical instance times, deterministic frequency/ID-tied bipartite matching recombines the **1,876 observed passenger links** into **eight paths**. A matching of size 165 and an equal-size vertex cover certify the minimum path count on this observed adjacency DAG. **All eight complete paths are new** relative to the 19,399 distinct complete sequences in the 39,390-record pool.

One deterministic path-order retiming attempt succeeds; separate final replay verifies **173 trips exactly once, zero plug-minute overloads, 29 charging sessions and 14.6310006 kWh aggregate grid terminal energy**. Expanded-grid nonfleet cost is **361.752**; continuously realized nonfleet cost is **343.2331426**, reported separately. Physics: 240 kWh battery/initial energy, zero reserve and zero terminal-energy target, 240 kW opportunity charging, **60 kW PARX**, 2.5 kWh SOC grid, five-minute charging discretization, five-unit charging-start fee and the existing finite minute plug limits (unlimited PARX).

No GIRO witness routes, duty labels or assignments were read for construction or retiming. The only construction and retiming attempt completed before an unsolicited post hoc aggregate message arrived; no subsequent path selection or retiming choice was made. This is observed-arc recombination with fixed-sequence retiming; no arc-fixing CG was executed. It is **one retrospective case using an expensive original pool**. It does not measure fresh-CG discovery time or establish general performance, charging-cost optimality, or a new CG/full-model pricing certificate. The eight-path certificate applies to the observed passenger adjacency DAG; energy and shared capacity require the separate replay checks.

| Measured stage | Elapsed | Scope / cap |
|---|---:|---|
| Matching and construction | 1.658 s | Input parsing/authentication, journal scan, matching and certificate, original-sequence comparison, private outputs; no configured cap |
| One retiming order | 5.097 s | Imports, input hashes, fixed-sequence graphs/searches, individual checks, occupancy and outputs; **180 s cap**, one ordering |
| Independent final replay | 0.941 s | Imports, matching/cover certificate, exact service, route energy/cost/charge-block and shared occupancy checks; no configured cap |

These are separate measured process scopes, each excluding interpreter startup; source discovery and historical pool-generation compute are outside them. The post hoc scan has its own clock above. The authenticated source summary reports **7,965.730 s (2.21 h) graph construction plus 8,465.745 s (2.35 h) root CG**, followed by later dive/MIP work. These historical costs are outside the short pilot clocks.

[ID-free blind aggregate and exact execution hashes](blind_aggregate.json) · [Matching CLI](blind_link.py) · [One-order retiming CLI](retime_once.py) · [Independent verifier](verify_pilot.py) · [Source hashes and adaptation record](reproduction.json).

### Predeclared root-prefix sensitivity

After the full-pool pilot, one root-only robustness protocol was declared and then executed: use only the authenticated first **28,253 root-CG records**, apply the identical deterministic matching and first path-order retiming, allow 180 seconds and no retries/tuning, then independently replay. No dive columns or witness labels were used; the previously received post hoc aggregate did not guide edge/schedule choices.

The prefix contains **1,753 observed passenger links** and **13,093 distinct complete sequences**. A size-165 matching and equal vertex cover again yield eight paths, all eight complete sequences new to this prefix. The first retiming order succeeds. Independent replay finds **173 exactly-once trips, zero overload, 30 charging sessions, 14.6310006 kWh grid terminal energy**, **374.688 grid nonfleet cost** and **354.8824193 continuously realized nonfleet cost**. Matching, retiming and replay take **1.060 / 2.923 / 0.810 s**, respectively; their measured sum is **4.793 s**, excluding source discovery, script preparation, orchestration and interpreter startup. Stage clock scopes and the 180-second retiming cap match the primary pilot above.

Thus the extra dive is unnecessary **for this selected repair**; root-CG columns already support a feasible eight-bus recombination. Root graph/CG historical cost remains required, and general performance/prospective success rates are untested. [Root-only aggregate](root_aggregate.json) · [Root-prefix matching CLI](root_blind_link.py).

### Tie-breaking and model scope

Successors are ranked by descending observed frequency then canonical input row ID; left vertices and initial path order are ranked by trip start time then row ID. In the root graph, 83 sources have equal-frequency successor ties (252 arcs in tied groups), and 37 trips share start times. Consistent semantic row permutations preserve the optimal graph path count but may change the chosen matching and first greedy retiming success. No row-permutation experiments were performed.

Both pilots use the **chain standard model**: 240 kWh battery/initial energy, zero reserve and zero required aggregate terminal energy, shared finite opportunity-station plugs, unlimited PARX plugs and PARX power 60 kW. Effective transfer settings are 57-minute trip-to-trip maximum and 1,560-minute station-to-trip wait. `problem_kwargs()` explicitly sets the latter and retains the former `build_problem` default. These are distinct from the five-bus 236.44 kWh / 15% reserve comparison and do not validate its different rules.

### Private reproduction

Requires the hash-matched private pool/instance and solver source modules listed in `reproduction.json`. Set `ARCHIVE`, `INSTANCE`, `DATA_DIR` and `SOLVER_SRC` to those local inputs. Set `CODE_DIR` to this public package directory. Use an output directory **outside the public repository**, because generated graph certificates, paths, native receipts and schedules contain private records.

```sh
PILOT_OUT="$(mktemp -d)"
python3 "$CODE_DIR/blind_link.py" \
  --pool "$ARCHIVE" --instance "$INSTANCE" --out "$PILOT_OUT"
PYTHONPATH="$SOLVER_SRC" python3 "$CODE_DIR/retime_once.py" \
  --data "$DATA_DIR" --instance "$INSTANCE" --out-dir "$PILOT_OUT" --start-fee 5
PYTHONPATH="$SOLVER_SRC" python3 "$CODE_DIR/verify_pilot.py" \
  --data "$DATA_DIR" --instance "$INSTANCE" --out-dir "$PILOT_OUT" --start-fee 5
```

For the root-only sensitivity, replace `blind_link.py` with `root_blind_link.py` and use a fresh private `PILOT_OUT`; the retiming/verifier commands are otherwise identical.

The full-pool/root-only matching and verifier sources are byte-identical to their executed versions. The published retiming script replaces temporary absolute path assignments with CLI inputs; its algorithm, 180-second cap, single order and physics are unchanged. This publication adaptation was syntax checked without another retiming attempt. The exact executed script's SHA-256 and published adaptation's SHA-256 are both retained. Native graph/path/schedule artifacts, logs and raw receipts stay private; only their hashes and non-sensitive aggregates are published.

## Optional supporting augmented-pool MIP

The primary blind candidate was also supplied as an accepted constructive start to an exact-partition, fixed-eight-bus MIP over the original 39,390 columns plus its eight new columns. This is a separate supporting check, not MIP discovery of the eight-bus candidate. With two local threads and a 120-second limit, the solver retains that start and finds no improvement: **361.752 grid nonfleet cost**, finite augmented-pool bound **238.368953**, **34.1071% charging gap**, time-limit status. Final independent replay again passes. All eight selected columns are the same novel retimed candidate routes. No charging-cost optimality or full-model bound follows.

Solver runtime is 120.053 s; complete driver time is 129.196 s. Matching + retiming + driver totals 135.950 measured incremental seconds, excluding invocation gaps/startup and historical graph/CG/dive/MIP compute. Initial replay, pool/model preparation, solve and final replay clocks are separately retained. [Supporting result and reproduction](supporting_mip/README.md) · [Safe aggregate](supporting_mip/aggregate.json) · [Exact executed driver](supporting_mip/solve_arc_pool.py).
