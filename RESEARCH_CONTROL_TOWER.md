# EVSP-DR research control tower

Snapshot date: **2026-08-30**. This page answers four questions: what is proved,
what is merely running, what the experiments are meant to establish, and what must
happen next. The CSVs in `analysis/research_control_tower_20260830/` are the
machine-readable source of truth for this view.

## Sixty-second state of the research

1. **Small-instance correctness is established for the current core sample.** Under
   the current 240/240 physics and preferred `event_2p5_event5` representation,
   `L_model` and `I_model` are proved on 9/9 core cells. The finite-pool integer
   optimum `I_pool` is proved on 7/9 in committed evidence. The 9-cell r4-r6
   extension is running as independent replication, but it is only at Stage 1.
2. **The medium ladder is now collecting current-model data.** Six event cells each
   at k=8, k=13, and k=20 were launched. These are combined-cost CG probes, not
   integer proofs. k=10 and the planned stratified sample do not exist yet.
3. **The likely bottleneck changes with scale and stage.** Historical 300/300 runs
   strongly implicate pricing/CG convergence at k>=13, while current k=5 evidence
   also shows a long finite-pool MIP tail. Current telemetry is designed to separate
   build, master, pricing, validation, and MIP time before making a publication claim.
4. **Large-instance work is a boundary study.** Current event k=30 x3 and k=40 x1
   Stage-1 probes were launched. Historical k=30/k=40 results were uncertified; the
   present work must not be described as optimality evidence until the full proof
   ladder closes.

Latest operator-provided queue snapshot: **39 launched tasks, 26 active, 13 absent
from `squeue` and therefore unaudited**. “Absent” does not mean completed.

## The evidence ladder

| Stage | Evidence | Licensed claim |
|---|---|---|
| S0 | Immutable input identity and validation | The named experiment is reproducible |
| S1 | Combined-cost exact CG certificate | Certified combined-cost LP for that representation |
| S2 | Fleet-only exact pricing certificate | `L_model`, the full-representation fleet LP bound |
| S3 | Same-model integer witness plus physical replay | A valid upper bound on `I_model` |
| S4 | S2 lower bound equals S3 upper bound | `I_model`, the true integer optimum **of the named mathematical representation** |
| S5 | Timed or exact MIP over a hashed frozen pool | `I_timed` or a finite-pool incumbent |
| S6 | Matching MIP bounds over that pool | `I_pool`, the optimum of that finite pool |
| S7 | Locked DR/fixed-duty policy comparison | A policy-specific result with full provenance |

`I_model` is not the Giro target. The target is a benchmark/feasibility reference.
`I_model` becomes known only when a certified lower bound and a physically valid
same-model integer schedule meet. `I_pool` can be larger than `I_model` because the
finite pool may omit routes; `I_timed` is only the best valid incumbent found before
the time limit.

“Physics 240/240” identifies the input/feasibility convention used by the current
campaign. A statement such as “240/240 checks passed” means route and schedule
artifacts survived the relevant physical replay under that convention; it is not a
count of optimality proofs and cannot be merged with historical 300/300 results.

## Current campaigns

| Campaign | Jobs | Stage | Latest view | What happens after it leaves the queue |
|---|---:|---:|---|---|
| Medium event k8/k13/k20 | 922550/922551/922553 | S1 | 13 active; 5 absent/unaudited | Audit root, classify every row, promote qualified cells |
| Small event r4-r6 | 922874 | S1 | 2 active; 7 absent/unaudited | Audit, then run S2-S4 replication pipeline |
| Event k30 | 922875 | S1 | 3 active | Audit as boundary evidence |
| Event k40 | 922876 | S1 | absent/unaudited | Inspect `sacct`, stderr, telemetry, and artifact before retry |
| Panel A frozen-pool 48h | 922864 | S5 | 5 active on scaglione | Run a new 172800-second audit; update only validated pool claims |
| Panel B frozen-pool 48h | 922865 | S5 | 3 active on scaglione | Same audit and validation rule |

The 48-hour MIP campaigns use native HiGHS because it is available and reproducible
on the compute nodes. Restoring Unicorn's Gurobi license is useful for backend
replication and future speed comparisons, but it does not invalidate HiGHS results
or justify discarding them.

## Goal-by-goal publication plan

### Goal 1 — demonstrate correctness on small instances

The core result is already strong: current preferred event physics closes the
full-model sandwich on 9/9 cells. The remaining work is replication and a clean
cell-level publication table, not a search for a different headline. The r4-r6
extension counts only after each cell moves from S1 to S2, obtains an independently
validated S3 witness, and closes S4. Finite-pool proofs are valuable diagnostics but
are not required to call `I_model` proved.

### Goal 2 — build a credible small-to-large computational ladder

The final table should be stratified rather than a single lucky seed per fleet size.
The intended sizes are k=1/2/3/5/8/10/13/20/30/40, with replication concentrated
around the transition region. Report trips, time horizon, route-overlap/separability
features, certified LP time, censoring reason, iterations, columns, peak RSS, target
status, finite-pool incumbent/bound, and proof flags. The present k=8/13/20 campaign
is a feasibility and instrumentation gate. After it is audited, generate k=10 and a
predeclared 72-cell design balancing bus count, replicate, trip count, and structural
difficulty. Do not select only cells that finished.

### Goal 3 — locate the bottleneck

For each cell, assign the *first limiting stage*: input/build, restricted-master LP,
exact pricing, convergence/iteration cap, physical validation, or final MIP. Use
censor-aware elapsed times for wall-limited or preempted work. Historical results
suggest exact pricing/CG convergence dominates at larger k, while the current
finite-pool experiments show that the final MIP can independently dominate even at
k=5. The current campaigns are meant to quantify, not assume, that transition.

### Goal 4 — characterize k=30 and full k=40

These are boundary case studies. A useful paper result may be a certified bound,
a timed incumbent with a reproducible gap, or a rigorous explanation of where the
pipeline stops. The result must carry its stage label. A route weight near the target
or a plausible incumbent is not proof.

## Exact next sequence

1. Let currently launched cells run; do not submit duplicates.
2. When the medium and extension arrays are absent from `squeue`, run:

   ```bash
   cd "$HOME/ladder-lite/repo-manager" && git pull --ff-only
   bash scripts/event_uniform_envelope/audit_medium_event_legacy.sh \
     "$HOME/ladder-lite/medium_event_legacy_20260830_44b6d5"
   bash scripts/event_uniform_envelope/audit_medium_event_legacy.sh \
     "$HOME/ladder-lite/event_extension_20260830_44b6d5"
   ```

3. Before the 48-hour pool jobs finish, implement the missing
   `audit_highs_unresolved_retry172800.sh`; after completion, audit Panels A and B.
4. Convert audit rows into a bottleneck-classification CSV. Promote only qualified
   small cells through S2-S4; keep large cells labeled as boundary evidence.
5. Use the audited k=8/13/20 behavior to lock the k=10 and 72-cell stratified design.
6. Merge the curator's stable ID registry and corrections before freezing any paper
   table.

## Update rule

The control tower is deliberately conservative. Live queue information may change,
but committed proof counts change only after artifact validation, hash capture, and a
records commit. Every future update must keep physics, representation, stage, and
provenance visible. Run:

```bash
python scripts/research/validate_research_control_tower.py
```

Sources frozen into this view include the committed current campaign artifacts, the
historical `analysis/scale_ladder/ll_20260820c/ladder_summary.csv`, the 8h/24h retry
audits, and the operator-provided Slurm snapshot on 2026-08-30. The separate Goal-2
research registry exists at commit `d7a7cbd` on
`codex/goal2-research-registry-20260830`; canonical record corrections still require
manager review before merge.
