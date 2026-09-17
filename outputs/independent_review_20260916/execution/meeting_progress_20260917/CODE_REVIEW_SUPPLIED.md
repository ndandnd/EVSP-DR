# EVSP–DR code review — 17 Sep 2026

Reviewer: Claude (Fable). Scope: the execution code on `codex/zero-fee-terminal-cg` (contains pinned CG commit `a0e0bb76` and MIP commit `871d057e`; 125 src files / 69.7k LOC, 64 test files / 28.3k LOC), the `main` checkout, branch and repository hygiene, and one measured profile (`w5_k31`, 716 trips, 222 CG iterations, 3.3 h). Every number below is from a file or log; recommendations are ranked by payoff per unit of effort.

## 1. Where the code actually is

| Location | What it is | State |
|---|---|---|
| `main` (local, d3aba878, 13 Sep) | **Legacy** codebase: 42 files, 39k LOC, six near-identical `run_experiments_b_dp_charge_duck*.py` (+ ` copy.py` variants), `run_10B_group0*.py` ×6, empty `tests/`. 577 untracked files incl. `.codex-work/`. | Nothing here produced current results. |
| `codex/zero-fee-terminal-cg` and siblings (`action3-full`, `review-strict-chain`, `capacity-parallel-followup`) | The real code: `event_pricer_network.py`, `exact_pricer_expanded.py`, `master_lp_gurobi.py`, `run_exact_pool_mip.py`, physics, audits, launchers, 60+ test files. | Diverged siblings, 130–132 src files each. |
| `codex/research-register-*`, `origin/codex/parallel-research-20260911` | Publication branches: `outputs/` only, 8 src files. | Fine as-is. |
| 126 branches; 12 worktrees under `/private/tmp` (7 `prunable`) | Campaign-per-branch pattern from the agents. | Needs consolidation. |

**Recommendation 1 (do first, one hour):** create `develop` from `codex/zero-fee-terminal-cg`, merge the three sibling branches' src deltas into it (they are small: 125→132 files), make it the default branch, and move `main`'s legacy `src/` to `legacy/` in one commit so no one runs it by accident. Then `git worktree prune`, delete branches whose tips are merged.

## 2. Where the time goes (measured, `w5_k31`, `cg.json.phase-telemetry.jsonl`)

| Phase | Seconds | Share | Calls | Note |
|---|---:|---:|---:|---|
| `master_attempt` (Gurobi LP) | 5,549 | 32.5% | 223 | persistent, warm-started; many solves 0–2 pivots / 0.6 s, but degenerate ones hit 2k–98k pivots / 30–86 s |
| `pricing_extra_columns` | 3,638 | 21.3% | 222 | sink-predecessor enrichment for 29 extra columns costs **as much as the exact shortest path** |
| `pricing_shortest_path` | 3,497 | 20.5% | 222 | 15.7 s per pass over 1.18 B packed arcs (19 GB) — memory-bandwidth bound |
| `inherited_event_pool_import` + `inherited_replay` | 3,554 | 30.8% | 1 | the warm-start tax: one hour to re-validate 247k inherited columns |
| `incidence_construction` | 736 | 4.3% | 223 | rebuilt every iteration although the Gurobi master doesn't use it (`skip_gurobi_incidence` exists but is off by default) |
| `network_build` | 85 | 0.5% | 1 | cache hit; **original build 39,827 s = 11.1 h**, 18.8 GB pickle |

Graph: 62,812 DAG nodes, **1,177,049,732 arcs** — mean out-degree 18,700. Each arc is a composite (leave trip → deadhead to station → charge to a target SOC level → deadhead to successor), so arcs ≈ trips × SOC levels × stations × successors × target levels.

Convergence (`cg.json.iters.csv`): route weight 30.000 by iteration 10; min rc −44 → −0.34 by iteration 60 → −1e−6 at 222. The last 160 iterations (~70% of CG time) move the objective by 0.16 on 3,001,338 (5×10⁻⁸ relative).

## 3. Algorithmic recommendations, ranked

### A. Fleet-certified early stop — 10× on the ladder, zero risk, ~50 lines
You already compute the Lagrangian bound post hoc (`execution/f3/`). Compute it inside the loop: `LB = z_RMP + K·rc_min` with `K = U/100000`. Stop when `ceil((LB − E_max)/100000) == ceil(z_RMP/100000)` — the fleet bound is certified. On `w5_k31` that happens around iteration 10–20 instead of 222. Keep the full run only when charging cost matters (Q3 runs), and even there stop on a relative electricity gap (e.g. 1e-4) rather than rc = 1e-4 absolute.

### B. Column management — LP 32% → ~3%, ~200 lines
The RMP carries the entire 250k-column pool. Standard practice: keep the LP to the columns that were basic or had rc < θ in the last N iterations (typically 5–20k), park the rest in the pool, and before calling the graph pricer do a **vectorized pool scan** (`rc = cost − A·π` over a CSR incidence matrix, milliseconds) — if the pool has negative columns, add them and skip pricing. Also flip `skip_gurobi_incidence` on by default (saves 4.3% outright) and set `Params.Sifting`/consider `Method=1` with `Presolve=0` once the RMP is small. The degenerate 2k–98k-pivot solves disappear when the LP is small.

### C. Extra-column enrichment — 21% → <2%, ~80 lines
`sink_predecessor_route_batch` sorts every finite sink arc with a `json.dumps` tie key, walks parents per candidate, and computes Jaccard novelty with fresh `frozenset`s. Use `heapq.nsmallest(limit·4, …)` on `(value, source)` only, walk only those, represent trip sets as bit-vectors (`int` masks or `np.packbits`) for novelty, and defer the JSON tie key exactly as `_add` already does.

### D. Inherited import — 30% → ~3%, moderate
One hour per stage to re-replay 247k columns whose graph identity hasn't changed. Cache per-column validation keyed by `(event_lattice_sha256, route_nodes_sha256)` and skip replay on match; store columns as node-index paths (uint32 arrays) rather than 4 KB JSON dicts (`cg.json.columns.jsonl` is **985 MB** for one case); use the multiprocessing pool that already exists in `inherited_event_pool_records` with all allocated CPUs. Or: given B, don't import the pool into the LP at all — import it into the *pool* and let pool-pricing pull what's needed.

With A–D, `w5_k31` should drop from 3.3 h to roughly 20–30 minutes without touching the graph.

### E. Factor the composite arc into a station time–SOC lattice — 1.18 B → ~10⁷ arcs; the structural fix
Replace `trip(soc) → [station, charge Δ, successor]` composite arcs with: `trip(soc) → station(t_arr, soc)` deadhead arcs; `station(t, s) → station(t+1, s+Δ_t)` charging arcs priced by the tariff at t (plus a charging/idle state bit to carry the start fee and, later, a minimum-duration counter); `station(t, s) → successor(soc′)` deadhead arcs. Sizes: stations have 484–852 event times × 96 SOC levels ≈ 80k lattice nodes each; arcs are O(nodes) per station plus O(trips × levels × stations) deadheads — tens of millions total, not a billion. Consequences:
- Graph build minutes instead of 11 h; pricing seconds; `network.pkl` MB instead of 18.8 GB; the six action3 arms would not need 40-hour graph allocations.
- **Capacity duals become additive lattice-arc costs.** The 7-hour capacity pricing (`execution/…/capacity_deadline5_completed`) exists only because the composite arc must re-optimize the charging window under duals. On the lattice there is nothing to re-optimize. This is the design change the Opus brief should be built on, and the price-maker master (convex Φ on lattice-node load) falls out of it.
- Exactness is preserved: the current `_best_charge_window` picks the cheapest window for a fixed energy; the lattice enumerates all windows as paths. Rounding stays conservative on the same 2.5 kWh grid.
Effort: weeks, one careful engineer; validate against the existing `arcflow_oracle` and the brute-force tests.

### F. Compiled relaxation kernel (stopgap if E is deferred), ~1 day
`_min_reduced_cost_route_lazy` is a Python loop over 62k nodes with numpy gathers on 19 GB of arrays. A `numba.njit` loop over the CSR arrays (`_arc_slices`, `_arc_targets_np`, `_arc_costs_np`) runs at memory bandwidth (~5–10 GB/s) — 2–4 s instead of 16. Same for `_walk`. No semantic change.

### G. Dual stabilization — fewer iterations, ~150 lines
Wentges smoothing (`π_used = α·π_center + (1−α)·π_LP`, α≈0.7–0.9 with mis-pricing fallback) typically halves the tail. Combined with A it matters less for the fleet ladder, more for the tariff (Q3) runs where you need the electricity term converged.

### H. The integer step — where the science says the bottleneck is
`run_exact_pool_mip.py` builds a set-covering IP with a binary per pool column (254k binaries at `w5_k31`) and gives the fleet stage 30 minutes. Two standard moves: (i) **reduced-cost fixing** — with certified duals, any column with `rc > 100000·(UB − LB)` cannot be in an optimal solution; on LP = k instances that leaves a few thousand columns; (ii) **diving with pricing** (price-and-branch): fix the largest-λ route, re-solve LP + price, repeat — it generates the integer-complementary columns the pool lacks, which is exactly the failure mode F5/F2 identified. Item 8's 12-hour search is the expensive way to ask the question diving answers in minutes.

## 4. Software engineering

- `exact_pricer_expanded.py:run_cg` is a **1,660-line function** (lines 1607–3267) with ~15 nested closures sharing `nonlocal` state; `exact_pricer_expanded.py` also still contains the legacy `ExpandedNetwork`. Split into a `CGRun` class with `iterate()`, `price()`, `solve_master()`, `checkpoint()`, `finalize()`; move telemetry to a small context-manager. This is the file every future change touches.
- 26 files over 800 LOC; 19 `run_*`, 12 `audit_*`, plus `launch_*`, `summarize_*`, `reconcile_*`, `build_*_evidence`, `monitor_*`, `migrate_*` all in `src/`. Proposed layout: `evsp/` (network, pricing, master, cg, mip, physics, io, certificates), `tools/` (launch, collect, audit, monitor), `tests/`. `pricing_dp_og.py` (2,922 LOC) is legacy — archive it.
- No `pyproject.toml`, no CI, no lint/type config; two `requirements-*.txt`. 28k LOC of tests that nothing runs automatically is the single biggest risk to the "every claim has a test" culture the project has built. Add `pyproject.toml` + `ruff` + a GitHub Action running `pytest -m "not gurobi"` on push (HiGHS backend exists for exactly this).
- Smells worth a pass: `json.dumps` in sort keys and dict keys; `deepcopy` of action dicts; `_window_cache` keyed on `round(x, 9)` floats; JSON-per-column journals (`continuous_realization` embedded in every record).

## 5. Repository and data hygiene

- `.git` is **3.8 GB**; packs 1.95 GB; **72 `tmp_pack_*` files (106 MB)** from interrupted pushes/gc — safe to delete after `git fsck`.
- History contains `src/Best_10bus_K555_columns.json` (562 MB), `Best_30bus` (259 MB), `Best_43bus` (185 MB), `Best_15bus` (171 MB), `Best_20bus` (103 MB), and `outputs/research_register/register.json` committed at 84–95 MB **seven-plus times**. `outputs/` is 7.6 GB in the working tree with 50 MB monitor snapshots.
- Do not rewrite history while Astra's worktrees are live. Instead: stop committing `register.json`/monitor JSONs (regenerate from CSV or store in `/share/scaglione` and commit hashes), move `outputs/*/status_*/` snapshots to LFS or a data repo, and schedule a `git filter-repo` for the `Best_*` blobs at a quiet moment with all agents stopped.

## 6. Scientific-computing hygiene

- Certificates are floating-point (Gurobi duals + Python `float` reduced costs with 1e-4 tolerance; relaxation tests at 1e-12). For the 67 "bound met" instances, add an **exact rational re-verification** of the final RMP: take the positive-support columns and the final duals, verify dual feasibility for every pool column in `fractions.Fraction`, and re-solve the tiny support LP exactly (SoPlex exact or QSopt_ex, or rational simplex on ≤700 columns). Astra's time-only bound already does this style for the fleet; do it for the weighted LP too and the paper can say "proven" without a footnote.
- State the discretization contract once, formally: SOC grid 2.5 kWh with outward rounding → the event model is a restriction of the continuous model; every event-feasible schedule is continuous-feasible; LP bounds are for the event model. F4's continuous witnesses vs event representability is the one place this bit.
- Determinism is good (`PYTHONHASHSEED=0`, explicit Gurobi `Seed`, ordered journals, hashes everywhere). Keep it; it is the project's best asset.
- `master_lp_gurobi.py` uses `Threads` from args and `Method=1`; fine. `BarHomogeneous=1` fallback is never triggered in the log — fine.

## 7. Suggested order

1. Recommendation 1 (branch consolidation) + `skip_gurobi_incidence` default + `git worktree prune` — today.
2. A (early stop) and C (enrichment) — a day; re-run one chain to confirm identical LP endpoints.
3. B (column management) and D (import cache) — a week.
4. H (reduced-cost fixing + diving) — a week; this is the scientific payoff for Q2.
5. E (lattice) — the version of the pricer that paper 2 needs; start as a parallel implementation validated against the current one on k ≤ 5.
6. Packaging/CI and the run_cg refactor alongside 3–5; exact re-verification (§6) before submission.
