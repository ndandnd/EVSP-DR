# Research results — 15 September, 06:19 EDT

**All 25 original k16–25 misses have now been recovered by separate MIP searches on unchanged pools.** The final case, chain5 at target25, uses25 buses and proves25 minimal in its207717-column pool. Fleet search takes17.86 minutes; the total3.5-hour run still leaves charging optimality open. This establishes that the original pool contained a target fleet. It does not establish that additional time alone caused the improvement: this was a new search tree, and the observed fleet runtime is shorter than the original30-minute allowance. [Unchanged-pool checks and results](longer_gap_results.csv).

**New original one-hour results:** chain4 at target27 uses27/proved in pool; chain5 at target27 uses29 with bound26, still open. Their CGs both hit four hours. Chain5's fractional route weight is26 despite target27; weighted RMP objective2,601,198.9161064. Without a pricing certificate that objective is not a full-model lower bound.

Largest individual target matched by the original one-hour MIP, chains1–6: **26,27,27,27,26,28**. This does not assert that all smaller cases matched in that budget. [All90 submitted k16–30 cases](CHAIN_TABLES.md) · [CSV including73 CG and73 MIP endpoints, timing, stopping reasons and hashes](all_chain_extension_results.csv).

## Complete larger compact-start comparison

Core retains prior integer-solution sequences and prior positive-LP-weight sequences. Expanded fills that starting set to512 distinct trip sequences. Neither number describes the final pool. Each cell below is actual integer buses found.

| Chain | Target20 core | Target20 expanded | Target25 core | Target25 expanded |
|---|---:|---:|---:|---:|
| 1 | 21 | 21 | 25 | 25 |
| 2 | 22 | 21 | 31 | 26 |
| 3 | 24 | 23 | 26 | 25 |
| 4 | 20 | 20 | 26 | 26 |
| 5 | 21 | 21 | 26 | 26 |
| 6 | 20 | 20 | 26 | 25 |

**Eight target matches, eight proved target exclusions, eight open misses.** The target is excluded in C1k20 both pools, C2k20 core, C5k20 core, C4k25both and C5k25both. Seven of those prove an exact above-target minimum; C2k20 core instead found22 with bound21, so20 is impossible but21versus22 remains open. Every other above-target cell is an unfinished integer gap, not a proved missing-column limitation.

Three CGs converged (C2k20expanded and C3k20both); all three still have open integer gaps. The other21 CGs hit their four-hour limits. MIPs allowed10800 seconds for fleet within12600 seconds total. Seven of eight target matches also prove charging optimality; C3k25expanded proves fleet25 but charging remains time-limited. [All24cases with exact bounds, CG values, times and source hashes](compact_large_results.csv).

The new `target_excluded_in_saved_pool` field checks bound > target +1e-5. A bound of20.000000000000078 does not exclude20;21does. Exact fleet optimality is not required to rule out a lower target. The classifier was checked against the prior snapshot and all83 current larger/pool diagnostic source bindings passed.

## Model and proof limits

These are baseline covering runs:240 kWh battery,240 kW charging,100000 bus cost plus electricity and5 per charging start. No reserve,shared capacity or ending-SOC floor. Individual-route replay passes for the new MIPs; removing duplicate trip assignments is not separately validated. A fleet proof concerns only its saved columns. CG certification, integer fleet proof, charging optimality and physical validation remain separate.

The completed smaller-start C1k15 audit and stricter-physics results are unchanged: [missing-column evidence](../status_20260915T080947Z/README.md) and [reserve screen](reserve_results.csv).

## Queue and follow-up

The06:11–06:19collection contains18 running and29 true-input-dependent jobs, excluding33 held historical tasks. There is no new invalid dependency or confirmed preemption. All12k29–30graphs are running; real previous-k and own-CG dependencies remain intact. C3k28longer search222757 is still active. Held historical and EVSPV2Gwork are untouched.

A controlled follow-up is being prepared separately: union the two existing compact pools on eight selected cases; four unchanged core512controls cover the cases where target remains possible. No new CG or GIRO columns. Native greedy initialization and frozen MIPsource/settings are retained. This tests complementarity of columns, not performance at equal total CG cost. Production status will be recorded after the native fixture passes.

The direct-login command was refused when it omitted the standing SSHsocket; the existing control connection remained usable. No loss of monitor access occurred.

Snapshot20260915T101137Z completed10:19:10.919UTC in453.0 seconds; SHA256 `a794a304bbe04a02cbaf8dd53c1941628260e843de661170771cbdc88ab76da8`. Register/workbook3262 records,73 sourcegroups,6 preserved supplements. Checks:321 core,179 evening,83 pooldiagnostic,146 originalchain endpoints. [17 new endpoints](new_endpoints.csv) · [Document verification](doc_verification.json). The current dashboard replaces tables in place and preserves both figure tabs. Morning consolidation remains scheduled around09:00EDT.
