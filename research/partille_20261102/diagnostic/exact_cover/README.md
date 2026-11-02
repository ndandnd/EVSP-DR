# Can the original pool serve every trip exactly once with eight buses?

**No.** This is an exhaustive result for the archived C2 k8 post-dive journal (SHA-256 `5744645fa8fc62e3c79e3880a12b1388dda8e0342fe46bb6c700d61e98c86d0d`). It applies even after ignoring all charger collisions. A dispatch on entirely different generated routes remains possible, as the independently replayed eight-bus witness shows.

1. Compute simultaneous passenger service from the 173 source trips using half-open service intervals. Four distinct peak sets each contain eight trips. No column covers two trips in the same peak set. An eight-bus solution must therefore use eight routes, each covering one trip in **each** peak set. This argument applies to both covering and partitioning. It discards 12,108 columns which cannot participate at this fleet size.
2. Retain all 27,282 eligible columns. For a relaxation concerned only with passenger-trip partitioning, collapse alternative charging plans that cover the same trips. There are 14,026 distinct trip sets. Shared plug feasibility can only further reduce their choices.
3. Search for pairwise-disjoint sets covering all 173 trips. Each selected set hits every peak exactly once, so any complete partition using eligible sets automatically uses exactly eight routes. The Python exhaustive search exhausted 132,499 search states. A separate full-JSON parser, minute-by-minute peak sweep and C++ sparse exact-cover solver exhausted 132,575 states with a different tie order. Neither found a partition. The 60-second Gurobi outcomes, by contrast, were inconclusive.

This result explains why additional **fixed-pool** MIP time cannot yield an exactly-once eight-bus schedule from these columns. A covering solve followed by reassignment or deletion may create a different route and is a separate treatment. The unrestricted eight-bus covering feasibility question remains open.

## Reproduce with authorized timetable and journal

The Python check is `../partition_search.py`, its input-hash checks and half-open event sweep are in `../original_pool_at_floor.py`. The independent sparse checker has no solver dependency. Use a private directory for the generated incidence file.

```sh
python3 dlx_input.py "$INSTANCE" "$ARCHIVED_COLUMNS_JSONL" "$PRIVATE/tripsets.txt"
c++ -O2 -std=c++17 verify_partition_dlx.cpp -o "$PRIVATE/check_dlx"
"$PRIVATE/check_dlx" 173 "$PRIVATE/tripsets.txt"
```

Compare the canonical incidence digest and the program's answer with `dlx_preparation.json` and `dlx_result.json`. The retained inputs, full route records and incidence file stay in private storage.

A third checker fully parsed the compressed journal and used just one peak in its search. The larger retained family contained 17,266 distinct trip sets. Exhausting 248,055 states independently confirmed there is no exactly-once eight-route selection. On the unrestricted 19,399 *distinct trip sets*, the linear relaxation with fleet weight eight does admit a fractional exact partition. The largest recalculated trip residual was below 1.1 × 10⁻¹³. This particular projection drops charger rows, so its LP alone gives no plug-feasibility conclusion. [Third search](independent/complement.json) · [Fractional projection](independent/fractional.json).

**The fractional solution survives the shared plugs.** A separate finite-pool LP retained all 27,282 eligible route records (including distinct charging placements), all 3,319 occupied finite station-minute rows, the 173 passenger-trip equalities, the original terminal requirement and fleet weight eight. It found a feasible solution with 152 fractional route weights. Reconstructing every constraint from the source records gave a maximum trip residual below 7 × 10⁻¹⁵, fleet residual below 6 × 10⁻¹⁵ and zero charger overload. Together with exhaustive partition infeasibility, this establishes an integrality gap in **feasibility at fleet eight** on the same archived columns and plug model. It does not establish a charging-cost gap. [LP and residuals](independent/shared_plug_lp.json) · [Checker and reproduction](independent/README.md).
