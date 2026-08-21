#!/usr/bin/env python3
"""Backfill qualified pool-MIP and model-optimality rows from committed artifacts.

Run id: model_optimality_20260821. Every value below is transcribed from a
committed artifact on this branch (path + sha256 recorded per row); nothing is
taken from cluster-only data. The ~100-cell cluster pool-MIP table (cells like
k08_s3_g2_b10) was never committed to any git ref and is NOT recorded here —
see records/runs/model_optimality_20260821/README.md.

Sandwich rule (operator, 2026-08-21): the certified fleet LP L is a valid
model-wide lower bound and fleets are integral, so I_model >= ceil(L); where a
physically validated incumbent equals ceil(L), the discrete-model optimum is
proven (model_optimality_method = sandwich). Per the operator's rule, cells
whose certified LP sits exactly at target and whose pool MIP returned that
target qualify immediately; rows promoted on that authority with no committed
witness-replay record carry physical_witness_valid = "" and say so in notes.
"""
import csv
import hashlib
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
TARGET = REPO / "records/RESULTS_LOG.csv"
RUN = "model_optimality_20260821"
DATE = "2026-08-21"
LABEL = "curator_backfill_from_committed_artifacts"
MEANING = "combined-cost-master route weight"

ARC = "analysis/arcflow_oracle_20260820/results.csv"
ARC_REPORT = "analysis/arcflow_oracle_20260820/REPORT.md"
BP_PAUSE = "analysis/branch_and_price_experiment_20260820/CORRECTED_K2_PAUSE.md"
BP_FINE = "analysis/branch_and_price_experiment_20260820/PRICING_OPTIMIZATION_20260820.md"
BP_PRE = "analysis/branch_and_price_experiment_20260820/PRE_PHASE1_EVIDENCE.md"
DLOG = "records/DECISION_LOG.csv"   # D0017, D0023
BLOG = "records/BUG_LOG.csv"        # B0021
MATRIX = "analysis/scale_ladder/ll_20260820c/resolution_matrix.csv"
LADDER = "analysis/scale_ladder/ll_20260820c/ladder_summary.csv"

PHYS = "historical 300/300 physics"

# source-CG facts, transcribed from the committed aggregates
# primary 15/10 (ladder_summary): cell -> (LP, iterations)
PRIMARY_CG = {
    "k02_s1": ("2.181818", 212), "k02_s2": ("2.187500", 93),
    "k02_s3": ("2.274725", 546), "k03_s1": ("3.181818", 330),
    "k03_s2": ("3.404698", 693), "k03_s3": ("3.000000", 264),
    "k05_s1": ("5.323651", 691), "k05_s2": ("5.000000", 798),
}
# fine grids (resolution_matrix): (cell, soc, blk) -> (LP, iterations)
FINE_CG = {
    ("k02_s1", "1.0", "5"): ("2.0000", 250),
    ("k02_s2", "1.0", "5"): ("2.0000", 88),
    ("k02_s3", "1.0", "5"): ("2.0000", 274),
    ("k03_s1", "1.0", "10"): ("3.0000", 438),
    ("k03_s3", "1.0", "10"): ("3.0000", 290),
    ("k05_s2", "5.0", "10"): ("5.0000", 681),
    ("k05_s2", "2.5", "10"): ("5.0000", 512),
    ("k05_s2", "1.0", "10"): ("5.0000", 389),
}
TARGETS = {"k02": 2, "k03": 3, "k05": 5}


def sha(rel):
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def main() -> int:
    with TARGET.open(newline="") as h:
        reader = csv.DictReader(h)
        fields = reader.fieldnames
        if "model_fleet_proven" not in fields:
            print("run migrate_results_log_qualified_optimality.py first")
            return 1
        if any(r["run_id"] == RUN for r in reader):
            print(f"rows for {RUN} already present; refusing to append twice")
            return 1

    shas = {p: sha(p) for p in (ARC, ARC_REPORT, BP_PAUSE, BP_FINE, BP_PRE, DLOG, BLOG)}
    rows = []

    def add(cell, soc, blk, **kw):
        scale = int(cell[1:3])
        row = {f: "" for f in fields}
        row.update({
            "date_utc": DATE, "run_id": RUN, "execution_mode": "local",
            "group": "MODEL_OPTIMALITY", "status": "completed",
            "cell_id": f"{cell}_soc{soc.rstrip('0').rstrip('.')}_blk{blk}_{kw.get('phase','').lower()}",
            "scale": scale, "sel_rep": cell[-1], "soc_step": soc, "block_min": blk,
            "target_fleet": TARGETS[cell[:3]], "label": LABEL,
        })
        row.update(kw)
        rows.append(row)

    def src_cg(cell, soc="15.0", blk="10"):
        if (soc, blk) == ("15.0", "10"):
            lp, iters = PRIMARY_CG[cell]
        else:
            lp, iters = FINE_CG[(cell, soc, blk)]
        return {"route_weight": lp, "route_weight_meaning": MEANING,
                "source_cg_certified": "True", "source_cg_stop_reason": "certified",
                "source_cg_iterations": iters}

    # --- A. arc-flow oracle rows: direct discrete-model solves, no pool ---
    arc_proven = {"k02_s1": (3, "600.872"), "k02_s2": (3, "113.485"),
                  "k02_s3": (3, "602.081"), "k03_s1": (4, "601.464"),
                  "k03_s2": (4, "603.405"), "k03_s3": (3, "167.766"),
                  "k05_s2": (5, "601.344")}
    for cell, (fleet, wall) in arc_proven.items():
        lp = PRIMARY_CG[cell][0]
        extra = "; independently agrees with corrected branch-and-price" if cell == "k02_s2" else ""
        extra2 = ("; sandwich-consistent with the replayed 3-bus PRE_PHASE1 witness"
                  if cell == "k03_s3" else "")
        add(cell, "15.0", "10", phase="ARCFLOW",
            model_fleet_proven=fleet, model_optimality_method="arcflow",
            optimality_scope="discrete_model", physical_witness_valid="True",
            wall_s=wall, artifact_path=ARC, artifact_sha256=shas[ARC],
            notes=f"{PHYS}; certified LP {lp}, ceil={fleet}, fully integral "
                  f"decomposition-verified witness, G4 physical replay (REPORT.md)"
                  f"; no pool — direct model solve; charging cost not optimized"
                  f"{extra}{extra2}")
    for cell, bound_note, wall in (
        ("k05_s1", "no incumbent within budget; true fleet 6-11", "601.342"),
        ("k05_s3", "no incumbent within budget; true fleet at least 5", "601.457"),
    ):
        add(cell, "15.0", "10", phase="ARCFLOW", status="censored",
            censor_reason="integer_limit_no_incumbent", wall_s=wall,
            artifact_path=ARC, artifact_sha256=shas[ARC],
            notes=f"{PHYS}; discrete-model integer optimum UNRESOLVED: {bound_note}")

    # --- B. corrected branch-and-price rows ---
    add("k02_s2", "15.0", "10", phase="BRANCH_AND_PRICE",
        model_fleet_proven=3, model_optimality_method="branch_and_price",
        optimality_scope="discrete_model", physical_witness_valid="True",
        commit="af45ed80d6e93f768224c118296dade7fb4b31cb", wall_s="6.855",
        artifact_path=BP_PAUSE, artifact_sha256=shas[BP_PAUSE],
        notes=f"{PHYS}; global LB 3.000000000, physically replayed 3-bus "
              "incumbent (cost 300148.744); tree closed by integer fleet bound; "
              "agrees with arc-flow (different witness, cost 300070.592)")
    add("k02_s2", "1.0", "5", phase="BRANCH_AND_PRICE",
        model_fleet_proven=2, model_optimality_method="branch_and_price",
        optimality_scope="discrete_model", physical_witness_valid="True",
        commit="8a33b187a93a40572d02782f8ddac4ee56821dc8", wall_s="6.816",
        artifact_path=BP_FINE, artifact_sha256=shas[BP_FINE],
        notes=f"{PHYS}; root fleet LP 2.000000000, conservative LB 1.999999971, "
              "ceil=2, root closed without branching; 2-bus partition passed "
              "full physical replay; end-to-end RAW recovery, no injected routes")

    # --- C. standalone sandwich: k02_s1 at 1/5 ---
    add("k02_s1", "1.0", "5", phase="SANDWICH", **src_cg("k02_s1", "1.0", "5"),
        model_fleet_proven=2, model_optimality_method="sandwich",
        optimality_scope="discrete_model", physical_witness_valid="True",
        artifact_path=BP_PRE, artifact_sha256=shas[BP_PRE],
        notes=f"{PHYS}; bound: certified CG LP 2.0000 at 1kWh/5min "
              "(resolution_matrix.csv), ceil=2; witness: PRE_PHASE1 replayed "
              "valid feasible 2-bus partition (validate_final_selected_routes "
              "passed; that driver's BOUNDS are invalidated, its replayed "
              "incumbents remain valid upper bounds per the artifact itself)")

    # --- D. primary-grid RAW pool-MIP rows (incumbents from arc-flow table) ---
    pool_primary = {"k02_s1": 4, "k02_s2": 4, "k02_s3": 7, "k03_s1": 5,
                    "k03_s2": 10, "k03_s3": 4, "k05_s1": 11, "k05_s2": 6}
    for cell, inc in pool_primary.items():
        proven = cell == "k02_s2"  # D0023: "proven optimal for that pool"
        model_note = (f"model optimum {arc_proven[cell][0]} proven by arc-flow "
                      f"(pool excess {inc - arc_proven[cell][0]})"
                      if cell in arc_proven else "model unresolved 6-11")
        add(cell, "15.0", "10", phase="MIP", group="MIP_RAW", arm="RAW",
            **src_cg(cell), mip_incumbent_fleet=inc,
            pool_fleet_proven="True" if proven else "",
            pool_mip_bound="4" if proven else "",
            optimality_scope="finite_pool" if proven else "",
            artifact_path=ARC, artifact_sha256=shas[ARC],
            notes=f"{PHYS}; pool incumbent as recorded in the arc-flow oracle "
                  f"table (pool_MIP_given); "
                  + ("proven optimal for its fixed pool per D0023; " if proven
                     else "proven-for-pool status not committed; ")
                  + f"NO discrete-model optimality claim from this row; {model_note}")

    # --- E. fine-grid RAW pool-MIP rows ---
    fine = [
        # cell, soc, blk, incumbent, witness_valid, witness_note
        ("k02_s2", "1.0", "5", 2, "",
         "incumbent replay not recorded in committed evidence; promoted per "
         "operator sandwich rule 2026-08-21; independently proven 2 by "
         "branch-and-price (fully replayed)"),
        ("k02_s3", "1.0", "5", 2, "True",
         "witness: PRE_PHASE1 replayed valid feasible 2-bus partition at this "
         "grid (bounds from that driver invalidated, incumbents remain valid)"),
        ("k03_s1", "1.0", "10", 3, "",
         "incumbent replay not recorded in committed evidence; promoted per "
         "operator sandwich rule 2026-08-21"),
        ("k03_s3", "1.0", "10", 3, "",
         "incumbent replay not recorded in committed evidence; promoted per "
         "operator sandwich rule 2026-08-21"),
    ]
    for cell, soc, blk, inc, wv, wnote in fine:
        lp = FINE_CG[(cell, soc, blk)][0]
        add(cell, soc, blk, phase="MIP", group="MIP_RAW", arm="RAW",
            **src_cg(cell, soc, blk), mip_incumbent_fleet=inc,
            pool_fleet_proven="True", pool_mip_bound=f"{inc}.0",
            model_fleet_proven=inc, model_optimality_method="sandwich",
            optimality_scope="discrete_model", physical_witness_valid=wv,
            artifact_path=DLOG, artifact_sha256=shas[DLOG],
            notes=f"{PHYS}; D0017: returned exactly {inc}, fleet_proven true, "
                  f"gap 0, OPTIMAL (pool scope); sandwich: certified CG LP {lp}, "
                  f"ceil={inc}=incumbent, so discrete-model optimum {inc}; {wnote}")

    for soc, inc in (("5.0", 8), ("2.5", 10), ("1.0", 13)):
        add("k05_s2", soc, "10", phase="MIP", group="MIP_RAW", arm="RAW",
            **src_cg("k05_s2", soc, "10"), mip_incumbent_fleet=inc,
            pool_fleet_proven="True", pool_mip_bound=f"{inc}.0",
            optimality_scope="finite_pool",
            artifact_path=BLOG, artifact_sha256=shas[BLOG],
            notes=f"{PHYS}; B0021: proven optimal for its pool; certified CG LP "
                  f"5.0000 gives ceil=5 != incumbent {inc}, sandwich does NOT "
                  "close — pool-composition/search failure, no discrete-model "
                  "claim; never present integer results as monotone in resolution")

    with TARGET.open("a", newline="") as h:
        csv.DictWriter(h, fieldnames=fields, lineterminator="\n").writerows(rows)
    print(f"appended={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
