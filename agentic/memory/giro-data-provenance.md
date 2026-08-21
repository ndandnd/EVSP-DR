---
name: giro-data-provenance
description: "Where EVSP-DR's simplifying assumptions come from — GIRO/Transdev raw data + email thread (Sep 2025–Feb 2026)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5411361d-3fc9-42c9-8275-f6af541497cb
  modified: 2026-07-31T23:45:08.844Z
---

Raw data in `EVSP-DR/data/original_giro_data/`: Partille (`Par_*`) + Frölunda (`FDL_*`) contracts (greater Gothenburg, Transdev, scheduled in GIRO/Hastus): DHD.xlsm (deadhead matrix per place, Base + up to 3 time-interval duration variants), VehicleDetails.xlsx (GIRO-solved schedule, activities Prep/Pull/Regular/Deadhead/Recharge with SOC-%, kWh), Notes.docx (scheduling rules), 2 schedule PDFs. Full email thread + provenance mapping written to `data/original_giro_data/EMAIL_EXCHANGE.md` (2026-07-31).

Key sanctioned decisions (from Karl-Magnus Andersson, Transdev):
- Trips = rows with Identifier=="Regular" (Sep 30).
- GIRO optimized with NO electricity cost; our TOD-cost objective is the novelty (Sep 30, Anne Mercier confirms; GIRO offered to re-optimize with costs for comparison — open offer!).
- Ref-number substitution for missing DHD locations explicitly approved (Dec 17: "Yes that's no problem").
- DHD matrix: 20+ years hand-curated, shared across 4 contracts, no systematic method; missing links mostly cross-contract; durations from GIS + test runs (Oct 29).
- DHD has time-interval-dependent durations (3rd interval 21:00–05:59); Hastus trip times can exceed 24:00 ("24:06" = 00:06 next day) (Feb 6).
- Real tariffs include monthly 15-min peak-kW demand charges (no numbers obtainable) (Oct 24) — unmodeled; the natural "demand response" extension.
- Karl's rough hourly SEK/kWh table (peak 17-19 = 1.5, overnight 0.2-0.4) saved as `data/hourly_prices_transdev_sek.csv` (2026-07-31); experiments so far used synthetic curves only.

What EVSP-DR drops vs GIRO reality: interval-dependent deadheads (keeps shortest), distance column, all Notes rules (interlining bans, route-specific layover minima ≥4min/10%, block ≥2h preference, FDL min-SOC 20% for >20h duties), prep activities, crew constraints; homogeneous 300kWh/300kW fleet vs real ≈239kWh/≈220kW (derived from Recharge rows: 40.35 kWh in 11 min, 66.4%→83.3%); FDL contract entirely unused; missing DHD links treated as forbidden arcs.

2026-07-31 review-driven fixes (uncommitted, from independent code review, verified then applied): LP master columns now unbounded above (ub=1 caused bound-dual degeneracy → pricer rediscovers columns → false "no_new_columns" stops; MIP keeps ub=1, warm starts clamped); run_final_mip.py rebuilds T from the checkpoint's csv_name instead of only route-covered trips (silent trip-dropping) and reports missing-column trips; final MIPs now audit q_i artificials + over-covered trips (old "dummy routes: 0" was vacuous); DP dominance now requires charging-stop count ≤ (recharge budget is a resource; pools grow slightly); master LP status checked before extracting duals; hysteresis scales with cap; rc_optimal renamed rc_optimal_restricted. NOT fixed (documented): --G 9999 VSP mode silently clamps to G=300; TARGET_OBJ (fleet−5)×1e5 early-stop is aspirational not a bound; MIN_TRIPS≥3 makes trip-count technically a dominance resource too (accepted heuristic slack); gap limits 57/61/220 remove ~91% of feasible trip-trip arcs per reviewer count — deliberate restriction to historical patterns, revisit for apples-to-apples batch.
