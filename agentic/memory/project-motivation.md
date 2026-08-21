---
name: project-motivation
description: "Why EVSP-DR exists — validate CG pipeline against GIRO's solution, then demonstrate savings from time/location-varying electricity prices"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5411361d-3fc9-42c9-8275-f6af541497cb
  modified: 2026-08-01T14:16:40.496Z
---

Core motivation (stated by Nathan, 2026-08-01):

1. GIRO's solver produced the schedules in `data/original_giro_data/*VehicleDetails.xlsx` **without considering electricity prices** that vary by time of day or location (e.g., could be ~free at a solar-equipped station).
2. Our trip set T is **reverse-engineered from GIRO's solved schedule** (Regular rows of selected VehicleTasks) — so even the instance definition inherits GIRO's solution structure.
3. We do NOT expect our elementary DP + column generation to out-engineer GIRO's sophisticated solver. **Goal 1 (validation): get NEAR GIRO's solution quality in reasonable time**, so we trust the pipeline. (This is what the CHEAT vs NO_CHEAT/GREEDY 3h/12h experiments measure: CHEAT = GIRO's own routes seeded as columns.)
4. **Goal 2 (the point): switch on time- and station-varying electricity prices and show GIRO there are savings from price-aware vehicle routing/charging.** The headline number should be: cost of GIRO's fixed schedule evaluated under the price curve (CHEAT columns, no CG) vs. our price-aware optimized schedule.

Implications to keep front of mind:
- Parity validation should use FLAT prices (`hourly_prices.csv`, all 1) so charging cost doesn't distort the GIRO comparison; fleet size (×1e5) is the dominant term. TARGET_OBJ=(N−5)×1e5 is beyond-parity aspiration, not a bound.
- Any bus-count win below GIRO's N may just reflect our dropped GIRO rules (crew/interlining/layover/20% SOC) — report as relaxation, not savings.
- **Structural threat to Goal 2**: the DP charges immediately on arrival (cannot wait for a cheaper hour) and the 57-min trip-to-trip gap limit (fit to GIRO's cost-blind schedule) forbids idle-through-the-peak patterns. The current pricer largely CANNOT express load-shifting — the very behavior the project wants to monetize. Delayed-charging-start options (or de Vos-style block-based charging nodes, see [[evsp-cg-literature-findings]]) are needed before the savings demo is credible.
- Spatial price variation is unblocked as of 2026-07-31 (copy-name price lookup fixed); `STATION_PRICE_MULTIPLIER` in config is still dead and the spatiotemporal CSVs replicate identical curves — a real spatial scenario file needs to be generated (e.g., solar station near-zero midday, or Karl's SEK curve with per-station multipliers).
- GIRO (Anne Mercier) offered to re-optimize with cost numbers and compare — an external benchmark for Goal 2 ([[giro-data-provenance]]).
