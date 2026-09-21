# Visual verification

All five PNG figures were opened and inspected at their native aspect ratios on 21 September 2026. Axes, points, bound segments, ticks and legends are readable; no data marks or legends are clipped. Figure 1's long axis labels were split across two lines and the regenerated image was inspected again. PDF and SVG versions use the identical Matplotlib artists and bounding box; SVG text is retained as editable text and PDF fonts are embedded.

Words inside the plots are limited to axes, tick labels and legends. Claims, scientific scope and interpretation remain editable in `RESULTS_PREVIEW.md`; the CSVs retain exact numerical values. Figure 1 uses logarithmic time; Figure 5 uses logarithmic pricing time. Figure 3's segments are optimization bounds, not statistical uncertainty intervals. Figure 4's shapes represent reversed run-order repeats, not independent instances.

The rebuild validates source identities and numeric agreement using archived collected CG/MIP payloads, original chain ancestry, complete local Gurobi logs and the packed benchmark result. It does not rerun the optimization. Local source and output hashes, execution revisions and settings are in `provenance.json` and `experiment_settings.json`.
