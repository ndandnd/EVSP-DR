# Figure library export QA — 12 September 2026

**Result: PASS.** The final `figure_library.pdf` has 14 pages, and the matching DOCX contains 14 PNG image instances. Each page has exactly one embedded figure; there are no blank pages, missing images, clipped image edges, or heading/caption breaks across pages.

## Checks

- PDF: `exports/figure_library.pdf`, 14 pages, letter size; all 14 rendered pages are present under `qa/figure_library/page-01.png` through `page-14.png`.
- DOCX: `exports/figure_library.docx`, 14 `word/media/image*.png` entries; all 14 page images have nonzero dimensions and render successfully.
- PDF resource audit: one image XObject on every page. Image dimensions are preserved as 2048 px wide; pages 1–8 and 14 are 2048×1152, pages 9–10 are 2048×1613, page 11 is 2048×1159, page 12 is 2048×870, and page 13 is 2048×1536.
- Layout audit: headings, explanatory text and their figure remain on the same page on all 14 pages. The lowest extracted text is above the page bottom margin, and every image box is wholly inside the printable page area.
- Visual audit: all 14 rendered pages show complete figures, legends, axes and plot labels. Page-level headings and captions are readable; the dense historical Gantt and charging plots retain their small internal labels, which may require zoom when viewed digitally.

## Caption and experiment-scope audit

- Pages 1–6 are explicitly grouped as the earlier 10 September fresh-start, 240 kWh / 240 kW **set-partitioning** convergence traces. The source is the original-chain `nested84` campaign with singleton RAW initialization; these pages are clearly separated from the later warm covering material. Page 1 also states that these are not the new warm covering runs.
- Page 7 remains the earlier fresh-start CG wall-time composition and says it excludes network preparation and the final integer solve.
- Page 8 identifies the earlier three-way charging-cost comparison and warns that return energy was not matched.
- Pages 9–10 correctly distinguish historical May RND002 GIRO (`CHEAT`) and no-GIRO (`NO_CHEAT`) schedules, including the artificial-feasibility and time-limit cautions.
- Page 11 is correctly labeled current chain-3 **set covering**, comparing fresh singleton initialization with previous-k inherited columns. Its `OPT`, `F` and `~` proof-code explanation matches the saved figure provenance.
- Page 12 is correctly labeled a current fresh **set-covering** k=10 schedule with 11 buses and explicitly distinguishes it from the later ten-bus warm solution.
- Pages 13–14 correctly identify earlier five-bus, 62-trip charging-demand/window figures and state the unequal terminal-energy and unobserved within-window-power limitations.

Editorial follow-up only: page 1 retains the source sentence “Use Week of 14 September for the latest verified results.” If this tab is meant to describe the 12 September cutoff, verify that forward-looking week label separately; it does not affect the export layout or figure recovery.
