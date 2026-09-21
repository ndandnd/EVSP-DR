# Google Doc update and preservation checks

Updated the existing Current results and Figures with explanations tabs. The journal has a short LP-agreement and battery-capacity audit, an editable results table, and source links. The figure tab adds Chain 1, the six-chain overview, and the full-day duty13309 comparison. Individual larger figures and all editable data are linked.

`verify.py` compares fresh before/after Markdown and PDF exports. All eleven checks pass: original text, tables, source links and all fifteen earlier figure images are preserved; exactly three new figure images are present; result counts, physics limits and numerical certificate scope are present; no placeholders remain. Current results grew from eight to nine pages, the figure collection from seventeen to twenty-one. History, CG-curve tabs and Google Slides were not edited.

The new pages were rendered and inspected. Chain 1 has its own page; the overview and route introduction share the next; the full-day three-panel graph has a full page, followed by editable scope notes and links. The earlier figure heading was restored during editing before final verification. No prior figure was removed or altered.

Sources were published at `3f2228605e0c9e1cc46140a3dbd03fd6c1f6e3ce`; remote ref verification passed. All activity was posthoc: no new CG/MIP solves or cluster submissions.
