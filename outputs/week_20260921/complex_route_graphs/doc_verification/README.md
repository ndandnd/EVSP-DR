# Google Doc verification — complex original bus day

21 September 2026. Edited only the **Figures with explanations** tab of the existing [research document](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i).

Added a first-page section for recorded GIRO duty 13309: the five-area, three-charging-site diagram, a short explanation, explicit original-only scope, and immutable source links. The previous figure section starts on the following page. No Slides, current-results, history or CG-curve tab was edited.

`figures_before.md` is the before export. The before PDF is the immediately preceding verified export at `../../spatial_schedule_graphs/doc_verification/figures_after.pdf`. `figures_after.md` and `figures_after.pdf` are the final exports from the live document. `verify.py` checks preservation of the earlier body and every earlier embedded figure by decoded image-pixel hashes, checks the single new image and source links, and rejects unfinished placeholders. Its output is `verification.json`.

The first two PDF pages were rendered and visually inspected for heading, paragraph, figure and page-break layout. The full-resolution figure and itinerary remain linked because a five-node day graph is necessarily smaller in a portrait document than in the standalone artifact.
