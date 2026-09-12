# Document figure recovery and clearer explanations

12 September 2026. The original Google Doc keeps its URL. Its weekly status is separate from two persistent figure tabs:

- [Week of 14 September](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lumf8xm66fow): three pages of status, editable tables and simpler explanations.
- [Figures with explanations](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i): five figures over five pages. The original timing, matched-energy charging and geography plots remain; two new charts explain the saved-route checking and capacity-pricing delays.
- [CG curves and bus schedules](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly): fourteen figures over fourteen pages, including all ten original meeting images plus the fresh-versus-inherited comparison, a recent fresh k10 Gantt, and two charging-time figures.

The [historical archive](https://docs.google.com/document/d/1f0orWtM1-_VWAqjj6GnWP78webCOnvoc01x2VQvaA_k/edit) remains unchanged. No figures were removed in this recovery. Future updates must preserve embedded figures and editable captions; simplifying a document must not remove its visual evidence. The hourly monitor and runbook now record this preference. Slides and cluster jobs were not changed.

## What the explanations now say

Checking saved routes happens before the new CG search. The earlier full-pool runs could spend their entire budget checking routes. The new runs select at most 512 sequences and allow 15 minutes for checking. Seven completed cases used roughly 5–8 minutes for this step. New starting pools may change the integer solution; no paired speedup or new integer match is inferred from these timings.

Capacity-aware pricing is a different delay inside CG. For k3 iteration 3, the LP took 0.005633926019072533 seconds and one new-route search took 25722.64842124097 seconds. The deadline-stopped run generated only three columns and did not certify pricing optimality. Its sixteen-bus pool solution is not proof that sixteen buses are physically necessary.

## Evidence and preservation

- `bottleneck_generator.py` reads existing source artifacts; `bottleneck_provenance.json` records exact values, paths and hashes.
- `figure_inventory.md` documents older figures and their model/proof limits.
- `recovered_meeting_figures.json` records extraction from the archived Google Doc export.
- `image_preservation_check.json` confirms all ten original meeting PNGs occur byte-for-byte in the restored DOCX export.
- `exports/SHA256.json` records final PDF, DOCX and Markdown exports; all nineteen image instances are embedded in the two figure tabs. Tables and captions remain editable.
- The first six restored convergence plots are earlier fresh set-partitioning traces, not the new warm set-covering experiments. The historical charging-time figures have unequal terminal energy and are labeled accordingly.

The result snapshot is still 12 September 2026 at 01:27 EDT. This task rewords and restores evidence; it does not collect or claim new solver results.
