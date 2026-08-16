# MIP statistics campaign inventory (cloud dry run)

`inventory-current.json`, `pilot-plan-current.json`, and
`secondary-plan-current.json` are the canonical generated read-only artifacts
from `src/launch_mip_statistics_campaign.py`.

The cloud checkout does not contain the Unicorn result roots or release
archives. Consequently the inventory contains no verified candidates and both
plans are blocked, with no Slurm commands or jobs. This is intentional:
similarly named pools are never substituted.

Regenerate from a detached tracked-clean checkout after installing the explicit
verified inputs and exact GIRO starts documented in
`MIP_STATISTICS_CAMPAIGN_20260815.md`. Only that regenerated plan can be
considered for approval.
