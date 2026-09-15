# Frozen six-chain extension k31–32 — launched 15 September

Production launched at 10:24 EDT. Graph array228593 has12 tasks; the paired CG/MIP jobs are228594–228617. All12 graph tasks were verified running;24 downstream CG/MIP tasks retain their required inputs. `root_launch_audit.json` checks effective scheduler resources/dependencies and `mip_registry_receipt.json` records12 new MIP registrations.

Twelve input files are byte-identical copies from the original150-case frozen chain_extension_20260913 input manifest; no new draw or shuffle. Stable Ordered_Trip_ID nesting and unchanged prior-trip attributes (except count_trip_id) pass independently.

Remote root `/home/nc437/ladder-lite/chain_extension_31_32_20260915`; large outputs `/share/scaglione/nc437/evsp-dr/chain_extension_31_32_20260915`.

| Chain | k31 trips | k32 trips | Parent k30 CG producer |
|---|---:|---:|---:|
| C1 | 737 | 768 | 224631 |
| C2 | 725 | 751 | 224635 |
| C3 | 716 | 753 | 224639 |
| C4 | 734 | 749 | 224643 |
| C5 | 716 | 731 | 224647 |
| C6 | 742 | 779 | 224651 |

CG commit a0e0bb7681c8451e3cbbbfa06aef390026d9af4b; MIP commit871d057e1067411f09581e37d78f7c1ca43f68bb. Physics/settings exactly equal k29–30:240kWh battery/initialSoC,240kW, covering, no reserve/shared capacity/terminal floor, flat tariff,100000 per route + electricity +5 per charge start. Unlimited previous-k full-pool inheritance with8workers; CG4h; two-stage MIP1h with30min fleet stage and stage2 fleet <= validated incumbent.

| Stage | CPU | RAM | Slurm | Native budget/watchdog |
|---|---:|---:|---:|---:|
| Graph |2|64G|25h|24h|
| CG |8|96G|5h|4h scientific;4.5h worker|
| MIP |8|24G|2h|1h scientific;1.75h worker|

All jobs default_partition and exclude scaglione-compute-01. All12 independent graphs eligible (0-11%12). Each k31 CG waits its graph and own k30 producer; each k32 CG waits its graph and own k31 CG. Each MIP waits only its own CG. Published parents may replace pending producer dependencies only after identity/hash/provenance authentication.

Graph resource basis retains prior measured12h watchdog exhaustion and successful24h recovery189917 (11:55:46,37861336KiB MaxRSS). Successful k28 CG187984/188002 report142762740/179778832KiB MaxRSS despite96G requests. Running187972190137176KiB observation cannot alone establish physical memory use or OOM; fork/shared-memory accounting interpretation remains unresolved. Existing requests retained. Raw successful accounting and current remote policy are in preparation/.

Native fixture228576 uses2CPU/8G/15min, default_partition with node exclusion and no requeue. Passed all8checks; validation.json and validation/job_228576 preserve certificate and all fixture artifact hashes. Fixture is not research data. Manifest/source/tool hashes and independent input checks precede the fixture. Immutable fixture artifacts/certificate are retained.

`review_checks.json` verifies unchanged scientific/resource settings, source/static hashes, exact case range, staged tool hashes, and AST equality of every campaign function except preparation metadata and the added prelaunch duplicate gate. `campaign.diff` reviews all scripts against k29–30. `duplicate_launch_audit.json` checks all campaign case_jobs/jobs maps and live queue; this gate repeats immediately before plan/submit. `submit` refuses an existing production jobs.json. No solver source edited.

The exact executable review plan is submission_plan.json (25 submissions representing36 tasks). Native fixture and prelaunch checks passed. Production command already executed once; do not repeat:

```sh
ssh -S /Users/nadan/.ssh/evsp-unicorn.sock -o BatchMode=yes -o ConnectTimeout=8 nc437@unicorn-login-01.coecis.cornell.edu '/home/nc437/evsp_env/bin/python /home/nc437/ladder-lite/chain_extension_31_32_20260915/campaign.py submit'
```

Root owns production review/submission, collector/register/preemption integration and Google Doc update. Shared state, Slides, publication worktree, held537227, and EVSPV2G untouched. Scheduler state, CG pricing certificate, finite-pool MIP proof, physical validation and GIRO target remain separate; an RMP value is not a full-model lower bound without the corresponding pricing certificate.

Manifest SHA256: `c05a9337f232d9b21acdeafeb5e9311630b28d88674c9b312ff97d7165f2e735`. Native certificate SHA256: `d8b78d9dea20672003dc806dba4d9d3b685047ab4c540fcbf4759794727aa954`. Final local review also authenticates all copied fixture artifacts and every planned dependency/resource/exclusion.
