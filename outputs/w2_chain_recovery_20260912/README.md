# Chain 2 continuation after importer recovery

Only the three successors blocked by original job949703 are recovered: k14 saved-pool MIP, k15 inherited CG and its MIP. The corrected parent15687 completed0:0, CG certified, no artificials. Original status SHA f82e815f0c2e02511ab9e3dee599b05c3a1d2ffb374977d0f040a1759dc52415; journal d9248ea26caa1af5e86070f9d1fb29a516bd871cdde4535819dcba8601d69013. Completed artifacts and exact source inputs validated. Original949704/5/6 blocked jobs remain historical and untouched.

CG source68fce009; MIP source871d057. Same240/240, covering, flat tariff,2.5kWh/5min graph,512-route/900s inherited replay,14400s CG budget. MIPs3600s total, stage1<=1800; stage2 fleet at most validated incumbent. CPU8; CG96GiB/4h45, MIP24GiB/2h. Default partition and reserved-node exclusion. Exclusive attempt outputs; no automatic requeue because a new attempt must receive a new path.

Completed k14 parent is frozen as actual data dependency. k15 MIP will use afterok on the new k15 CG. Two immediately eligible successors can run concurrently. K15 cache uses audited unchanged graph semantics with separately verified input and actual pickle hash. Consumer attestation retains original producer identity.

Validation before submission: parent returncode0/watchdogfalse and hashes verified; input hashes verified; fixed CG/MIP source heads checked; entry Python compiled and shell syntax checked. Generic isolated worker is the already tested68fce009 implementation. No new solver algorithm changes. Final submission ledger records exact jobs and effective node exclusion.
