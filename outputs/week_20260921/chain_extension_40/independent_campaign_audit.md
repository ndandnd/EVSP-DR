# Independent k33–40 campaign audit

**No blocking issue found in the prepared campaign.** Native validation is passed.

Manifest SHA256: `e3d7164834a388b69def78840acf6038129d9aa8dddc18f9db3bcfe6e575046e`. Review time: 2026-09-21T18:38:16.715986+00:00.

The k31–32 comparison is exact for both solver commits, all recorded Python source hashes, physics/objective/discretization settings, and tariff/reference/deadhead hashes. All48 predecessor case/hash/status links were checked. Each CG depends on its own graph and prior-k CG; each MIP depends on its own CG.

The44 graph owners are safe byte-identical aliases under common source/configuration. Shared groups are C1/C4 at k40, C2/C3 at k39, and C2/C3/C6 at k40. CG pools remain separate. The native test explicitly exercises the same bytes under two filenames.

Pinned MIP code imposes **fleet <= validated stage1 incumbent** during the charging stage. Proven-fleet and unproven-cap results remain distinguished. Published CG output only means a usable zero-artificial LP; its pricing-certificate flag remains separate. MIP publication requires physical replay.

CG and MIP budgets/resources are unchanged. Graph memory rises64G→96G and allocation25h→37h, with a36h watchdog; the51.30GiB observed maximum and quadratic projection78.17GiB/24.37h support the headroom. All44 distinct graph tasks may run concurrently; default_partition and scaglione-compute-01 exclusion are explicit.

Preemption requeue preserves unique attempts. CG resume validates identity and includes previous elapsed time in its four-hour budget. Unpublished graph and MIP attempts restart; failed/signal/watchdog attempts are retained and not published. Existing jobs.json prevents a blind second launch. A partial submission or corrupt checkpoint therefore needs deliberate recovery, not deletion of records.

The native four-trip test is an integration check, not a guarantee that full graphs fit. Scheduler acceptance/resources/dependencies must still be checked after submission. This continuation uses the original240kWh/240kW/zero-reserve/no-shared-capacity baseline; it does not answer strict GIRO feasibility.

[Machine-readable checks and source hashes](independent_campaign_audit.json). No campaign files were changed and no jobs were submitted by this reviewer.
