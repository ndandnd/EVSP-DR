# Bounded independent follow-up for Claude

Please independently audit the time-only VSP results in outputs/independent_review_20260916/time_only_vsp_20260916/ once available. Check:

1. The 102 inputs, trip-to-vehicle-group mapping, production time conventions and deadhead sources match the audited chain cases.
2. Every EV-feasible connection is admitted by the claimed time-only relaxation. Direct deadhead compatibility alone can exclude a connection possible via a charger; inspect the shortest-path closure and distinguish it from the direct-matrix comparison.
3. Maximum-matching / minimum-vertex-cover certificates agree and reconstructed path covers serve every trip exactly once.
4. For which cases, if any, the per-group time-only minimum equals GIRO's duty count. Combine this only with a valid group-separated feasible upper bound. Do not infer global optimality from a restricted-pool LP.
5. Correct the proof scope in advisor_seg_lp_20260916/README.md. The weighted LP's route sum is not automatically a fleet-only lower bound; integer-valued group sums do not imply integral route selections. The field segregated_integer_fleet_lower_bound_in_pool is unsupported by that script alone. Keep the verified nine +1 / three unchanged observations.

Return a short table of claims marked verified, refuted, or unresolved, each with file/line or certificate references. No cluster submissions, pool re-solves, new experiments, Doc edits, or changes to active jobs are needed for this task. Put your report in a new dated directory so prior evidence is preserved. The six action3 arms and the full-Partille job are managed separately by Astra; do not duplicate them.
