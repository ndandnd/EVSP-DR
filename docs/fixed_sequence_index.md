# Opt-in indexed event replay

Use `--time-model event --fixed-sequence-index` with
`src/exact_pricer_expanded.py`, or pass `fixed_sequence_index=True` to
`EventExpandedNetwork`. For an existing graph, call
`network.set_fixed_sequence_index(True)`; `False` restores the reference scan.
The default remains the reference scan. The execution's network metrics record
`fixed_sequence_index`; this field is not part of cached graph metrics.

The flag changes only `fixed_sequence_record` outgoing traversal. Binary searches
select the complete half-open node range of the required trip, including every
SOC state, or the sink. All retained arcs stay in original order. The existing
cost/edge-list tie comparison, station-alternative deduplication, action lookup,
record realization, and physical validation are unchanged. Core pricing is
unchanged. This does not introduce a new certificate or change proof metadata.

The index stores two integer bounds per trip: O(trips) additional runtime memory,
with no per-arc Python index. Bounds are checked in O(nodes) on first opt-in.
The builder already allocates contiguous trip nodes and sorts every finalized
source row by target. A versioned invariant is recorded only after graph
construction succeeds, so new graphs and their hash-validated caches need no
additional O(arcs) validation pass. Runtime bounds are omitted from pickles and
replay mode resets to baseline on load; the CLI reapplies its flag after loading.
The only additional persisted fields are a small invariant version and the false
runtime mode, independent of graph size.

Old pickles without the invariant remain readable. Their first opt-in validates
row ordering once: explicit graphs scan their Python rows; lazy graphs use NumPy
comparisons in at most 65,536-entry chunks, including cross-chunk comparisons,
with O(nodes + arcs) work and bounded temporary memory. Subsequent toggles and
newly serialized validated caches reuse the invariant. Unsorted rows or
noncontiguous trip-node blocks raise `ValueError`, leaving indexing disabled.
As elsewhere in the graph implementation, arrays/rows must remain immutable
following construction. A pickle is trusted executable input; the existing
loader's full hash and source-identity checks still run before unpickling.

Cache identity remains strict, including execution git commit, physical inputs,
and arc mode. The replay flag is deliberately excluded because both modes use
identical graphs. This permits paired baseline/index runs to use the same cache
under the **same execution commit**; it does not authorize relabeling a historical
cache with a new commit or bypassing source identity. Structural compatibility
with old pickle state does not weaken that provenance gate.

Inherited selection, deduplication, worker scheduling and budgets are unchanged.
Keep the campaign's 512 selected sequences, 900 seconds, and 8 workers in both
paired runs. Tests exercise that configuration on a frozen tiny pool, comparing
selected inputs, full accepted records (sorted because workers are unordered),
rejections, and audit metadata. A binding deadline can still produce different
completed subsets as execution times change; exact per-sequence equivalence does
not imply identical deadline-truncated pools or a whole-CG speedup.

Validation: `tests/test_fixed_sequence_index.py` exhausts sequences of lengths
0–4 on explicit/lazy tiny graphs, including repeats, unknown IDs, infeasible
transitions, trips with no SOC nodes, tight rounding, tariff boundaries, and
equal station alternatives. It compares full records and action traces, verifies
complete-successor arc subsets, tests legacy/new cache handling and source/hash
rejection, and exercises the actual bounded importer with eight fork workers.
