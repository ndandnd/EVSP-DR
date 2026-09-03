# k3/k4/k6/k7 transition-gap inputs

This directory freezes the inputs for the 2 September 2026 transition-gap
campaign. It contains six fixed-seed probability selections and three
pre-outcome structural-stress selections at each of k3, k4, k6, and k7.

The files have three distinct purposes:

- `candidate_universe.csv` records all 512 eligible candidates per scale and
  identifies the selected rows.
- `selection_manifest.csv` identifies the 36 immutable solver inputs and their
  pre-outcome features.
- `known_duty_continuous_240_240.csv` records an independently optimized and
  physically replayed continuous charging certificate for every source GIRO
  duty order at 240 kWh / 240 kW / zero reserve.

The physical certificates establish that the original k duty orders compose a
k-route physical upper bound under the current idealized continuous model.
They do not establish representability in the 2.5-kWh event lattice, do not
prove that k is optimal, and are not injected into the RAW column-generation
runs.

The probability selections are the first six eligible draws from independent
scale-specific fixed-seed streams. Existing six-selection ladder sets are
excluded. The stress selections are then chosen, without solver outcomes, as
the remaining candidate with maximum trip count, maximum service energy per
duty, and minimum median scheduled inter-trip gap. Previously chosen stress
sets are skipped so the three roles are distinct.

`input_plan.json` binds the source-data identities, selection rules, physical
assumptions, and hashes of the selection evidence. Validate the directory with:

```bash
python3 scripts/event_uniform_envelope/validate_transition_gap_inputs.py \
  --repo .
```

The generator requires the validated six-selection manifest from input commit
`ff7fb2ba93cf13a31171e1e4aeb2d28dc8aeee20` so it can prove that no new duty
set duplicates the existing ladder.
