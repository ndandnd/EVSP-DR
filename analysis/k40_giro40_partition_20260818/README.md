# Verified frozen-k40 GIRO40 partition

This immutable input contains 40 physically validated duties covering the
frozen 947-trip k40 instance exactly once under 300 kWh, 300 kW, zero reserve,
15 kWh SOC steps, 10-minute charging blocks, and the flat tariff.

- Artifact SHA-256:
  `8f9944f93f26cf0121e9ecab2fa412d573e90a0189b7a38008d3b2535f54d428`
- Canonical partition SHA-256:
  `9e007d51c6bbbdc4f01a00a26ba3bcfa1ec4340df9aab8227a12cf0dc35ecb11`
- Route-set SHA-256:
  `9b42579ae2d013706cc8d523eb9313fdef4e36eb492a99356483cb526d00085a`
- Trip-set SHA-256:
  `35604b22facf1646963e85eb98a858906f0dd7dbebd86ea0d3ac7b797de62ed0`
- Instance SHA-256:
  `3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd`
- Tariff SHA-256:
  `1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200`
- GIRO source SHA-256:
  `6b46acce8b0870aff967c73aac372b90873ed32a6e424e55b851e4b8676ab57f`

The authoritative source has 42 literal duty IDs representing 40 base duties.
The frozen k40 instance includes `13316uwt` and `13324t`; it excludes
`13316m` and `13324muw` because each is the weekday variant of an included
base duty. The “43” in `Practice_43bus.csv` is a historical filename/target,
not a third duplicate variant and not this partition's source.

All routes contain deterministic continuous charging schedules and tariff-bound
block hashes. Their costs are valid master costs for the injected routes. They
have no continuous-cost reduced-cost or pricing certificate.

Two duties (`13323` and `13406`) use a zero-energy station waypoint because a
direct timing arc is unavailable. The current model charges its $5 station-stop
cost for those waypoints, so the recorded total is $10 above a hypothetical
positive-energy-only start-cost convention. This campaign preserves the
reviewed model convention and does not claim cross-convention cost optimality.

Regeneration is fail-closed on both frozen status hashes, the reference and
deadhead hashes, 15 kWh/10-minute discretization, and the reviewed route-set
hash. A solver/version-dependent alternative realization is rejected rather
than silently replacing this artifact.
