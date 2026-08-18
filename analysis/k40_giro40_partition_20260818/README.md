# Verified frozen-k40 GIRO40 partition

This immutable input contains 40 physically validated duties covering the
frozen 947-trip k40 instance exactly once under 300 kWh, 300 kW, zero reserve,
15 kWh SOC steps, 10-minute charging blocks, and the flat tariff.

- Artifact SHA-256:
  `2afdc10c142b468e065b6330c7be43b0b91479402c924f3c23e7b45e9e09a06b`
- Canonical partition SHA-256:
  `9a71179b79072969264d04326f58214c51cf16096de7cd17b05d3a140d30ebe6`
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
