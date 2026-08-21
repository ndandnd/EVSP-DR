# Six-selection duty-union extension

The active manifest is
`scale_ladder_instance_manifest_6sel_seed20260803.csv` (SHA-256
`8bf292bf71229d29feffa7dca4bfaa2f5d6b5943863559468c594a731bd904d3`).

It is a strictly additive duty-union extension:

- its first 22 data rows are the byte-identical legacy manifest at
  `scale_ladder_instance_manifest.csv`;
- the legacy manifest SHA-256 remains
  `a7ef8b77351440a8d7873b949891663ca7b28f135d366d4c6b003d09ca84839a`;
- all legacy paths and instance SHA-256 values are unchanged;
- k40 remains one frozen comparator row;
- only duty-union selections 4–6 for k2, k3, k5, k8, k13, and k20 are
  appended.

The original seed `20260803` already defined six selections in the `per6`
families. The extension emits the previously unused replicates 4–6:

- k2: `pair_union_k2_seed20260803`
- k3/k5/k8/k13: `small_3_5_8_13_per6_seed20260803`
- k20: `large_15_20_30_40_per6_seed20260803`

Every row retains
`weekday_variant_policy=one_literal_per_numeric_base_no_siblings`, records
`duty_count`, `duties_json`, and `duty_set_sha256`, and sets
`target_fleet=scale`.

## Appended rows

| Scale | Selection | Duties | Instance SHA-256 |
|---:|---:|---|---|
| 2 | 4 | `13319,13414` | `62d819d26255fe4daf70d26793bea3dfb072fa74ecfc3b21cad2ef4f8e1bea61` |
| 2 | 5 | `13321,13323` | `64e59a7e8b7e675aceea96dcf33a2e2120a204a87f0755c083413b253d0cb225` |
| 2 | 6 | `13312,13317` | `6fee11f84d907aff9b17429280faf8f74aae0e5f4d1721f9dd79346dce74b4ca` |
| 3 | 4 | `13312,13317,13321` | `3e94695d03f9947d7c0d69c212936d31fa1acca1bc2508b549500cd1ab89d0de` |
| 3 | 5 | `13305,13310,13313` | `f46be52d5512d77690c448d4a94415dbf4a2f50ff07745fcdc29c0ab66d46e01` |
| 3 | 6 | `13323,13409,13410` | `fb1dc3fac8ddc75f629c74ff5fc7602a0ca1758647595b51e6ea5c7f07350f21` |
| 5 | 4 | `13303,13324t,13326,13404,13407` | `8c76d093404ad593da1a4b7e47ba6e3eab34de8726a3b2054c3bd7aabb4b523f` |
| 5 | 5 | `13304,13311,13406,13409,13410` | `f69254a3d6137ca09204360fd68de5f56273df51b6a211650e7f18a9f26ae976` |
| 5 | 6 | `13305,13309,13313,13316uwt,13404` | `bbaed5d68ecaa4d90b3f9d2e7a4b02007e3c6a61f335df74b590205ee2da5b19` |
| 8 | 4 | `13307,13312,13313,13315,13406,13407,13408,13410` | `c8bad8466ec37e2670deccc6e565bc2e8054a747ffb9efa4ec4c54aeef28734a` |
| 8 | 5 | `13306,13316uwt,13321,13323,13402,13405,13408,13411` | `e52495a48dd6c7d79b716a27f7a438303adf1035563afc57bc7c8c76d9ec8e28` |
| 8 | 6 | `13311,13313,13316m,13323,13402,13405,13409,13413` | `ef872319d81bb2cdd580a7c935d8f158bb7b686d0eff8916eac17922353576f8` |
| 13 | 4 | `13303,13305,13307,13308,13312,13315,13320,13405,13407,13408,13410,13411,13412` | `5bf142221055f18b5c83bc294720dfa96cb8196cb5b32aebcb0598f7e459f72d` |
| 13 | 5 | `13302,13303,13304,13306,13307,13308,13310,13315,13319,13325,13411,13413,13414` | `eb8bade5418d3c6ab936cfc30b304c4851cea4cc4d5af95996f606cbb1744343` |
| 13 | 6 | `13301,13303,13305,13315,13317,13319,13325,13326,13404,13406,13409,13410,13413` | `de0187979317c0efa2ce262d6d3de8601de2b2f71084b245c81e9db5d22bdb74` |
| 20 | 4 | `13301,13303,13304,13310,13311,13312,13316m,13318,13320,13321,13322,13323,13324t,13325,13401,13403,13404,13407,13409,13414` | `6bc3f0c4d62e0372f1da40808399121153e070cba12aac765ac7f76392740250` |
| 20 | 5 | `13304,13305,13309,13310,13312,13314,13315,13317,13318,13319,13321,13323,13324t,13401,13402,13403,13409,13412,13413,13414` | `f82e759fccadbefbe8f9362c7f2899f34bf43c738575ca198926bc0e99e4f2f2` |
| 20 | 6 | `13301,13302,13304,13305,13308,13310,13311,13315,13316m,13323,13324t,13325,13404,13405,13406,13407,13409,13410,13412,13413` | `d3ff8894e1de36ac0fc0e6bb7ce325c733688a793c4aee1a185dd747620f9a16` |

The expanded duty-union membership preflight contains 40 cells at
`../known_membership_preflight_6sel_seed20260803.json` with SHA-256
`ba7074a7ed5b342cb64d350fd95945099170b68bc3847619ef94d6f728fbe656`.

The SyntheticRandom corpus from seed `20260821` remains separate under
`random_goal1_seed_20260821/`; it is absent from this manifest and has a
dedicated results table at
`analysis/synthetic_random_goal1_seed_20260821/results.csv`.
