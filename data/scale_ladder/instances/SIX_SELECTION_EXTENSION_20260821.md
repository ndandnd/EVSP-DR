# Six-selection scale-ladder extension

The active manifest is
`scale_ladder_instance_manifest_6sel_seed20260821.csv` (SHA-256
`d137fcdfaa6aba5d9425d5b0631d448609b71dc5c0474612c5ab38095a36a53d`).

This is a strictly additive extension:

- its first 22 data rows are the byte-identical legacy manifest at
  `scale_ladder_instance_manifest.csv`;
- the legacy manifest SHA-256 remains
  `a7ef8b77351440a8d7873b949891663ca7b28f135d366d4c6b003d09ca84839a`;
- all legacy instance paths and SHA-256 values are unchanged;
- k40 remains the single frozen `selection_replicate=2` row;
- only selections 4–6 for k2, k3, k5, k8, k13, and k20 are appended.

The new rows use `src/generate_random_goal1_instances.py` version 1 with seed
`20260821`, replicate range 4–6, and generator family
`generate_random_goal1_instances_v1_seed20260821`. The source table SHA-256 is
`6b46acce8b0870aff967c73aac372b90873ed32a6e424e55b851e4b8676ab57f`.
The complete generator record is in `random_goal1_seed_20260821/manifest.json`.

## Appended rows

| Scale | Selection | Relative path | Instance SHA-256 |
|---:|---:|---|---|
| 2 | 4 | `random_goal1_seed_20260821/Practice_SyntheticRandom_2bus_s20260821_r04.csv` | `ac38cf11c29d05aa8c6d8f8c20f95b22fb5a75eb279dce27369226dc73580965` |
| 2 | 5 | `random_goal1_seed_20260821/Practice_SyntheticRandom_2bus_s20260821_r05.csv` | `fc5961e568321643317f89fa0398d750c077181b8ad49c0e5bf94e64ef873645` |
| 2 | 6 | `random_goal1_seed_20260821/Practice_SyntheticRandom_2bus_s20260821_r06.csv` | `39c2e81cfc88c0d8772c13e4f34e7bde506be8230ab098fc76290ab160ed7229` |
| 3 | 4 | `random_goal1_seed_20260821/Practice_SyntheticRandom_3bus_s20260821_r04.csv` | `8f7e00da6b11bba1a96778b5df0a90aad04c687165a2809f140de0f883a26c96` |
| 3 | 5 | `random_goal1_seed_20260821/Practice_SyntheticRandom_3bus_s20260821_r05.csv` | `8d1001850dba6d12e48c415124a2c7814c8371722520bac9c089bbfc46aaf6bb` |
| 3 | 6 | `random_goal1_seed_20260821/Practice_SyntheticRandom_3bus_s20260821_r06.csv` | `dac224a86c74a6c31bd168a63ed1018e03cef2bc0db802861d134cbd19985864` |
| 5 | 4 | `random_goal1_seed_20260821/Practice_SyntheticRandom_5bus_s20260821_r04.csv` | `d102ac26515de61e01b37e4ffc8ba00c8d7a9b2148f6d8a871a3bac1221d2aa0` |
| 5 | 5 | `random_goal1_seed_20260821/Practice_SyntheticRandom_5bus_s20260821_r05.csv` | `6d812db104e3388858fab40f8da85da96cdb65ec578728177ff21360e5f3ce9e` |
| 5 | 6 | `random_goal1_seed_20260821/Practice_SyntheticRandom_5bus_s20260821_r06.csv` | `1958dcb3b98f1164f7979cef88a43cd451658aff0e1e07c6fbdc0a266433c313` |
| 8 | 4 | `random_goal1_seed_20260821/Practice_SyntheticRandom_8bus_s20260821_r04.csv` | `c0b3f61876b5979af2fc41e700795aaac7a102a981068045e0eaef2e358211a8` |
| 8 | 5 | `random_goal1_seed_20260821/Practice_SyntheticRandom_8bus_s20260821_r05.csv` | `80ff6780febc2af0ec1cfe7e92f3c06480b04307de1e239b2832d24d557813d6` |
| 8 | 6 | `random_goal1_seed_20260821/Practice_SyntheticRandom_8bus_s20260821_r06.csv` | `26da9ad1b3839a6ed35db7e6ecb9123f035521b9270c9aef1e92d8a9e1b699ed` |
| 13 | 4 | `random_goal1_seed_20260821/Practice_SyntheticRandom_13bus_s20260821_r04.csv` | `9bba791c3ec26f6b3ae277593c00bfbe2f3a90c7fc54ebbfb6e134605a8edbec` |
| 13 | 5 | `random_goal1_seed_20260821/Practice_SyntheticRandom_13bus_s20260821_r05.csv` | `c7a23f8648e7336fd4a22d1dc865034601f8432ca52b9e5102c053bc5296e19c` |
| 13 | 6 | `random_goal1_seed_20260821/Practice_SyntheticRandom_13bus_s20260821_r06.csv` | `f698b47764bf1a86de3b16849e9a56c22f8376b5c848d0bec33bbd6e43142823` |
| 20 | 4 | `random_goal1_seed_20260821/Practice_SyntheticRandom_20bus_s20260821_r04.csv` | `bade6e36e562c4cbdf760f18c8756e40ea13d75a14ba9b969e7ca7e76897d4ef` |
| 20 | 5 | `random_goal1_seed_20260821/Practice_SyntheticRandom_20bus_s20260821_r05.csv` | `6f3e91a7869d6d3283a72a06c648dde32ddce7e7cfbe545b169032e3f2f0d77c` |
| 20 | 6 | `random_goal1_seed_20260821/Practice_SyntheticRandom_20bus_s20260821_r06.csv` | `f32181e6b544ba45beb4c63b64981b10fc3f0fc6b48e302eeb389fa437a26923` |

The expanded membership preflight contains 40 cells and is stored as
`../known_membership_preflight_6sel_seed20260821.json` with SHA-256
`d7791702562ca648b076de0c2554a70696ca394756876c8aba8b7857c9958f69`.
