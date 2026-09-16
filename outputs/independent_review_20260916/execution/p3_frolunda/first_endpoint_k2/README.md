# F8 — Frölunda k2 endpoint independently checked

**VERIFIED:** two buses cover all **38 trips exactly once**. Both selected routes contain 19 trips and pass a fresh physical replay with **zero charging-arrival grace**. Their continuous charging blocks start after vehicle arrival and reproduce the stored continuous costs.

Original CG/MIP bytes match their saved hashes. The input, reference, deadhead and tariff hashes match the execution records. Replay source modules match the pinned submitted MIP commit `0c51b1fa491492212f5d875d1c082b476db0f1c3`.

The native CG reports a pricing certificate after **110 iterations / 51.64 seconds**, weighted LP objective **200,099.976**, fractional route weight **2**, zero artificials and last reduced cost about **−1.39×10⁻¹⁰**. Native MIP reports fleet 2 and saved-pool fleet bound 2. Those optimization claims were read from native artifacts, not independently re-solved here.

An independent elementary fleet bound is also available: mandatory trip **1314** occupies **06:06–06:48**, while trip **1337** occupies **06:36–07:15**. Their strict overlap requires at least two vehicles. Together with the replay-validated two-vehicle solution, this establishes **fleet optimum 2 for this baseline instance**, independently of the CG or MIP solver's proof flags. It does not establish charging-cost optimality.

Scope: 240 kWh battery, uniform 240 kW, zero reserve, no terminal SOC requirement, flat electricity price and 5 per charging start; set covering in optimization, exact-once service in the returned solution. Shared station capacity and the omitted GIRO constraints were not checked. The Frölunda deadhead conversion assumptions remain those documented in the parent README.

`source.json` preserves original remote paths and collected hashes; `cg.json` and `result.json` are byte-identical copies. `verify.py` reproduces this audit; `verification.json` contains route memberships, replay outcomes and the overlapping-trip witness.
