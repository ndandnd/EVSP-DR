# Reserve-feasibility screen

This ten-case campaign asks whether four adjacent k1 GIRO duties remain tractable in the conservative constant-rate event model after jointly changing usable battery energy from 240 to 236.44 kWh and imposing a 35.466 kWh reserve. It separates baseline and PARX 60 kW charging for all duties, then adds shared-capacity and combined cells only for duty 13408, the capacity case already known to certify quickly at 240 kWh/reserve zero.

The 236.44/35.466 setting changes two parameters together and is a sensitivity test. It is not an isolated reserve effect, does not implement the nonlinear 18E1 curve, and does not impose or infer a 65% terminal floor.

Every case uses driver commit `309d98d266ebaf6b7e99543a67f8f2be5736874a`, flat prices, prefix-memo, a 13,200-second CG budget, and a 600-second dedicated finite-pool MIP diagnostic. A CG certificate requires exact nonnegative reduced cost and zero artificial weight. A timed pool may remain usable, but is never a full-model certificate. The MIP proves only its saved pool. Physical claims are limited to driver construction and its continuous half-open station-capacity audit; no separate continuous route replay is added.

Standalone collection:

```bash
python collect.py --root /home/nc437/ladder-lite/reserve_feasibility_screen_20260914
```
