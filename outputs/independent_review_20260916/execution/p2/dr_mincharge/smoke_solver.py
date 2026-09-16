"""Two-trip integration smoke; bounded seconds, not a research data point."""
from pathlib import Path
import sys,json
CODE=Path('/home/nc437/ladder-lite/code_pins/review_dr_mincharge_67636d44');sys.path[:0]=[str(CODE/'src'),str(CODE/'tests')]
from run_terminal_energy_cg import run,save
from minimum_charge_pricer import MinimumChargeNetwork
from terminal_duplicate_cleanup import frontier
from validate_terminal_pool_partition import solve
from test_event_pricer_network import two_trip_problem,prices
b=Path(__file__).resolve().parent/'integration_smoke';b.mkdir(exist_ok=False)
net=MinimumChargeNetwork(two_trip_problem(145.4),prices(),soc_step=2.5,block_min=5,g_kwh=240,charge_kw=240,reserve_kwh=0,minimum_charge_minutes=3,fixed_sequence_index=True)
cg=run(net,b/'cg',target=20,fleet_cap=1,cg_seconds=30,mip_seconds=2)
assert cg['cg_pricing_certified'] and cg['charging_stage']['fleet']==1
routes=frontier(net,[0,1]);(b/'fixed').mkdir();fixed,chosen=solve(routes,net.problem.trips,target=20,cap=1,seconds=2,log_dir=b/'fixed',threads=1)
assert len(chosen)==1 and fixed['exact_once_verified']
assert abs(cg['charging_stage']['cost']-fixed['charging_cost'])<1e-5
for r in chosen+json.loads((b/'cg/selected_routes.json').read_text()):
 for a,z,e in zip(r['charging_stops']['cst'],r['charging_stops']['cet'],r['charging_stops']['kwh']):
  if e>1e-9:assert z-a>=3-1e-8
save(b/'passed.json',dict(status='passed',cg=cg,fixed=fixed,synthetic_smoke_only=True))
print('Integration smoke passed: fresh CG certificate, matched fixed-duty one-bus MIP, identical graph costs, actual active duration>=3min.')
