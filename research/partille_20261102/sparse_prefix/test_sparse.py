"""Tests against the pinned solver's synthetic networks; no private trip files required."""
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from sparse_worker import limits,TOTAL
base=Path(os.environ['PREFIX_SOLVER'])
sys.path[:0]=[str(base/'src'),str(base/'tests')]
saved_argv=sys.argv[:]
sys.argv.extend(['--start-fee','5'])
from run_capacity_terminal_cg import run
sys.argv=saved_argv
from test_capacity_event_pricer import problem,prices,PLUGS
from capacity_event_pricer import CapacityTerminalNetwork, record_occupancy
from prefix_initializer import REQUIRED_PHYSICS, authenticated_sequences,digest,regenerate,remap_sequences


def write_instance(path, ids, mutate=False):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['Ordered_Trip_ID','From1','Start1','End1','To1','Usage kWh'])
        w.writeheader()
        for i in ids:
            w.writerow(dict(Ordered_Trip_ID=i,From1='a',Start1='10:00',End1='10:20',To1='b',
                            **{'Usage kWh':str(i+int(mutate))}))


class Check(unittest.TestCase):
    def test_total_phase_limits_and_overrun(self):
        for n in (2,3):
            alloc=[limits(n,i,dict.fromkeys(TOTAL,0)) for i in range(n)]
            self.assertEqual({k:sum(a[k] for a in alloc) for k in TOTAL},TOTAL)
        self.assertEqual(limits(2,1,dict(cg=4000.8,heuristic=0,mip=0))['cg'],6799)

    def test_row_order_and_changed_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'old.csv',Path(tmp)/'new.csv'
            write_instance(a,[100,101]);write_instance(b,[101,999,100])
            selected=[dict(trips=[0,1],cost=-100,charging_stops={'cst':[-99]})]
            self.assertEqual(remap_sequences(a,b,selected),[[2,0]])
            write_instance(b,[101,999,100],True)
            with self.assertRaisesRegex(ValueError,'physics changed'):remap_sequences(a,b,selected)
            write_instance(b,[100,100,999])
            with self.assertRaisesRegex(ValueError,'duplicate stable'):remap_sequences(a,b,selected)
            write_instance(b,[100,101,999])
            with self.assertRaisesRegex(ValueError,'omits'):remap_sequences(a,b,[dict(trips=[0])])

    def test_mutated_source_rejected_before_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'solve').mkdir()
            (d/'solve/summary.json').write_text('{}')
            f=d/'prefix.json';f.write_text(json.dumps(dict(summary_sha256='accidental-hash')))
            with self.assertRaisesRegex(ValueError,'mutated predecessor'):
                authenticated_sequences(f,d/'child.csv',d,d/'price',object())

    def test_sealed_handoff_rebuilds_only_sequences(self):
        with tempfile.TemporaryDirectory() as tmp:
            from types import SimpleNamespace
            from station_power import StationPower
            d=Path(tmp);(d/'solve').mkdir()
            src=d/'old.csv';dst=d/'new.csv';write_instance(src,[100,101]);write_instance(dst,[101,999,100])
            shared=[d/x for x in ('price','Ref_dict.csv','par_ref_dhd.csv')]
            for file in shared: file.write_text('toy reference')
            net=SimpleNamespace(plugs=PLUGS,power=StationPower(240,{'PARX':60}),reserve=0,g=240,soc_step=2.5,block_min=5)
            summary=dict(physics=REQUIRED_PHYSICS,plugs=PLUGS,terminal_target_kwh=0,inputs=dict(instance=str(src)),
                         input_hashes={str(p):digest(p) for p in [src,*shared]})
            sp=d/'solve/summary.json';sel=d/'solve/selected_routes.json';vf=d/'verification.json'
            sp.write_text(json.dumps(summary));sel.write_text(json.dumps([{'trips':[0,1],'cost':-87,'charging_stops':{'cst':[-99]}}]))
            val=dict(schema='partille-selection-replay-v1',arm='baseline',cover_valid=True,grid_plug_overload_minutes=0,
                     validated=True,summary_sha256=digest(sp),selected_sha256=digest(sel),instance_sha256=digest(src))
            vf.write_text(json.dumps(val))
            seal=dict(base_solver_commit='3a0ab6a8ab50e4af7d5a7be7688d5ffa1fbbf78c',
                      solver_source_sha256=digest(base/'src/run_capacity_terminal_cg.py'),initializer_sha256=digest(base/'src/prefix_initializer.py'),
                      parent_csv=src.name,receiver_csv=dst.name,source_instance_sha256=digest(src),receiver_instance_sha256=digest(dst),
                      summary_sha256=digest(sp),selected_sha256=digest(sel),validation_sha256=digest(vf))
            f=d/'prefix.json';f.write_text(json.dumps(seal))
            self.assertEqual(authenticated_sequences(f,dst,d,shared[0],net),[[2,0]])
            shared[0].write_text('wrong tariff')
            with self.assertRaisesRegex(ValueError,'shared inputs differ'):authenticated_sequences(f,dst,d,shared[0],net)

    def test_receiver_native_entry_and_legacy_guard(self):
        net=CapacityTerminalNetwork(problem(),prices(),plugs=PLUGS,soc_step=2.5,block_min=5,g_kwh=240,
                                    charge_kw=240,station_kw={'PARX':60},reserve_kwh=0)
        seq=[0,1]
        rebuilt=regenerate(net,[seq])[0]
        self.assertGreater(rebuilt['cost'],100005)
        self.assertTrue(rebuilt['capacity_occupancy'])
        self.assertEqual(rebuilt['capacity_occupancy'],sorted([s,m,n] for (s,m),n in record_occupancy(rebuilt).items() if net.plugs[s] is not None))
        with tempfile.TemporaryDirectory() as tmp:
            opts=dict(target=0,fleet_cap=2,cg_seconds=4,mip_seconds=4)
            with self.assertRaisesRegex(ValueError,'not supported'):
                run(net,Path(tmp)/'unsafe',seeds=[rebuilt],**opts)
            result=run(net,Path(tmp)/'safe',prefix_sequences=[seq],**opts)
            self.assertEqual(result['sparse_timing']['prefix_admitted'],1)
            records=[json.loads(x) for x in (Path(tmp)/'safe/columns.jsonl').read_text().splitlines()]
            self.assertEqual(records[0]['trips'],seq)
            self.assertTrue(any(len(r['trips'])==1 for r in records[1:]))
            self.assertEqual(result['selected_plug_violations'],{})


if __name__=='__main__':
    unittest.main()
