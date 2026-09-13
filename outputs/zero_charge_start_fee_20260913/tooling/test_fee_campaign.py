import importlib.util
from pathlib import Path
import unittest
p=Path(__file__).with_name('campaign.py');spec=importlib.util.spec_from_file_location('fee_campaign',p);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class FeeCampaignTests(unittest.TestCase):
    def manifest(self):
        return dict(python='python',code='/cg',mip_code='/mip',common=dict(soc_step_kwh=2.5,block_minutes=5,columns_per_iter=30,rc_epsilon=1e-4,cg_seconds=7200,battery_kwh=240,charge_kw=240,inherit_workers=8,inherit_time_limit_s=0,mip_seconds=3600,stage1_seconds=1800,threads=8),inputs={'case':dict(csv='case.csv',cache='cache.pkl',cache_source_commit='a'*40,parent_descriptor='parent.json')},arms={name:dict(charge_start_cost=fee,inherit_max_columns=0,fixed_sequence_index=True,skip_gurobi_incidence=False) for name,fee in [('fee0',0),('fee5',5)]})
    def test_only_fee_changes_in_commands(self):
        m=self.manifest();pair={'case_id':'case'};root=Path('/same_output')
        for builder in [lambda arm:c.cg_command(m,pair,arm,root),lambda arm:c.mip_command(m,root,arm)]:
            a=builder('fee0');b=builder('fee5');index=a.index('--charge-start-cost')+1
            self.assertEqual(a[index],'0');self.assertEqual(b[index],'5');a[index]=b[index];self.assertEqual(a,b)
    def fixture(self,fee):
        # One charging activity crosses two tariff periods. It incurs one fee, not two.
        return {'physical_replay_validated':True,'selected_routes':[{'expanded_grid_charging_stops':{'cst':[50],'cet':[70],'kwh':[6]},'charging_stops':{'cst':[50],'cet':[70],'kwh':[5]},'continuous_realized_charging_blocks':[{'price_per_kwh':1,'expanded_grid_kwh':3,'realized_kwh':2},{'price_per_kwh':2,'expanded_grid_kwh':3,'realized_kwh':3}],'expanded_grid_cost':100009+fee,'continuous_realized_cost':100008+fee,'physical_realization':{'expanded_grid_terminal_soc_kwh':10,'continuous_terminal_soc_kwh':11}}]}
    def test_two_tariff_blocks_one_start(self):
        x=c.charging_metrics(self.fixture(5),5);self.assertTrue(x['cost_components_reconcile']);self.assertEqual(x['expanded_grid_charging_starts'],1);self.assertEqual(x['expanded_grid_electricity_cost'],9);self.assertEqual(x['continuous_electricity_cost'],8)
    def test_zero_fee_not_replaced_by_default(self):
        x=c.charging_metrics(self.fixture(0),0);self.assertTrue(x['cost_components_reconcile']);self.assertEqual(x['expanded_grid_start_fees'],0)
    def test_wrong_fee_detected(self):
        self.assertFalse(c.charging_metrics(self.fixture(5),0)['cost_components_reconcile'])
    def test_unknown_detail_not_silently_zero(self):
        self.assertFalse(c.charging_metrics({'selected_routes':[{}]},0)['available'])

if __name__=='__main__':unittest.main()
