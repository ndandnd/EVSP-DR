import itertools,math,random,unittest
from fractions import Fraction
from core import structure,dual_certificate,validate_start


def feasible(columns,m,cap=None):
    for bits in itertools.product([0,1],repeat=len(columns)):
        selected=[i for i,b in enumerate(bits) if b]
        try:validate_start(columns,m,selected,cap)
        except ValueError:continue
        yield bits


class CoreTests(unittest.TestCase):
    def test_reject_invalid_starts(self):
        for selected in [[],[0,0],[8],[0]]:
            with self.assertRaises(ValueError):validate_start([[0],[1]],2,selected)
        with self.assertRaises(ValueError):validate_start([[0],[1]],2,[0,1],1)

    def test_dominance_and_row_direction(self):
        cols=[[0],[0,1],[1,2],[2]];cost=[3,2,4,1]
        d=structure(cols,cost,3)
        self.assertIn([0,1],d['safe_cost_respecting_column_witnesses'])
        self.assertEqual(d['components'],[{'rows':3,'columns':4}])
        r=structure([[0,1],[1]], [1,2],2)
        self.assertIn([1,0],r['redundant_row_witnesses'])
        self.assertNotIn([0,1],r['redundant_row_witnesses'])

    def test_identical_ties_and_components(self):
        d=structure([[0],[0],[1,2],[1],[2]],[1,1,2,3,3],3)
        self.assertEqual(d['identical_incidence_pairs'],[[1,0]])
        self.assertIn([1,0],d['safe_cost_respecting_column_witnesses'])
        self.assertEqual(sorted(c['rows'] for c in d['components']),[1,2])

    def test_safe_dominance_preserves_optimum_and_fleet_cap(self):
        rng=random.Random(24)
        for _ in range(60):
            m=3;cols=[[i] for i in range(m)]+[rng.sample(range(m),rng.randint(1,m)) for _ in range(4)];cost=[rng.randint(0,7) for c in cols];d=structure(cols,cost,m);drop={j for j,k in d['safe_cost_respecting_column_witnesses']}
            for objective in [cost,[1]*len(cols)]:
                for cap in [None,2,3]:
                    vals=[sum(v*c for v,c in zip(b,objective)) for b in feasible(cols,m,cap)];vals2=[sum(v*c for v,c in zip(b,objective)) for b in feasible(cols,m,cap) if not any(b[j] for j in drop)]
                    self.assertEqual(min(vals,default=math.inf),min(vals2,default=math.inf))

    def test_dual_upper_bound_contribution_required(self):
        # Invalid naive unbounded-variable dual: y=10 gives q=-9; bound term
        # restores L=1. Forcing x=0 gives lower bound10 (infeasible cover).
        d=dual_certificate([[0]],[1],[10],0,None,[0]);self.assertEqual(d['lower_bound'],1);self.assertEqual(d['safe_fix_one_indices'],[0]);self.assertEqual(d['safe_fix_zero_indices'],[])

    def test_exhaustive_bounds_aware_certificates(self):
        rng=random.Random(931)
        for _ in range(80):
            m=3;cols=[[i] for i in range(m)]+[[0,1],[1,2],[0,2]];cost=[rng.uniform(-1,5) for c in cols];y=[rng.uniform(-2,8) for i in range(m)];mu=rng.uniform(-5,2);cap=rng.choice([None,2,3]);witness=[3,4]
            d=dual_certificate(cols,cost,y,mu,cap,witness,scale=10**6);S=d['scale'];upper=Fraction(d['incumbent_upper_bound_integer'],S)
            for b in feasible(cols,m,cap):
                obj=sum((Fraction.from_float(c) for c,x in zip(cost,b) if x),Fraction(0));self.assertLessEqual(Fraction(d['lower_bound_integer'],S),obj)
                for j,x in enumerate(b):
                    forced=d['forced_one_lower_bounds_integer'][j] if x else d['forced_zero_lower_bounds_integer'][j];self.assertLessEqual(Fraction(forced,S),obj)
                if obj<=upper:
                    self.assertFalse(any(b[j] for j in d['safe_fix_zero_indices']));self.assertTrue(all(b[j] for j in d['safe_fix_one_indices']))

    def test_strict_screen_retains_equal_objective(self):
        d=dual_certificate([[0],[0]],[1,1],[1],0,None,[0]);self.assertEqual(d['safe_fix_zero_indices'],[])

    def test_negative_cost_disables_simple_dominance(self):
        with self.assertRaises(ValueError):structure([[0]],[-1],1)

    def test_timeout_diagnostic_is_explicit(self):
        d=structure([[0],[1]],[1,1],2,max_seconds=-1);self.assertFalse(d['dominance_complete']);self.assertEqual(d['dominance_processed_columns'],0)

if __name__=='__main__':unittest.main(verbosity=2)

class LaunchTests(unittest.TestCase):
    def test_case_specific_dependencies_and_resources(self):
        from launch import command,ARMS
        from pathlib import Path
        self.assertEqual(len(ARMS),5)
        for case,job in [('c1_k08_fresh','12'),('c4_k08_fresh','34')]:
            self.assertFalse(any('dependency' in s for s in command(Path('/example'),case,'prepare')))
            for arm in ARMS:
                c=command(Path('/example'),case,'trial',job,arm)
                self.assertIn('--dependency=afterok:'+job,c)
                self.assertIn('--exclude=scaglione-compute-01',c)
                self.assertIn('--partition=default_partition',c)
                self.assertIn('--cpus-per-task=8',c)
                self.assertIn('--mem=32G',c)
                self.assertFalse(any('--array' in s for s in c))

class IdempotenceTests(unittest.TestCase):
    def test_second_launch_submits_nothing_and_uncertain_intent_blocks(self):
        import tempfile,json,io,contextlib
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        import launch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'manifest.json').write_text(json.dumps({'cases':[{'id':'a'},{'id':'b'}]}));calls=[]
            def fake(cmd,**kwargs):
                calls.append(cmd);return SimpleNamespace(stdout=str(1000+len(calls))+'\n',stderr='')
            with patch('sys.argv',['launch',str(root)]),patch('launch.subprocess.run',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):
                launch.main();launch.main()
            self.assertEqual(len(calls),12)
            self.assertTrue(all('--dependency=afterok:1001' in c for c in calls[1:6]))
            self.assertTrue(all('--dependency=afterok:1007' in c for c in calls[7:12]))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'manifest.json').write_text(json.dumps({'cases':[{'id':'a'}]}));(root/'a__prepare_SUBMIT_INTENT.json').write_text('{}')
            with patch('sys.argv',['launch',str(root)]),patch('launch.subprocess.run') as submit:
                with self.assertRaises(RuntimeError):launch.main()
                submit.assert_not_called()
