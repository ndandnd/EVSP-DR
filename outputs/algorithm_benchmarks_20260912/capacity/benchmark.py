#!/usr/bin/env python3
"""Benchmark-only capacity selector. Production modules are read-only oracles.

Run: PYTHONDONTWRITEBYTECODE=1 python3 benchmark.py
Iteration contract: begin_iteration snapshots duals and hashes all selector
configuration; do not mutate configuration during the following iteration.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import sys
sys.dont_write_bytecode = True
import argparse, copy, hashlib, json, math, os, platform, random, resource, statistics, subprocess, time
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / '.codex-work/capacity-timeout-checkpoint-20260911'
sys.path.insert(0, str(SOURCE / 'src'))
import event_pricer_network as ep
from utils_v2 import base_station_name
PIN = '253588e9b22d68fcbc67cb56bc3eb30cbb0e16b6'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(obj): return hashlib.sha256(repr(obj).encode()).hexdigest()

class PrefixMemoNetwork(ep.EventExpandedNetwork):
    def begin_iteration(self, duals, sites, grid, *, deadline=None, clock=time.perf_counter):
        ep._check_pricing_deadline(deadline, clock)
        signature = repr((self.events, self.prices, self.station_charge_kw, self.charge_kw, ep.CHARGE_START_COST, grid))
        if getattr(self, '_bench_signature', None) != signature:
            self._bench_options = {}
            self._bench_signature = signature
        self._bench_memo = {}
        self._bench_duals = dict(duals)
        self._bench_sites = None if sites is None else frozenset(sites)
        self._bench_grid = int(grid)
        if self._bench_grid <= 0: raise ValueError('capacity grid must be positive')
        self._bench_prefix = {}
        self.stats = dict(calls=0, charge_calls=0, hits=0, options=0, interval_queries=0, tie_fallbacks=0, options_hits=0)
        by_station = {}
        for (station, minute), value in duals.items():
            ep._check_pricing_deadline(deadline, clock)
            if minute % self._bench_grid == 0:
                by_station.setdefault(station, {})[int(minute // self._bench_grid)] = float(value)
        for station, rows in by_station.items():
            lower, upper = min(rows), max(rows) + 1
            if upper-lower > 100000: raise ValueError('benchmark prefix span safety cap')
            prefix = [0.0]
            for i in range(lower, upper):
                if i % 256 == 0: ep._check_pricing_deadline(deadline, clock)
                prefix.append(prefix[-1] + rows.get(i, 0.0))
            self._bench_prefix[station] = lower, upper, prefix
        # Conservative floating error envelope for both prefix cancellation and
        # the original arbitrary frozenset sum; near ties use the exact oracle.
        total_abs = sum(abs(float(v)) for v in duals.values())
        self._bench_error = 64 * sys.float_info.epsilon * max(1, len(duals)) * max(1., total_abs)

    def interval(self, station, start, end):
        self.stats['interval_queries'] += 1
        if self._bench_sites is not None and station not in self._bench_sites: return 0.0
        info = self._bench_prefix.get(station)
        if info is None: return 0.0
        g = self._bench_grid
        first, last = math.floor(start/g), math.ceil(end/g)
        # Exactly the source predicates, including TOL near both boundaries.
        while first < last and (first+1)*g <= start+ep.TOL: first += 1
        while last > first and (last-1)*g >= end-ep.TOL: last -= 1
        lower, upper, prefix = info
        first, last = max(lower, min(upper, first)), max(lower, min(upper, last))
        return prefix[last-lower]-prefix[first-lower] if last > first else 0.0

    def _options(self, action, deadline, clock):
        station = action['station']
        key = (station, float(action['arrival_min']), float(action['deadline_min']), float(action['kwh']), self._charge_power(station), self._bench_grid)
        if key in self._bench_options:
            self.stats['options_hits'] += 1
            return key, self._bench_options[key]
        arrival, end, energy, power, grid = key[1:]
        duration = energy*60.0/power
        latest = end-duration
        candidates = {arrival, latest}
        for event in self.events[station]:
            ep._check_pricing_deadline(deadline, clock)
            if abs(event/60-round(event/60)) <= ep.TOL:
                candidates.add(float(event)); candidates.add(float(event)-duration)
        for index in range(math.floor(arrival/grid), math.ceil(end/grid)+1):
            if index % 256 == 0: ep._check_pricing_deadline(deadline, clock)
            candidates.add(float(index*grid)); candidates.add(float(index*grid)-duration)
        options = []
        for start in sorted(candidates):
            ep._check_pricing_deadline(deadline, clock)
            if start >= arrival-ep.TOL and start <= latest+ep.TOL:
                options.append((ep._window_cost(station, start, duration, energy, self.prices, power), start, start+duration))
        # Same source ordering; tie fallback therefore observes the same oracle.
        result = tuple(sorted(options))
        self._bench_options[key] = result
        return key, result

    def _capacity_adjusted_arc(self, cost, action, capacity_duals, capacity_sites, capacity_grid_min, *, deadline=None, clock=time.perf_counter):
        ep._check_pricing_deadline(deadline, clock)
        self.stats['calls'] += 1
        if action.get('kind') != 'charge' or not self._bench_duals: return float(cost), action
        self.stats['charge_calls'] += 1
        key = (action['station'], float(action['arrival_min']), float(action['deadline_min']), float(action['kwh']), self._charge_power(action['station']), self._bench_grid)
        if key in self._bench_memo:
            self.stats['hits'] += 1
            adjusted, start, end = self._bench_memo[key]
            return adjusted, {**action, 'cst': start, 'cet': end}
        key, options = self._options(action, deadline, clock)
        best = None
        station = base_station_name(action['station'])
        def exact(option):
            energy_cost, start, end = option
            rows = ep.conservative_capacity_rows({**action, 'cst':start, 'cet':end}, sites=self._bench_sites, grid_min=self._bench_grid)
            return (ep.CHARGE_START_COST+energy_cost-sum(float(self._bench_duals.get(r,0.)) for r in rows), start, end)
        best_option = None
        for option in options:
            ep._check_pricing_deadline(deadline, clock)
            self.stats['options'] += 1
            energy_cost, start, end = option
            candidate = (ep.CHARGE_START_COST+energy_cost-self.interval(station,start,end),start,end)
            if best is not None and abs(candidate[0]-best[0]) <= self._bench_error:
                self.stats['tie_fallbacks'] += 1
                candidate, best = exact(option), exact(best_option)
            if best is None or candidate < best:
                best, best_option = candidate, option
        if best is None: raise RuntimeError('stored event charge arc has no feasible window')
        best = exact(best_option) # returned cost exactly uses original row sum
        self._bench_memo[key] = best
        adjusted, start, end = best
        return adjusted, {**action,'cst':start,'cet':end}

    def min_reduced_cost_route(self, alpha, **kwargs):
        self.begin_iteration(kwargs.get('capacity_duals') or {}, kwargs.get('capacity_sites'), kwargs.get('capacity_grid_min',1), deadline=kwargs.get('deadline'), clock=kwargs.get('clock',time.perf_counter))
        return super().min_reduced_cost_route(alpha, **kwargs)

def fixture():
    s=ep.STATIONS[0]
    return SimpleNamespace(trips=(0,1), start_min={0:0.,1:180.},end_min={0:10.,1:190.},trip_energy={0:190.,1:100.},adjacency={ep.DEPOT:[(0,0.,0.,'depot_trip')],0:[(s,0.,0.,'trip_station'),(ep.DEPOT,0.,0.,'trip_depot')],s:[(1,0.,0.,'station_trip'),(ep.DEPOT,0.,0.,'station_depot')],1:[(ep.DEPOT,0.,0.,'trip_depot')]})

def prices():
    return {base_station_name(s):{h:(.11 if h%3==0 else .17 if h%3==1 else .29) for h in range(27)} for s in ep.STATIONS}

def blank():
    net=PrefixMemoNetwork.__new__(PrefixMemoNetwork)
    net.prices=prices(); net.events={s:tuple(float(i) for i in range(0,1621,5)) for s in ep.STATIONS}
    net.charge_kw=240.; net.station_charge_kw={ep.STATIONS[0]:60.}
    return net

def action(net, station, arrival, duration, slack):
    energy=duration*net._charge_power(station)/60.
    end=arrival+duration+slack
    options=ep._charge_window_options(station,arrival,end,energy,event_times=net.events,station_prices=net.prices,charge_kw=net._charge_power(station))
    cost,start,finish=min(options)
    return ep.CHARGE_START_COST+cost,dict(kind='charge',station=station,arrival_min=arrival,deadline_min=end,kwh=energy,cst=start,cet=finish,extra_metadata='preserved')

def compare(net, items, duals, sites, grid):
    net.begin_iteration(duals, sites, grid)
    error=0.
    for cost,a in items:
        old=ep.EventExpandedNetwork._capacity_adjusted_arc(net,cost,a,duals,sites,grid)
        new=net._capacity_adjusted_arc(cost,a,duals,sites,grid)
        error=max(error,abs(old[0]-new[0]))
        assert math.isclose(old[0],new[0],rel_tol=1e-12,abs_tol=1e-8), (old,new)
        assert old[1]==new[1], (old,new)
    return error

def main():
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==PIN
    rng=random.Random(20260912); net=blank(); base=base_station_name(ep.STATIONS[0]); sites={base}
    cases=[]
    for i in range(120):
        arrival=rng.choice([0.,59.,60.,119.9999999995,60.0000000005,rng.uniform(1,300)])
        duration=rng.choice([.000000002,.25,1.,15.5,59.9999999995,60.,rng.uniform(.2,90)])
        cases.append(action(net,ep.STATIONS[i%len(ep.STATIONS)],arrival,duration,rng.choice([0.,.3,7.,65.])))
    vectors=[{}, {(base,60):0.}, {(base,60):-123.4567}, {(base,i):-rng.uniform(.01,100.) for i in range(0,500) if rng.random()<.1}, {(base,i):-100. for i in range(50,100)}]
    maxerr=0.; checks=0
    for grid in [1,5]:
        for selected_sites in [sites,set(),None]:
            for duals in vectors:
                maxerr=max(maxerr,compare(net,cases,duals,selected_sites,grid)); checks+=len(cases)
    configuration_checks=0
    # Reuse the same object while changing complete selector configuration.
    for change in ['power_alias','tariff','events','dual_in_place']:
        if change=='power_alias': net.station_charge_kw={base:60.}
        if change=='tariff': net.prices[base][1]=.031
        if change=='events': net.events[ep.STATIONS[0]]=tuple(range(0,1621,60))
        changed=dict(vectors[2]); changed[(base,60)]=-7.125
        candidate=[action(net,ep.STATIONS[0],59.25,15.125,80.)]
        compare(net,candidate,changed,sites,1)
        changed[(base,60)]=-314.159
        compare(net,candidate,changed,sites,1)
        configuration_checks+=2
    net=blank()
    # Direct occupied-row boundary oracle, including degenerate and near endpoints.
    occupancy_checks=0; occupancy_error=0.
    net.begin_iteration(vectors[3],None,1)
    for i in range(1000):
        start=rng.choice([60-5e-10,60+5e-10,rng.uniform(-2,500)])
        end=start+rng.choice([0.,1e-10,1e-8,.5,60.,rng.random()*120])
        a=dict(kind='charge',station=ep.STATIONS[0],cst=start,cet=end)
        expected=sum(vectors[3].get(r,0.) for r in ep.conservative_capacity_rows(a))
        observed=net.interval(base,start,end)
        occupancy_error=max(occupancy_error,abs(expected-observed)); assert abs(expected-observed)<1e-8
        occupancy_checks+=1
    # Deadlines checked on cold generation, memo-hit and begin/setup.
    deadline_checks=0
    for mode in ['setup','cold','memo']:
        net.begin_iteration(vectors[2],sites,1)
        c,a=cases[1]
        if mode=='memo': net._capacity_adjusted_arc(c,a,vectors[2],sites,1)
        try:
            if mode=='setup': net.begin_iteration(vectors[2],sites,1,deadline=0.,clock=lambda:1.)
            else: net._capacity_adjusted_arc(c,a,vectors[2],sites,1,deadline=0.,clock=lambda:1.)
        except ep.PricingDeadlineExceeded: deadline_checks+=1
    assert deadline_checks==3
    import fcntl
    timing_lock=open('/private/tmp/evsp-algorithm-benchmarks-20260912.lock','a')
    fcntl.flock(timing_lock,fcntl.LOCK_EX)
    # Bounded synthetic workloads: varying long windows; sparse one-row dual.
    workloads={}
    unique=[action(net,ep.STATIONS[0],10.+i*.125,15.+(i%4)*7.25,180.) for i in range(12)]
    workloads['single_first_use']=unique[:1]
    workloads['unique_12']=unique
    workloads['repeated_120']=unique*10
    measurements=[]
    for name,items in workloads.items():
        for rep in range(6):
            for mode in (['reference','prototype'] if rep%2==0 else ['prototype','reference']):
                worker=blank(); start=time.perf_counter()
                if mode=='prototype': worker.begin_iteration(vectors[2],sites,1)
                for c,a in items:
                    fn=worker._capacity_adjusted_arc if mode=='prototype' else lambda *args: ep.EventExpandedNetwork._capacity_adjusted_arc(worker,*args)
                    fn(c,a,vectors[2],sites,1)
                elapsed=time.perf_counter()-start
                measurements.append(dict(workload=name,rep=rep,mode=mode,regime='cold_one_iteration',seconds=elapsed,stats=worker.stats if mode=='prototype' else None))
    # Repeated iterations retain immutable window/energy options, reset best memo
    # and build prefixes for a changed dual every iteration; all costs timed.
    for rep in range(6):
        for mode in (['reference','prototype'] if rep%2==0 else ['prototype','reference']):
            worker=blank(); start=time.perf_counter(); per=[]
            for duals in [vectors[2],vectors[3],vectors[4]]:
                if mode=='prototype': worker.begin_iteration(duals,sites,1)
                for c,a in workloads['repeated_120']:
                    if mode=='prototype': worker._capacity_adjusted_arc(c,a,duals,sites,1)
                    else: ep.EventExpandedNetwork._capacity_adjusted_arc(worker,c,a,duals,sites,1)
                if mode=='prototype': per.append(dict(worker.stats))
            measurements.append(dict(workload='repeated_120_x3_dual_changes',rep=rep,mode=mode,regime='three_iterations_including_first_cold',seconds=time.perf_counter()-start,stats=per))
    # Real source graph and full route realization on its two-trip test fixture.
    graph_args=dict(soc_step=15,block_min=5,g_kwh=240.,charge_kw=240.,reserve_kwh=0.,arc_mode='explicit',station_charge_kw={ep.STATIONS[0]:60.})
    t=time.perf_counter(); ref=ep.EventExpandedNetwork(fixture(),prices(),**graph_args); build=time.perf_counter()-t
    opt=copy.copy(ref); opt.__class__=PrefixMemoNetwork
    network_checks=[]; network_times=[]
    for rep in range(6):
        duals=vectors[2+rep%3]
        answers={}
        for mode in (['reference','prototype'] if rep%2==0 else ['prototype','reference']):
            worker=ref if mode=='reference' else opt
            start=time.perf_counter(); route=worker.min_reduced_cost_route({0:100000.,1:100000.},capacity_duals=duals,capacity_sites=sites)
            elapsed=time.perf_counter()-start; answers[mode]=route
            network_times.append(dict(rep=rep,mode=mode,seconds=elapsed,stats=dict(worker.stats) if mode=='prototype' else None))
        a,b=answers['reference'],answers['prototype']
        assert a['trips']==b['trips']; assert abs(a['rc']-b['rc'])<1e-8
        assert a['_event_record']==b['_event_record']
        network_checks.append(dict(rep=rep,rc_error=abs(a['rc']-b['rc']),same_full_record=True,trips=a['trips'],rc=a['rc'],physical_status=a['_event_record']['physical_realization']['status']))
    # Cold selector state on each network call, but same prebuilt graph.
    network_cold=[]
    for rep in range(6):
        for mode in (['reference','prototype'] if rep%2==0 else ['prototype','reference']):
            worker=copy.copy(ref)
            if mode=='prototype': worker.__class__=PrefixMemoNetwork
            start=time.perf_counter()
            route=worker.min_reduced_cost_route({0:100000.,1:100000.},capacity_duals=vectors[2],capacity_sites=sites)
            network_cold.append(dict(rep=rep,mode=mode,seconds=time.perf_counter()-start,stats=dict(worker.stats) if mode=='prototype' else None))
    fcntl.flock(timing_lock,fcntl.LOCK_UN)
    result=dict(schema='capacity-prefix-memo-benchmark-v1',source_commit=PIN,source_path=str(SOURCE),source_hashes={str(p.relative_to(SOURCE)):sha(p) for p in sorted((SOURCE/'src').glob('*.py'))},script_sha256=sha(__file__),python=sys.version,platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),cpu_count=os.cpu_count(),hardware=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),hash_seed=os.environ.get('PYTHONHASHSEED','random'),peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_rss_units='bytes on macOS; includes imported modules and all benchmark runs',input_hashes=dict(cases=digest(cases),vectors=digest(vectors),workloads=digest(workloads),fixture=digest(fixture()),prices=digest(prices())),correctness=dict(arc_comparisons=checks,max_adjusted_cost_error=maxerr,exact_action_equality=True,interval_checks=occupancy_checks,max_interval_sum_error=occupancy_error,deadline_checks=deadline_checks,configuration_invalidation_checks=configuration_checks,cost_absolute_tolerance=1e-8,cost_relative_tolerance=1e-12,network=network_checks),measurements=measurements,network_measurements=network_times,network_cold_measurements=network_cold,graph_build_seconds=build,graph_metrics=ref.metrics(),physics=graph_args,objective='combined-cost; source CHARGE_START_COST and tariffs unchanged')
    Path(__file__).with_name('results.json').write_text(json.dumps(result,indent=2)+'\n')
    summary={}
    for workload in workloads.keys()|{'repeated_120_x3_dual_changes'}:
        row={mode:statistics.median(m['seconds'] for m in measurements if m['workload']==workload and m['mode']==mode) for mode in ['reference','prototype']}
        row['speedup']=row['reference']/row['prototype']; summary[workload]=row
    row={mode:statistics.median(m['seconds'] for m in network_times if m['mode']==mode) for mode in ['reference','prototype']}
    row['speedup']=row['reference']/row['prototype']; summary['tiny_network']=row
    summary['tiny_network_cold']={mode:statistics.median(m['seconds'] for m in network_cold if m['mode']==mode) for mode in ['reference','prototype']}
    summary['tiny_network_cold']['speedup']=summary['tiny_network_cold']['reference']/summary['tiny_network_cold']['prototype']
    print(json.dumps(dict(summary=summary,correctness=result['correctness'],graph_metrics=result['graph_metrics']),indent=2))

if __name__=='__main__': main()
