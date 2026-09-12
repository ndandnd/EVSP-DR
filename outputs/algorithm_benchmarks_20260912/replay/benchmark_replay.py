#!/usr/bin/env python3
"""Benchmark-only traversal substitution; no production module mutations."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import fcntl, bisect, hashlib, inspect, itertools, json, platform, random, resource, statistics, sys, textwrap, time
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
sys.dont_write_bytecode=True
SOURCE=Path('/private/tmp/evsp-algorithm-review-source-20260912')
assert hashlib.sha256((SOURCE/'src/event_pricer_network.py').read_bytes()).hexdigest()=='d939fe981cb5b3f40293996e8f3a2430da8a365b392a62d6c21959316cb03de6', 'Pinned source mismatch'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE/'src'))
sys.path.insert(0,str(SOURCE/'tests'))
from event_pricer_network import EventExpandedNetwork
from audit_giro_known_columns import build_problem, DEPOT, STATIONS
from utils_v2 import load_station_hourly_prices
from test_event_pricer_network import two_trip_problem, four_trip_chain_problem, prices
BASE=EventExpandedNetwork.fixed_sequence_record
# Keep the complete original dynamic program and record realization verbatim;
# only substitute the two outgoing-arc iteration expressions.
method=textwrap.dedent(inspect.getsource(BASE)).replace('def fixed_sequence_record(', 'def indexed_fixed_sequence_record(')
method=method.replace('self._iter_arcs(0)', 'self._iter_matching(0, trips[0])').replace('self._iter_arcs(source)', 'self._iter_matching(source, successor)')
namespace={}
exec(method,namespace)
class IndexedNetwork(EventExpandedNetwork):
    indexed_fixed_sequence_record=namespace['indexed_fixed_sequence_record']
    def prepare_index(self):
        # O(nodes+arcs) validation, O(trips) storage. No per-arc Python index.
        blocks={}
        for node,meta in enumerate(self.node_meta):
            if meta[0]=='trip': blocks.setdefault(meta[1],[]).append(node)
        self._trip_bounds={}
        for trip,nodes in blocks.items():
            assert nodes==list(range(nodes[0],nodes[-1]+1))
            self._trip_bounds[trip]=(nodes[0],nodes[-1]+1)
        for source in range(len(self.node_meta)):
            if self.arc_mode=='lazy':
                lo,hi=self._arc_slices[source]
                assert all(self._arc_targets[i-1]<=self._arc_targets[i] for i in range(lo+1,hi))
            else:
                rows=self.out[source]
                assert all(rows[i-1][0]<=rows[i][0] for i in range(1,len(rows)))
    def _iter_matching(self,source,trip):
        low,high=(self.SINK,self.SINK+1) if trip is None else self._trip_bounds[trip]
        if self.arc_mode=='lazy':
            start,end=self._arc_slices[source]
            left=bisect.bisect_left(self._arc_targets,low,start,end)
            right=bisect.bisect_left(self._arc_targets,high,left,end)
            if getattr(self,'counting',False):
                self.counts['baseline_arcs_scanned']+=end-start
                self.counts['indexed_arcs_selected']+=right-left
                self.counts['source_queries']+=1
            for offset in range(left,right):
                yield int(self._arc_targets_np[offset]),float(self._arc_costs_np[offset])
        else:
            rows=self.out[source]
            left=bisect.bisect_left(rows,low,key=lambda row:row[0])
            right=bisect.bisect_left(rows,high,lo=left,key=lambda row:row[0])
            if getattr(self,'counting',False):
                self.counts['baseline_arcs_scanned']+=len(rows)
                self.counts['indexed_arcs_selected']+=right-left
                self.counts['source_queries']+=1
            for offset in range(left,right):
                target,cost,_dual,_action=rows[offset]
                yield target,cost
    def _record(self,actions):
        if getattr(self,'capture',False): self.last_actions=deepcopy(actions)
        return super()._record(actions)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(x):
    if isinstance(x,dict):return {str(k):canonical(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [canonical(v) for v in x]
    return x
def digest(x):return hashlib.sha256(json.dumps(canonical(x),sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def reset(n):
    n._window_cache={};n._selected_action_cache={}
def run(n,seqs,kind):
    f=BASE if kind=='baseline' else IndexedNetwork.indexed_fixed_sequence_record
    return [f(n,s) for s in seqs]
def timed(n,seqs,kind,warm=False):
    with open('/private/tmp/evsp-algorithm-benchmarks-20260912.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        reset(n)
        if warm:run(n,seqs,kind)
        t=time.perf_counter(); result=run(n,seqs,kind);elapsed=time.perf_counter()-t
        fcntl.flock(lock,fcntl.LOCK_UN)
    return elapsed,result

def benchmark(name,p,tariff,seqs,mode,step=15):
    physics=dict(soc_step=step,block_min=5,g_kwh=240.,charge_kw=240.,reserve_kwh=0.,strict_tariff_coverage=True)
    with open('/private/tmp/evsp-algorithm-benchmarks-20260912.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        t=time.perf_counter();n=IndexedNetwork(p,tariff,arc_mode=mode,**physics);build=time.perf_counter()-t
        t=time.perf_counter();n.prepare_index();setup=time.perf_counter()-t
        fcntl.flock(lock,fcntl.LOCK_UN)
    n.capture=True; traces={}; results={}
    for kind in ('baseline','indexed'):
        reset(n); records=[]; actions=[]
        for seq in seqs:
            n.last_actions=None
            records.extend(run(n,[seq],kind));actions.append(n.last_actions)
        results[kind]=records;traces[kind]=actions
    assert results['baseline']==results['indexed'],name+' record mismatch'
    assert traces['baseline']==traces['indexed'],name+' action mismatch'
    n.capture=False
    n.counts=dict(baseline_arcs_scanned=0,indexed_arcs_selected=0,source_queries=0);n.counting=True
    run(n,seqs,'indexed');n.counting=False
    samples=[]
    for cache in ('cold_action_window','warm_action_window'):
        for rep in range(6):
            order=('baseline','indexed') if rep%2==0 else ('indexed','baseline')
            for kind in order:
                elapsed,records=timed(n,seqs,kind,warm=cache.startswith('warm'))
                assert records==results['baseline']
                samples.append(dict(cache=cache,repetition=rep,order=list(order),kind=kind,seconds=elapsed))
    med={cache:{kind:statistics.median(s['seconds'] for s in samples if s['cache']==cache and s['kind']==kind) for kind in ('baseline','indexed')} for cache in ('cold_action_window','warm_action_window')}
    result=dict(name=name,mode=mode,trip_count=len(p.trips),physics=physics,initial_energy_kwh=240,terminal_energy='reserve only',shared_capacity=False,objective='existing weighted bus/charge objective unchanged',master_sense='not applicable (fixed sequence pricing)',tariff=tariff,tariff_sha256=digest(tariff),problem_sha256=digest(vars(p) if not hasattr(p,'frame') else {k:v for k,v in vars(p).items() if k!='frame'}),graph=n.metrics(),graph_build_seconds=build,index_setup_seconds=setup,index_storage='O(trips) two integer node bounds; no per-arc additional storage',sequence_count=len(seqs),sequences=[list(s) for s in seqs],sequence_sha256=digest(seqs),feasible=sum(r is not None for r in results['baseline']),rejected=sum(r is None for r in results['baseline']),full_records_sha256=digest(results['baseline']),full_actions_sha256=digest(traces['baseline']),exact_record_and_action_equality=True,counts=n.counts,raw_timing_samples=samples,median_seconds=med,warm_speedup=med['warm_action_window']['baseline']/med['warm_action_window']['indexed'],peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (OUT/(name+'_'+mode+'.json')).write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(name,mode,'arcs',n.n_arcs,'seqs',len(seqs),'feasible',result['feasible'],'build',round(build,3),'warm speedup',round(result['warm_speedup'],3),flush=True)
    return result

def main():
    started=time.time();allresults=[]
    cases=[('tiny_chain',four_trip_chain_problem(),prices(),15),('tiny_tight_soc',two_trip_problem(191.2),prices(),2.5)]
    boundary=two_trip_problem(191.2)
    boundary.end_min[0]=59.75;boundary.start_min[1]=120.;boundary.end_min[1]=130.
    # Equal station alternatives at fractional arrival and exact tariff boundary.
    second=STATIONS[1];first=STATIONS[0]
    boundary.adjacency[0].append((second,0.,0.,'trip_station'))
    boundary.adjacency[second]=list(boundary.adjacency[first])
    tariff=prices()
    for curve in tariff.values():curve.update({0:.3,1:.05,2:.4})
    cases.append(('tiny_tariff_station_ties',boundary,tariff,5))
    for name,p,tariff,step in cases:
        seqs=[()] + [s for length in range(1,5) for s in itertools.product(p.trips,repeat=length)] + [(999999,),(p.trips[0],999999)]
        for mode in ('lazy','explicit'):allresults.append(benchmark(name,p,tariff,seqs,mode,step))
    data=SOURCE/'data';csv=data/'tariff_response/frozen_instances/Practice_Custom_DutyUnion_k05_r2.csv'
    pfull=build_problem(csv.parent,csv.name,reference_data_dir=data)
    pricefile=data/'tariff_response/peak12_alpha_1p0_h26.csv'
    tariff=load_station_hourly_prices(pricefile,STATIONS)
    for size,step in ((12,15),(48,15),(48,5),(48,2.5)):
        chosen=tuple(sorted(pfull.trips,key=lambda t:(pfull.start_min[t],t))[:size]); allowed=set(chosen)|set(STATIONS)|{DEPOT}
        p=SimpleNamespace(trips=chosen,start_min={t:pfull.start_min[t] for t in chosen},end_min={t:pfull.end_min[t] for t in chosen},trip_energy={t:pfull.trip_energy[t] for t in chosen},adjacency={s:[a for a in arcs if a[0] in allowed] for s,arcs in pfull.adjacency.items() if s in allowed})
        rng=random.Random(20260912+size)
        seqs=list(dict.fromkeys([(t,) for t in chosen]+[tuple(sorted(rng.sample(chosen,min(length,len(chosen))),key=lambda t:(p.start_min[t],t))) for length in (2,3,4,6) for _ in range(40)] + [(chosen[-1],chosen[0]),(),(999999,)]))
        for mode in (('lazy','explicit') if step==15 else ('lazy',)):
            r=benchmark('real_first_'+str(size)+'_soc'+str(step),p,tariff,seqs,mode,step);r['source_input_hashes']={str(f):sha(f) for f in (csv,pricefile,data/'Ref_dict.csv',data/'par_ref_dhd.csv')};allresults.append(r)
    summary=dict(source_commit='a29992196acb74d02b8c7891be4061718889999f',source_module_sha256=sha(SOURCE/'src/event_pricer_network.py'),harness_sha256=sha(Path(__file__)),python=sys.version,executable=sys.executable,host=platform.platform(),machine=platform.machine(),cpu_count=os.cpu_count(),thread_environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},elapsed_seconds=time.time()-started,results=allresults)
    (OUT/'results.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
if __name__=='__main__':main()
