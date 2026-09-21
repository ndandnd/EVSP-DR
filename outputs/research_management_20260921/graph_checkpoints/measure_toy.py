"""Local descriptive overhead sample; not a cluster or large-graph benchmark."""
import sys,time,tempfile,json,hashlib,statistics
from pathlib import Path
from types import SimpleNamespace
repo=Path(__file__).resolve().parents[3]/'.codex-work/graph-checkpoints-20260921'
sys.path[:0]=[str(repo/'src'),str(repo/'tests')]
from event_pricer_network import EventExpandedNetwork
from test_event_pricer_network import prices,DEPOT
n=32;trips=tuple(range(n));adj={DEPOT:[(t,0.,0.,'depot_trip') for t in trips]}
for t in trips:adj[t]=[(u,0.,0.,'trip_trip') for u in trips[t+1:]]+[(DEPOT,0.,0.,'trip_depot')]
p=SimpleNamespace(trips=trips,start_min={t:float(45*t) for t in trips},end_min={t:float(45*t+10) for t in trips},trip_energy={t:1. for t in trips},adjacency=adj)
kw=dict(soc_step=2.5,block_min=5,g_kwh=240,charge_kw=240,reserve_kwh=0)
def sha(g):return hashlib.sha256(g._arc_targets.tobytes()+g._arc_costs.tobytes()+g._arc_recipes.tobytes()).hexdigest()
samples=[]
for repeat in range(3):
 with tempfile.TemporaryDirectory() as path:
  values={};fingerprints=[]
  for label,folder in [('fresh',None),('checkpoint_build',path),('completed_resume',path)]:
   t=time.perf_counter();g=EventExpandedNetwork(p,prices(),**kw,arc_checkpoint_dir=folder,arc_checkpoint_identity={'case':'synthetic32'});elapsed=time.perf_counter()-t
   values[label]={'wall_s':elapsed,'checkpoint':getattr(g,'graph_checkpoint_report',None)};fingerprints.append(sha(g))
  assert len(set(fingerprints))==1
  values.update(repeat=repeat,dag_nodes=len(g.node_meta),dag_arcs=g.n_arcs,packed_bytes=g.n_arcs*16,all_buffer_hashes_equal=True)
  samples.append(values)
result={'scope':'local synthetic32trip no-charge graph; 3 repetitions; hardware/cache ordering uncontrolled; not production performance','samples':samples,'medians_s':{label:statistics.median(v[label]['wall_s'] for v in samples) for label in ['fresh','checkpoint_build','completed_resume']}}
Path(__file__).with_name('toy_overhead.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['medians_s']))
