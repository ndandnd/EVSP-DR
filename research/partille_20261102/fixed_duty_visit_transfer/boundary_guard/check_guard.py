"""Read-only guard on the depot, native first-visit window, session assignment and duty IDs.

Run with the October retiming directory and its own original source checkout. It reads
all 104 fixed rows; it does not import Gurobi, solve a model, or write to that directory.
"""
from collections import Counter
import argparse,csv,datetime,hashlib,importlib.util,json,math,sys
from pathlib import Path


def require(flag,message):
    if not flag:raise ValueError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_bytes())

def duty_map(rows):
    mapped={str(row['duty_id']):tuple(row['trips']) for row in rows}
    require(len(rows)==len(mapped)==5,'expected five distinct duty IDs');return mapped

def check_route(problem,nodes,trips,charge,arcs,horizon,depot):
    require(nodes[0]==nodes[-1]==depot,'incorrect literal depot')
    require([n for n in nodes if type(n) is int]==trips,'node sequence does not equal trip list')
    require(all(type(n) is int or isinstance(n,str) for n in nodes),'unrecognized route node')
    # Reconstruct visit windows with the native retimer's upper bound on duty time
    # before the first passenger service. Subsequent windows are bracketed by trips.
    def travel(a,b):return 0. if a==b and isinstance(a,str) else arcs[a,b][0]
    windows=[]; leading=0;first_open=None
    for i,node in enumerate(nodes[1:-1],1):
        if type(node) is int:continue
        prev,nxt=nodes[i-1],nodes[i+1]
        if i==1:
            require(type(nxt) is int,'invalid leading charging topology')
            left=problem.start_min[nxt]-60.+travel(prev,node);leading+=1;first_open=left
        else:
            require(type(prev) is int,'unexpected consecutive charging visits')
            left=problem.end_min[prev]+travel(prev,node)
        right=(problem.start_min[nxt] if type(nxt) is int else horizon)-travel(node,nxt)
        base=node.rsplit('_',1)[0]; windows.append((base,math.ceil(left-1e-9),math.floor(right+1e-9),i))
    assigned=set();positive=0;leading_charges=0
    for s in charge:
        require(math.isfinite(s['kwh']) and s['kwh']>=-1.e-8,'invalid energy')
        if s['kwh']<=0:continue
        t0,t1=s['start'],s['end'];require(type(t0) is int and type(t1) is int and 0<=t0<t1<=horizon,'invalid session interval')
        admissible=[i for base,lo,hi,i in windows if base==s['station'] and lo<=t0 and t1<=hi]
        require(len(admissible)==1,'session has zero or multiple matching native windows');v=admissible[0];require(v not in assigned,'two sessions assigned to one visit');assigned.add(v);positive+=1
        if v==1: require(first_open is not None and t0>=math.ceil(first_open-1e-9),'pre-opening charge');leading_charges+=1
    return leading,leading_charges,positive

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--guard-sha256',required=True);p.add_argument('--registered-checker',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.source))
    from audit_giro_known_columns import build_problem,HORIZON_MIN,DEPOT
    from expanded_path_realization import _arc_map
    root=a.root
    batch=(root/'retime.sbatch').read_text();require(str(a.source) in batch,'source path differs from October wrapper')
    jobs=root/'retime_tasks.tsv';rows=[r for r in csv.DictReader(jobs.open(),delimiter='\t') if r['arm']=='fixed_parx60'];require(len(rows)==104,'wrong source population'); cells=[]
    for row in rows:
        native=root/'results'/row['tag']/row['cell']/'fixed_parx60.json';out=read(native);s=read(row['fixed_summary']);routes=read(row['routes']);duty=duty_map(routes);require(duty_map(s['duties'])==duty,'native source duties/route trips disagree');require(all([n for n in r['route_nodes'] if type(n) is int]==r['trips'] for r in routes),'route nodes vs duty trips');require(str(row['routes'])==out['routes_file'],'native route path mismatch');
        require(all(sha(f)==h for f,h in s['input_hashes'].items()),'fixed input hash mismatch')
        require(out['physics']==dict(s['physics'],station_kw={'PARX':60}) and out['target']==s['terminal_target_kwh'],'native physical settings not linked to source');require(out['physics']['coverage']=='exact_once_fixed_duties' and out['physics']['initial_kwh']==out['physics']['battery_kwh']==240 and out['physics']['charge_kw']==350 and out['physics']['reserve_kwh']==0 and out['physics']['charge_start_fee']==0,'unexpected fixed-duty physics')
        key=(row['tag'],s['inputs']['instance'],s['inputs']['data_dir'],out['target'],json.dumps(out['physics'],sort_keys=True),tuple(sorted(duty.items())))
        cells.append((row,out,s,routes,key,native))
    donors=[c for c in cells if c[0]['cell'].endswith('_flat') and c[1]['plugs']['sol_count']>0];misses=[c for c in cells if c[1]['plugs']['status']==3];require(len(misses)==12,'changed miss population'); per=[]; replayed={}; problem_cache={}
    for receiver,native,summary,routes,key,recpath in misses:
        matches=[d for d in donors if d[4]==key]; require(len(matches)==1,'ambiguous or missing predeclared donor'); d,o,s,dr,_,donorpath=matches[0];dk=(d['tag'],d['cell']);
        require({k:v for k,v in s['input_hashes'].items() if k!=s['inputs']['tariff']}=={k:v for k,v in summary['input_hashes'].items() if k!=summary['inputs']['tariff']},'non-price data hashes differ')
        if dk not in replayed:
            instance=summary['inputs']['instance'];problem=problem_cache.get(instance)
            if problem is None:problem=build_problem(Path(summary['inputs']['data_dir']),instance,max_station_to_trip_wait_min=HORIZON_MIN); problem_cache[instance]=problem
            arcs=_arc_map(problem);rr=o['plugs']['sessions'];require(all(type(x['route']) is int and 0<=x['route']<len(dr) for x in rr),'session labels out of range')
            counts=[check_route(problem,r['route_nodes'],r['trips'],[x for x in rr if x['route']==i],arcs,HORIZON_MIN,DEPOT) for i,r in enumerate(dr)];replayed[dk]=dict(duties=len(dr),exact_depot_endpoint_pairs=len(dr),leading_station_visits=sum(c[0] for c in counts),leading_positive_sessions=sum(c[1] for c in counts),positive_sessions_with_unique_native_window=sum(c[2] for c in counts))
        per.append(dict(family=receiver['tag'],cell=receiver['cell'],flat_donor=d['cell'],receiver_native_sha256=sha(recpath),flat_donor_native_sha256=sha(donorpath),receiver_routes_sha256=sha(receiver['routes']),donor_routes_sha256=sha(d['routes']),receiver_fixed_summary_sha256=sha(receiver['fixed_summary']),donor_fixed_summary_sha256=sha(d['fixed_summary']),**replayed[dk]))
    print(json.dumps(dict(verified=True,as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),schema='october-native-visit-boundary-v1',n_source_cells=len(cells),source_cells_with_exact_duty_route_map=len(cells),registered_misses=len(per),unique_matched_donors=len(replayed),unique_donor_counts={k:sum(c[k] for c in replayed.values()) for k in next(iter(replayed.values()))},guard_sha256=a.guard_sha256,registered_checker_sha256=sha(a.registered_checker),wrapper_sha256=sha(root/'retime.sbatch'),retimer_sha256=sha(root/'plug_retime.py'),task_table_sha256=sha(jobs),native_modules={f:sha(a.source/f) for f in ['audit_giro_known_columns.py','expanded_path_realization.py']},cells=per),indent=2))

if __name__=='__main__':main()
