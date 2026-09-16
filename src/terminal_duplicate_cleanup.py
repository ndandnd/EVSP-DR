"""Separate postprocessing: remove repeated trips, then reoptimize charging.

Only subsequences of the five selected CG duties are allowed. This is not CG
and imports no GIRO duties. A failed repair does not prove general infeasibility.
"""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

from run_terminal_energy_cg import audit_energy, save, sha
from terminal_energy_pricer import TerminalEnergyNetwork
from validate_terminal_pool_partition import solve
from audit_giro_known_columns import build_problem,HORIZON_MIN
from config import CHARGING_STATIONS,BUS_COST_KX
from utils_v2 import load_station_hourly_prices


def frontier(net, trips):
    trips=tuple(trips)
    if not trips:return []
    current={node:(cost,[(0,node)]) for node,cost in net._iter_sequence_arcs(0,trips[0])}
    for trip in trips[1:]:
        following={}
        for source,(cost,edges) in current.items():
            for target,edge_cost in net._iter_sequence_arcs(source,trip):
                candidate=(cost+edge_cost,edges+[(source,target)])
                if target not in following or candidate<following[target]:
                    following[target]=candidate
        current=following
    candidates=[]
    for node,(cost,edges) in current.items():
        for sink_cost,energy,action in net.terminal_options.get(node,[]):
            candidates.append((cost+sink_cost,energy,edges,action))
    candidates.sort(key=lambda c:(c[0],-c[1]))
    best_energy=float('-inf');records=[]
    for cost,energy,edges,action in candidates:
        if energy<=best_energy:continue
        best_energy=energy
        record=net._record([net._edge_action(s,t) for s,t in edges]+[action])
        assert abs(record['cost']-cost)<1e-5
        record['terminal_energy_kwh']=energy
        records.append(audit_energy(net,record))
    return records


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True);p.add_argument('--source-sha256',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args();source=Path(args.source);assert sha(source)==args.source_sha256
    summary=json.loads(source.read_text());selected_path=source.parent/'selected_routes.json'
    selected=json.loads(selected_path.read_text())
    for path,h in summary['input_hashes'].items():assert sha(path)==h
    paths=[Path(path) for path in summary['input_hashes']]
    instance=next(path for path in paths if 'Practice_Custom' in path.name)
    tariff=next(path for path in paths if path.name.startswith('peak'))
    data=next(path.parent for path in paths if path.name=='Ref_dict.csv')
    problem=build_problem(data,str(instance),max_station_to_trip_wait_min=HORIZON_MIN)
    counts=Counter(t for r in selected for t in r['trips'])
    assert set(counts)==set(problem.trips)
    prices=load_station_hourly_prices(tariff,CHARGING_STATIONS)
    net=TerminalEnergyNetwork(problem,prices,soc_step=2.5,block_min=5,g_kwh=240,charge_kw=350,
                              reserve_kwh=0,strict_tariff_coverage=True,fixed_sequence_index=True)
    routes=[];sequences=set()
    for original in selected:
        duplicated=[t for t in original['trips'] if counts[t]>1]
        assert len(duplicated)<=10,'Require an explicit larger repair design'
        for n in range(len(duplicated)+1):
            for removed in combinations(duplicated,n):
                sequence=tuple(t for t in original['trips'] if t not in removed)
                if sequence and sequence not in sequences:
                    sequences.add(sequence);routes.extend(frontier(net,sequence))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    save(out/'repair_routes.json',routes)
    result,chosen=solve(routes,problem.trips,target=summary['terminal_target_kwh'],
                        cap=summary['fleet_cap'],seconds=3600,log_dir=out,threads=8)
    result.update(source=str(source),source_sha256=sha(source),selected_source_sha256=sha(selected_path),
                  treatment='remove duplicated trips from selected duties; optimize charging frontiers; exact-once selection',
                  generated_sequences=len(sequences),repair_columns=len(routes),
                  individual_replay_verified=bool(chosen),shared_capacity_validated=False,
                  continuous_charging_cost=sum(r['continuous_realized_cost']-BUS_COST_KX for r in chosen) if chosen else None,
                  terminal_kwh=sum(r['terminal_energy_kwh'] for r in chosen) if chosen else None)
    save(out/'selected_routes.json',chosen);save(out/'summary.json',result)


if __name__=='__main__':main()
