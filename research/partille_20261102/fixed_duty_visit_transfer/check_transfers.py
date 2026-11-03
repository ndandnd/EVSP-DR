"""Replay one same-duty flat-price witness at the prices of each failed fixed-visit cell."""
import argparse
import ast
import csv
import hashlib
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
import sys

TOL = 1e-5
CAPS = {"2190L": 1, "4808": 1, "7880C": 1, "JON_A": 1, "3127L": 2, "PARX": None}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


@lru_cache(None)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def base(node):
    return node.rsplit("_", 1)[0]


def duty_ids(routes):
    return {str(r["duty_id"]): tuple(r["trips"]) for r in routes}


def replay(problem, prices, routes, schedule, physics, target, horizon):
    arcs = {(u, v): (float(d), float(e)) for u, out in problem.adjacency.items() for v, d, e, _ in out}
    trip_count = Counter()
    occupied = Counter()
    energy_ends = []
    checks = set()
    total_cost = 0.0
    require(physics["charge_start_fee"] == 0 and physics["station_kw"] == {"PARX":60.0}, "unexpected charging physics")
    require(len(duty_ids(routes)) == len(routes), "duplicate duty identifier")
    for i, route in enumerate(routes):
        nodes = route["route_nodes"]
        require(base(nodes[0]) == base(nodes[-1]) == "PARX", "depot endpoints")
        require([n for n in nodes if isinstance(n, int)] == route["trips"], "route sequence")
        events = [ (j,s) for j,s in enumerate(schedule) if s["route"]==i and s["kwh"]>0 ]
        clock = 0.0
        soc = float(physics["initial_kwh"])
        for j, (u, v) in enumerate(zip(nodes, nodes[1:]), 1):
            delay, consumption = (0.,0.) if u == v and isinstance(v,str) else arcs[u,v]
            require(delay >= 0 and consumption >= 0, "invalid deadhead")
            clock += delay
            soc -= consumption
            require(soc >= physics["reserve_kwh"] - TOL, "reserve after deadhead")
            if isinstance(v, int):
                require(clock <= problem.start_min[v] + TOL, "late arrival")
                clock = problem.end_min[v]
                trip_count[v] += 1
                soc -= problem.trip_energy[v]
                require(soc >= physics["reserve_kwh"] - TOL, "reserve after passenger service")
            elif j < len(nodes) - 1:
                nxt = nodes[j+1]
                require(isinstance(nxt, int) or j+1 == len(nodes)-1, "adjacent intermediate stations")
                outbound = (0.,0.) if v == nxt else arcs[v,nxt]
                departure_by = (problem.start_min[nxt] if isinstance(nxt,int) else horizon) - outbound[0]
                require(clock <= departure_by + TOL, "cannot leave station in time")
                station = base(v)
                require(station in CAPS, "unknown station")
                hits = [(idx,s) for idx,s in events if idx not in checks and s["station"] == station and
                        clock-TOL <= s["start"] and s["end"] <= departure_by+TOL]
                require(len(hits) <= 1, "ambiguous consecutive sessions")
                for idx,s in hits:
                    start,end = s["start"],s["end"]
                    require(isinstance(start,int) and isinstance(end,int) and 0 <= start < end <= horizon, "session window")
                    rate = physics["station_kw"].get(station,physics["charge_kw"])/60.
                    gain = s["kwh"]/(end-start)
                    require(0 <= gain <= rate + 1e-8, "station power")
                    # Constant delivery within a recorded occupied interval is one feasible reconstruction.
                    for minute in range(start,end):
                        occupied[station,minute] += 1
                        total_cost += gain * prices[station][min(minute//60,max(prices[station]))]
                    clock = end
                    soc += s["kwh"]
                    require(soc <= physics["battery_kwh"]+TOL, "battery overflow")
                    checks.add(idx)
        require(clock <= horizon+TOL, "late depot return")
        energy_ends.append(soc)
    require(all(s["kwh"]>=-TOL for s in schedule), "negative delivered energy")
    require(checks == {j for j,s in enumerate(schedule) if s["kwh"]>0}, "unassigned session")
    require(trip_count == Counter({t:1 for t in problem.trips}), "exactly-once service")
    require(all(CAPS[s] is None or c <= CAPS[s] for (s,_),c in occupied.items()), "plug overload")
    require(sum(energy_ends) >= target-TOL, "terminal floor")
    return {"fleet":len(routes),"trips":len(trip_count),"terminal_kwh":sum(energy_ends),"target_kwh":target,
            "feasible_cost_upper_bound":round(total_cost,6),"sessions":len(checks)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("root","source","out"):
        parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.source))
    from audit_giro_known_columns import build_problem, HORIZON_MIN
    from utils_v2 import load_station_hourly_prices
    from config import CHARGING_STATIONS
    jobs=args.root/"retime_tasks.tsv"
    producer_ast=ast.parse((args.root/"plug_retime.py").read_text())
    plug_specs=[ast.literal_eval(stmt.value) for stmt in producer_ast.body if isinstance(stmt,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="PLUGS" for t in stmt.targets)]
    require(plug_specs==[CAPS],"stored source differs from replay plug counts")
    rows=[r for r in csv.DictReader(jobs.open(),delimiter="\t") if r["arm"]=="fixed_parx60"]
    def native(r):
        return args.root/"results"/r["tag"]/r["cell"]/"fixed_parx60.json"
    donors=[r for r in rows if r["cell"].endswith("_flat") and load(native(r))["plugs"]["sol_count"]>0]
    misses=[r for r in rows if load(native(r))["plugs"]["status"]==3]
    require(len(rows)==104 and len(misses)==12,"source population changed")
    results=[]
    for r in misses:
        result={"cell":r["cell"],"family":r["tag"],"recipient_receipt_sha256":sha(native(r))}
        try:
            f=load(r["fixed_summary"]); original=load(r["routes"]); q=load(native(r))
            matching=[]
            for d in donors:
                if d["tag"]!=r["tag"]: continue
                s=load(d["fixed_summary"]); a=load(native(d))
                if duty_ids(load(d["routes"])) != duty_ids(original): continue
                if (s["inputs"]["instance"],s["inputs"]["data_dir"],a["physics"],a["target"]) == (f["inputs"]["instance"],f["inputs"]["data_dir"],q["physics"],q["target"]):
                    matching.append(d)
            require(len(matching)==1,"expected one matching flat-price sibling")
            d=matching[0]; s=load(d["fixed_summary"]); a=load(native(d)); donor_routes=load(d["routes"])
            require(q["plugs"]["status"]==3 and q["noplugs"]["status"]==3 and q["plugs"]["sol_count"]==0, "source failure")
            require(a["plugs"]["sol_count"]>0 and a["plugs"]["sessions"] and a["plugs"]["status"] in (2,9), "donor result")
            require(q["plugs"] is not None and q["target"] is not None and q.get("physics"),"recipient physics")
            require(Path(a["routes_file"]) == Path(d["routes"]) and Path(q["routes_file"]) == Path(r["routes"]), "input mismatch")
            for summary in (s,f):
                require(all(sha(p)==h for p,h in summary["input_hashes"].items()),"source inputs changed")
            require(duty_ids(f["duties"]) == duty_ids(original) == duty_ids(s["duties"]), "duty sequence mismatch")
            prices=load_station_hourly_prices(Path(f["inputs"]["tariff"]),CHARGING_STATIONS)
            problem=build_problem(Path(f["inputs"]["data_dir"]),str(f["inputs"]["instance"]),max_station_to_trip_wait_min=HORIZON_MIN)
            # The native output reuses "plugs" for the solver result; the six counts are pinned in the source.
            result.update(replay(problem,prices,donor_routes,a["plugs"]["sessions"],q["physics"],q["target"],HORIZON_MIN))
            result.update(verified=True,donor=d["cell"],donor_receipt_sha256=sha(native(d)),donor_routes_sha256=sha(d["routes"]),
                          target_routes_sha256=sha(r["routes"]),instance_sha256=sha(f["inputs"]["instance"]),price_sha256=sha(f["inputs"]["tariff"]))
        except (ValueError,KeyError,AssertionError) as exc:
            result.update(verified=False,reason=str(exc))
        results.append(result)
    out={"task_table_sha256":sha(jobs),"producer_sha256":sha(args.root/"plug_retime.py"),"checker_sha256":sha(__file__),
         "support_sha256":{f:sha(args.source/f) for f in ("audit_giro_known_columns.py","utils_v2.py","config.py")},
         "population":len(rows),"fixed_visit_infeasible":len(misses),"verified":sum(r["verified"] for r in results),"cells":results}
    args.out.write_text(json.dumps(out,indent=2)+"\n")
    print("Verified",out["verified"],"out of",len(misses),Counter(r.get("reason","pass") for r in results))

if __name__=="__main__":
    main()
