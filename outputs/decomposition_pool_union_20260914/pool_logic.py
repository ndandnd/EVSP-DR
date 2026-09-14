#!/usr/bin/env python3
"""Build hash-bound parent-index pools without constructing a parent graph."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, resource, sqlite3, time
from pathlib import Path

SOURCE_COMMIT = "a29992196acb74d02b8c7891be4061718889999f"

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def atomic_json(p,v):
    p=Path(p); t=p.with_name("."+p.name+f".tmp.{os.getpid()}")
    with t.open("x") as f:
        json.dump(v,f,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    t.replace(p)
def require(p,h):
    got=sha(p)
    if got!=h: raise ValueError(f"hash mismatch {p}: {got} != {h}")

def trip_maps(path):
    rows=list(csv.DictReader(Path(path).open(newline="")))
    local_to_real={}; real_to_local={}; physical={}
    for row in rows:
        local=int(row["count_trip_id"]); real=int(row["Ordered_Trip_ID"])
        if local in local_to_real or real in real_to_local: raise ValueError(f"nonunique trip identity in {path}")
        local_to_real[local]=real; real_to_local[real]=local
        physical[real]={k:v for k,v in row.items() if k!="count_trip_id"}
    return local_to_real,real_to_local,physical

def remap_record(record, local_to_parent):
    out=dict(record)
    trips=record.get("trips")
    if not trips or len(trips)!=len(set(trips)) or any(type(x) is not int for x in trips):
        raise ValueError("invalid trips")
    try: mapped=[local_to_parent[x] for x in trips]
    except KeyError as e: raise ValueError(f"unknown local trip {e.args[0]}")
    nodes=record.get("route_nodes")
    if not isinstance(nodes,list): raise ValueError("missing route_nodes")
    ints=[x for x in nodes if type(x) is int]
    if ints != trips: raise ValueError("route_nodes integer sequence differs from trips")
    out["trips"]=mapped
    out["route_nodes"]=[local_to_parent[x] if type(x) is int else x for x in nodes]
    if [x for x in out["route_nodes"] if type(x) is int] != mapped: raise AssertionError("remap failed")
    # These hashes are part of the preserved realization payload but bind the
    # child-local IDs. Rebind exactly the ID-dependent fields after remapping.
    canonical=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if isinstance(out.get("continuous_realization"),dict):
        cr=dict(out["continuous_realization"]);cr["trip_sequence_sha256"]=canonical(mapped);cr["route_nodes_sha256"]=canonical(out["route_nodes"])
        cr["mapping_sha256"]=canonical({k:v for k,v in cr.items() if k not in ("trace","mapping_sha256")});out["continuous_realization"]=cr
        if isinstance(out.get("physical_realization"),dict):
            pr=dict(out["physical_realization"]);pr["realization_mapping_sha256"]=cr["mapping_sha256"];out["physical_realization"]=pr
    return out

def validate_source(source,parent_real_to_local,parent_physical,spec):
    for pk,hk in (("input_path","input_sha256"),("status_path","status_sha256"),("mip_path","mip_sha256")):
        require(source[pk],source[hk])
    status=read(source["status_path"]); mip=read(source["mip_path"])
    if status.get("provenance",{}).get("git_commit")!=SOURCE_COMMIT: raise ValueError("source commit")
    if status.get("provenance",{}).get("instance_sha256")!=source["input_sha256"]: raise ValueError("source input provenance")
    if (Path(spec["data_dir"])/status.get("csv","")).resolve()!=Path(source["input_path"]).resolve(): raise ValueError("source CSV path binding")
    expected={"g_kwh":240,"charge_kw":240,"min_soc_frac":0,"soc_step":2.5,"block_min":5,"time_model":"event","master_sense":"cover","prices_csv":"hourly_prices_flat.csv"}
    for k,v in expected.items():
        if status.get(k)!=v: raise ValueError(f"source model mismatch {source['case']} {k}")
    for k in ("capacity_enforced","shared_station_capacity","terminal_soc_floor"):
        if status.get(k) not in (None,False,{},[]): raise ValueError(f"unsupported source constraint {source['case']} {k}")
    for k,v in spec["model_hashes"].items():
        if status.get("provenance",{}).get(k)!=v: raise ValueError(f"source provenance mismatch {source['case']} {k}")
    if mip.get("source_result_sha256")!=source["status_sha256"] or mip.get("source_journal_sha256")!=source["journal_sha256"]: raise ValueError("source MIP binding")
    audit=mip.get("physical_pool_audit",{})
    if not mip.get("physical_replay_validated") or audit.get("rejected_columns")!=0 or audit.get("deterministically_repaired")!=0: raise ValueError("source physical admission")
    child_local_to_real,_,child_physical=trip_maps(source["input_path"])
    if status.get("trip_ids")!=list(child_local_to_real): raise ValueError("source trip_ids differ from hashed CSV")
    for real,row in child_physical.items():
        if parent_physical.get(real)!=row: raise ValueError(f"physical trip attribute mismatch {source['case']} Ordered_Trip_ID={real}")
    local_to_parent={k:parent_real_to_local[v] for k,v in child_local_to_real.items()}
    mandatory={json.dumps(sorted(r["trips"]),separators=(",",":")) for r in mip["selected_routes"]}
    return status,mip,local_to_parent,mandatory,set(child_local_to_real.values())

def construct_partition(spec,out):
    started=time.monotonic(); out=Path(out); journal=Path(str(out)+".columns.jsonl"); dbpath=Path(str(out)+".sqlite")
    require(spec["parent_input_path"],spec["parent_input_sha256"])
    parent_l2r,parent_r2l,parent_physical=trip_maps(spec["parent_input_path"])
    if set(parent_l2r)!=set(range(len(parent_l2r))): raise ValueError("parent indices not contiguous")
    db=sqlite3.connect(dbpath); db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; CREATE TABLE cols(component INT, tripkey TEXT, n INT, ratio REAL, cost REAL, mandatory INT, payload TEXT, PRIMARY KEY(component,tripkey));")
    source_audits=[]; component_reals=[]
    try:
      for source in spec["sources"]:
        status,mip,mapping,mandatory,reals=validate_source(source,parent_r2l,parent_physical,spec); component_reals.append(reals)
        h=hashlib.sha256(); seen=0; matched=set()
        with Path(source["journal_path"]).open("rb") as raw:
          for lineb in raw:
            h.update(lineb)
            if not lineb.strip(): continue
            record=json.loads(lineb); localkey=json.dumps(sorted(record["trips"]),separators=(",",":"))
            mapped=remap_record(record,mapping); key=json.dumps(sorted(mapped["trips"]),separators=(",",":")); cost=float(mapped["cost"])
            if not math.isfinite(cost): raise ValueError("nonfinite cost")
            ismandatory=localkey in mandatory
            if ismandatory: matched.add(localkey)
            payload=json.dumps(mapped,sort_keys=True,separators=(",",":"),allow_nan=False)
            old=db.execute("SELECT cost,mandatory FROM cols WHERE component=? AND tripkey=?",(source["component"],key)).fetchone()
            if old is None:
              db.execute("INSERT INTO cols VALUES(?,?,?,?,?,?,?)",(source["component"],key,len(mapped["trips"]),cost/len(mapped["trips"]),cost,int(ismandatory),payload))
            elif cost < old[0]-1e-9:
              db.execute("UPDATE cols SET n=?,ratio=?,cost=?,mandatory=?,payload=? WHERE component=? AND tripkey=?",(len(mapped["trips"]),cost/len(mapped["trips"]),cost,int(ismandatory or old[1]),payload,source["component"],key))
            elif ismandatory and not old[1]: db.execute("UPDATE cols SET mandatory=1 WHERE component=? AND tripkey=?",(source["component"],key))
            seen+=1
            if seen%5000==0: db.commit()
        db.commit()
        if h.hexdigest()!=source["journal_sha256"]: raise ValueError(f"journal hash mismatch {source['case']}")
        if matched!=mandatory: raise ValueError(f"mandatory routes absent {source['case']}")
        source_audits.append({"case":source["case"],"records":seen,"mandatory":len(mandatory),"journal_sha256":h.hexdigest(),"source_cg_certified":bool(source["certified"]),"source_stop_reason":source["source_stop_reason"]})
      union=set()
      for s in component_reals:
        if union&s: raise ValueError("component real-trip overlap")
        union|=s
      if union!=set(parent_r2l): raise ValueError(f"components do not partition parent: {len(union)} != {len(parent_r2l)}")
      selected=[]; component_audits=[]
      for component in range(4):
        mandatory_n=db.execute("SELECT COUNT(*) FROM cols WHERE component=? AND mandatory=1",(component,)).fetchone()[0]
        target=max(spec["per_component_cap"],mandatory_n); exceeded=mandatory_n>spec["per_component_cap"]
        rows=db.execute("SELECT payload,mandatory FROM cols WHERE component=? ORDER BY mandatory DESC,n DESC,ratio ASC,tripkey ASC LIMIT ?",(component,target)).fetchall()
        selected += [json.loads(x[0]) for x in rows]
        component_audits.append({"component":component,"mandatory":mandatory_n,"selected":len(rows),"requested_cap":spec["per_component_cap"],"mandatory_exceeded_cap":exceeded})
      with journal.open("x") as f:
        for r in selected: f.write(json.dumps(r,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
        f.flush(); os.fsync(f.fileno())
      start_routes=[]
      for (payload,) in db.execute("SELECT payload FROM cols WHERE mandatory=1 ORDER BY component,tripkey"):
        r=json.loads(payload); warm=dict(r); warm["route"]=warm.pop("route_nodes")
        warm.setdefault("master_cost_semantics","expanded_grid_cost");warm.setdefault("expanded_grid_cost",warm["cost"])
        start_routes.append(warm)
      start_path=out.parent/"mandatory_routes.json"; atomic_json(start_path,{"routes":start_routes})
      covered=[x for r in start_routes for x in r["route"] if type(x) is int]
      if set(covered)!=set(parent_l2r): raise ValueError("mandatory start does not cover every parent trip")
      construction={"schema":"evsp-parent-mapped-pool-construction-v1","kind":"pool_construction","artifact_kind":"parent_mapped_partition_pool","optimization_run":False,"certified":False,"full_model_lp_certified":False,"parent_graph_constructed":False,"partition":spec["partition"],"selection_policy":"all source-MIP selected route trip sets (cheapest same-trip-set payload retained), then longest route, cheapest cost/trip, stable parent trip IDs","per_component_cap":spec["per_component_cap"],"component_audits":component_audits,"source_audits":source_audits,"columns":len(selected),"mandatory_route_count":len(start_routes),"mandatory_total_expanded_grid_cost":sum(float(r["cost"]) for r in start_routes),"mandatory_covers_every_parent_trip":True,"mandatory_overcovered_trip_visits":len(covered)-len(parent_l2r),"journal_sha256":sha(journal),"mandatory_routes_sha256":sha(start_path),"wall_s":time.monotonic()-started,"peak_rss_native":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
      status={"schema":"evsp-parent-mapped-pool-v1","artifact_kind":"parent_mapped_partition_pool","optimization_run":False,"certified_rc_optimal":False,"full_model_lp_certified":False,"stop_reason":"data_only_pool_construction_no_cg","csv":spec["parent_csv"],"prices_csv":"hourly_prices_flat.csv","soc_step":2.5,"block_min":5,"g_kwh":240,"charge_kw":240,"min_soc_frac":0,"master_sense":"cover","time_model":"event","trip_ids":list(range(len(parent_l2r))),"columns_journal":str(journal),"final":{"artificials":0,"iter":0,"pool_columns":len(selected)},"provenance":{"instance_sha256":spec["parent_input_sha256"],**spec["model_hashes"],"git_commit":SOURCE_COMMIT,"scope":"source model identity only; construction has no pricing certificate or LP bound"},"pool_construction":construction}
      atomic_json(out,status); atomic_json(out.parent/"construction.json",construction)
      return construction
    finally:
      db.close()
      if dbpath.exists(): dbpath.unlink()

def union_pools(spec,out):
    """Make a private, full-payload union for one solver attempt."""
    out=Path(out); journal=Path(str(out)+".columns.jsonl"); statuses=[]; best={}
    for source in spec["sources"]:
        require(source["pool_path"],source["pool_sha256"]); require(source["journal_path"],source["journal_sha256"])
        s=read(source["pool_path"])
        if s["artifact_kind"]!="parent_mapped_partition_pool" or s["certified_rc_optimal"] is not False: raise ValueError("invalid parent pool")
        if statuses:
            first=statuses[0]
            for k in ("csv","prices_csv","soc_step","block_min","g_kwh","charge_kw","min_soc_frac","master_sense","time_model","trip_ids"):
                if s.get(k)!=first.get(k): raise ValueError(f"union model identity mismatch {k}")
            for k in ("instance_sha256","prices_sha256","reference_sha256","deadhead_sha256","git_commit"):
                if s.get("provenance",{}).get(k)!=first.get("provenance",{}).get(k): raise ValueError(f"union provenance mismatch {k}")
        statuses.append(s)
        with Path(source["journal_path"]).open() as f:
            for line in f:
                r=json.loads(line); key=tuple(sorted(r["trips"])); old=best.get(key)
                if old is None or float(r["cost"])<float(old["cost"])-1e-9: best[key]=r
    first=statuses[0]
    with journal.open("x") as f:
        for key in sorted(best): f.write(json.dumps(best[key],sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
        f.flush(); os.fsync(f.fileno())
    status={k:first[k] for k in ("csv","prices_csv","soc_step","block_min","g_kwh","charge_kw","min_soc_frac","master_sense","time_model","trip_ids")}
    status.update({"schema":"evsp-parent-mapped-union-v1","artifact_kind":"parent_mapped_pool_union","optimization_run":False,"certified_rc_optimal":False,"full_model_lp_certified":False,"stop_reason":"data_only_pool_union_no_cg","columns_journal":str(journal),"final":{"artificials":0,"iter":0,"pool_columns":len(best)},"provenance":dict(first["provenance"]),"pool_construction":{"kind":"solver_private_union","source_partitions":spec["partitions"],"source_hashes":spec["sources"],"union_journal_sha256":sha(journal),"parent_graph_constructed":False}})
    atomic_json(out,status); return status

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=("construct","union")); p.add_argument("--spec",required=True); p.add_argument("--out",required=True); a=p.parse_args()
    (construct_partition if a.mode=="construct" else union_pools)(read(a.spec),a.out)
