#!/bin/bash
# Dump every result table on the cluster. Read-only.
H=$HOME/ladder-lite
python3 - "$H" <<'PY'
import csv,glob,json,os,re,sys
H=sys.argv[1]
def tail(p):
    for s in (".iters.csv",".lexicographic.iters.csv"):
        if os.path.exists(p+s):
            r=list(csv.DictReader(open(p+s)))
            if r: return r[-1]
    return None
print("#"*70); print("# CG / LP results by batch"); print("#"*70)
for sub in ("bridge300","warmpool","big240c","big240g","finetime","factfill",
            "cpi_sweep","diversify","b5_check","lexfleet","n6"):
    d=os.path.join(H,sub)
    fs=[p for p in sorted(glob.glob(d+"/*.json"))
        if not any(x in p for x in (".done",".stderr","snapshot","override"))]
    if not fs: continue
    print(f"\n=== {sub} ({len(fs)}) ===")
    for p in fs:
        try: d0=json.load(open(p))
        except Exception: continue
        t=tail(p); g=d0.get
        rw=(t and t.get("route_weight")) or g("phase_2_fleet_lp_bound")
        print(f"  {os.path.basename(p)[:-5]:<34} L={str(rw)[:13]:>13}"
              f" rc={str(t and t.get('min_rc'))[:12]:>12}"
              f" it={str(t and t.get('iteration')):>6}"
              f" h={(float(t['elapsed_s'])/3600 if t else 0):>6.2f}"
              f" {g('stop_reason') or ''}")
print("\n"+"#"*70); print("# arc-flow oracle (discrete-model optima)"); print("#"*70)
for p in sorted(glob.glob(H+"/arcflow/*.json")):
    g=json.load(open(p)).get
    print(f"  {os.path.basename(p)[:-5]:<30}",
          " ".join(f"{k}={g(k)}" for k in
          ("arcflow_lp","lp_bound","integer_fleet","best_integer_fleet",
           "proven_optimal","gap","status","wall_s") if g(k) is not None))
print("\n"+"#"*70); print("# chained: feasibility + finite-pool integer"); print("#"*70)
print(f"  {'cell':<32}{'tgt':>4}{'tf':>14}{'I_pool':>7}{'bound':>8}{'proven':>7}"
      f"{'src_it':>8}  status")
n=at=0
for p in sorted(glob.glob(H+"/chain/mip_*.json")):
    b=os.path.basename(p)[4:-5]
    try: g=json.load(open(p)).get
    except Exception: continue
    tfp=H+"/chain/tf_"+b+".json"
    tf=json.load(open(tfp)).get("outcome") if os.path.exists(tfp) else None
    m=re.match(r'k0*(\d+)_',b); k=int(m.group(1)) if m else None
    hit = (k is not None and g('buses')==k)
    at+= 1 if hit else 0; n+=1
    print(f"  {b:<32}{str(k):>4}{str(tf):>14}{str(g('buses')):>7}"
          f"{str(g('mip_bound'))[:7]:>8}{str(g('fleet_proven')):>7}"
          f"{str(g('source_cg_iterations')):>8}  {g('status_name')}"
          f"{'  <== AT TARGET' if hit else ''}")
print(f"\n  finite-pool results={n}  at_target={at}")
print("  NOTE: 'proven' is scope=finite_pool. Check source_cg_iterations.")
print("  Promote to discrete_model only when ceil(certified L) == validated incumbent.")
print("\n  skipped:",len(glob.glob(H+"/chain/*.skipped")))
PY
echo; echo "=== still running ==="
squeue --me -h -o '%j' | sed 's/_k[0-9].*//' | sort | uniq -c
