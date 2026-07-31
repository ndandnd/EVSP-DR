"""
Parse MIP results from log files and print a clean labeled table.
Usage: python3 parse_mip_results.py
Run from: ~/demandresponse/src/logs
"""
import re
import subprocess
from pathlib import Path

LOG_DIR  = Path.home() / "demandresponse/src/logs"
BUS_COST = 100_000

# ── 1. Discover all mip_run_*.out files that start with "3h" job names ──────
def get_job_names(log_files):
    """Use sacct to map job_id -> job_name for all mip_run_* logs."""
    ids = []
    for f in log_files:
        m = re.search(r'mip_run_(\d+)', f.stem)
        if m:
            ids.append(m.group(1))
    if not ids:
        return {}

    r = subprocess.run(
        ['sacct', '--jobs=' + ','.join(ids),
         '--format=JobID,JobName%60', '--noheader', '--parsable2'],
        capture_output=True, text=True
    )
    mapping = {}
    for line in r.stdout.strip().split('\n'):
        if '|' not in line:
            continue
        jid, jname = line.split('|', 1)
        jid = jid.strip()
        if '.' not in jid and '_' not in jid:   # top-level job only
            mapping[jid] = jname.strip()
    return mapping

# ── 2. Parse a single log file for Gurobi MIP results ───────────────────────
def parse_log(path):
    text = path.read_text(errors='replace')
    err  = path.with_suffix('.err')

    if err.exists() and 'Traceback' in err.read_text(errors='replace'):
        return {'status': 'CRASHED'}

    # Gurobi summary line: "Best objective X, best bound Y, gap Z%"
    m = re.search(
        r'Best objective\s+([\d.e+\-]+),\s*best bound\s+([\d.e+\-]+),\s*gap\s+([\d.]+)%',
        text, re.IGNORECASE
    )
    if m:
        return {
            'status': 'DONE',
            'obj':    float(m.group(1)),
            'bound':  float(m.group(2)),
            'gap':    float(m.group(3)),
        }

    # Still running
    # Grab the latest incumbent if available
    inc = re.findall(r'^\s*\w\s+([\d.e+\-]+)\s+[\d.e+\-]+\s+[\d.e+\-]+\s+\d+s', text, re.MULTILINE)
    return {'status': 'RUNNING', 'obj': float(inc[-1]) if inc else None}

# ── 3. Parse instance label from job name ────────────────────────────────────
def parse_instance(job_name):
    """
    Expected formats:
      3h_RND001_CH        -> (10B, RND001, CH)
      3h_RND002_NC        -> (10B, RND002, NC)
      3h_15B_RND001_CH   -> (15B, RND001, CH)
      3h_15B_RND003_NC   -> (15B, RND003, NC)
    Returns None if not a 3h job.
    """
    if not job_name.startswith('3h_'):
        return None
    m = re.match(r'3h_(?:(15B|20B|30B)_)?RND(\d+)_(CH|NC)$', job_name, re.IGNORECASE)
    if not m:
        return None
    size = m.group(1) or '10B'
    rnd  = f'RND{int(m.group(2)):03d}'
    mode = m.group(3).upper()
    return (size, rnd, mode)

# ── 4. Main ──────────────────────────────────────────────────────────────────
log_files = sorted(LOG_DIR.glob('mip_run_*.out'))
job_names = get_job_names(log_files)

# Collect best result per (size, rnd, mode) — prefer lowest gap, then lowest obj
best = {}   # key -> {'status', 'obj', 'bound', 'gap', 'job_id'}
for f in log_files:
    m = re.search(r'mip_run_(\d+)', f.stem)
    if not m:
        continue
    job_id   = m.group(1)
    job_name = job_names.get(job_id, '')
    inst     = parse_instance(job_name)
    if inst is None:
        continue

    parsed = parse_log(f)
    parsed['job_id'] = job_id

    key = inst
    if key not in best:
        best[key] = parsed
    elif parsed['status'] == 'DONE':
        existing = best[key]
        if existing['status'] != 'DONE':
            best[key] = parsed
        elif parsed.get('gap', 999) < existing.get('gap', 999):
            best[key] = parsed

# ── 5. Print table ───────────────────────────────────────────────────────────
def flag(r):
    if r['status'] == 'CRASHED':  return '✗ CRASH'
    if r['status'] == 'RUNNING':  return '🔄 RUN  '
    gap = r.get('gap', 999)
    if gap <= 0.01:  return '✓ OPT  '
    if gap <  5.0:   return '~ <5%  '
    return               '! GAP  '

sizes = sorted(set(s for (s,_,_) in best))
for size in sizes:
    n     = int(size.replace('B',''))
    floor = n * BUS_COST
    print()
    print(f"{'─'*70}")
    print(f"  {size}  (fleet target = {n} buses, cost floor = {floor:,.0f})")
    print(f"{'─'*70}")
    print(f"  {'Instance':<18} {'Mode':<9} {'MIP Obj':>13} {'LP Bound':>13}"
          f"  {'Gap':>6}  {'Buses':>5}  {'Ovhd':>6}  Status")
    print(f"  {'─'*16} {'─'*8} {'─'*13} {'─'*13}  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*7}")

    rnds = sorted(set(rnd for (s,rnd,_) in best if s == size))
    for rnd in rnds:
        for mode in ['CH', 'NC']:
            r = best.get((size, rnd, mode))
            label = 'CHEAT' if mode == 'CH' else 'NO_CHEAT'
            if r is None:
                print(f"  {rnd:<18} {label:<9} {'—':>13} {'—':>13}  {'—':>6}  {'—':>5}  {'—':>6}  (not run)")
                continue
            if r['status'] in ('CRASHED', 'RUNNING'):
                obj_s = f"{r['obj']:>13,.0f}" if r.get('obj') else f"{'—':>13}"
                print(f"  {rnd:<18} {label:<9} {obj_s} {'—':>13}  {'—':>6}  {'—':>5}  {'—':>6}  {flag(r)}")
                continue
            obj   = r['obj']
            bound = r['bound']
            gap   = r['gap']
            buses = obj / BUS_COST
            ovhd  = (obj - floor) / floor * 100
            print(f"  {rnd:<18} {label:<9} {obj:>13,.0f} {bound:>13,.0f}"
                  f"  {gap:>5.2f}%  {buses:>4.1f}  +{ovhd:>4.1f}%  {flag(r)}")

print()
print("  Legend:  ✓ OPT gap≤0.01%  ~ <5%  ! GAP≥5%  🔄 still running  ✗ crashed")
print("  Ovhd = charging cost above N×100k bus-cost floor")
print("  Uses best result (lowest gap) when a run appears multiple times.")
