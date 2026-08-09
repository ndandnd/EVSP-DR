#!/bin/bash
# One-stop campaign monitor: job states across ALL families, re-realization
# and injection health, per-cell MIP verdicts (repaired vs pre-repair),
# open-cell listing, live CG trajectories, and error excerpts. Read-only.
#
#   cd ~/EVSP-DR/src && bash monitor_campaigns.sh [days-back, default 3]
set -uo pipefail

DAYS="${1:-3}"
SINCE=$(date -d "-${DAYS} days" '+%Y-%m-%d' 2>/dev/null || date '+%Y-%m-%d')

echo "=== Job families since $SINCE (sacct) ==="
sacct -X -n -P -S "$SINCE" --format=JobName%20,State 2>/dev/null |
  awk -F'|' '{ split($2, s, " "); print $1, s[1] }' | sort | uniq -c |
  sort -k2,2 -k1,1nr

echo
echo "=== Currently in queue ==="
squeue --me -o "%18i %12P %10j %2t %10M %R" 2>/dev/null

echo
echo "=== Re-realization / injection health (all .out files) ==="
OUTS=$(find . -maxdepth 1 -name '*.out' -newermt "$SINCE" 2>/dev/null)
if [ -n "$OUTS" ]; then
  echo "injection attempts:  $(grep -h '^\[MIP\] merged' $OUTS 2>/dev/null | wc -l)"
  echo "rejected routes:     $(grep -h 'REJECTED injected route' $OUTS 2>/dev/null | wc -l)"
  echo "rrz fallbacks:       $(grep -h 're-realization failed' $OUTS 2>/dev/null | wc -l)"
  echo "rrz infeasible seqs: $(grep -h '^\[RRZ\].*INFEASIBLE' $OUTS 2>/dev/null | wc -l)"
  echo
  echo "--- fallback contexts (what broke, 3 lines before each) ---"
  grep -h -B3 're-realization failed' $OUTS 2>/dev/null | tail -24
  echo
  echo "--- provably infeasible sequences (physics findings) ---"
  grep -h '^\[RRZ\].*INFEASIBLE' $OUTS 2>/dev/null | sort | uniq -c | sort -rn | head -8
fi

echo
echo "=== MIP verdicts: repaired (_rrz) vs pre-repair ==="
python3 - <<'PY'
import glob, json, re
def scan(patterns):
    rows = {"closed": [], "open": []}
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            n = f.split("/")[-1]
            m = re.search(r"_k(\d+)_", n)
            k = int(m.group(1)) if m else -1
            b = d.get("buses")
            key = "closed" if isinstance(b, int) and b <= k else "open"
            rows[key].append((k, b, d.get("status_name"), n))
    return rows

rrz = scan(["results/repool_small/*_match_mip_rrz_p*.json",
            "results/exact_big/*_match_mip_rrz_p*.json",
            "results/repool_small/*_cheat_mip_rrz.json",
            "results/tariff_matrix/*_cheat_mip_rrz.json"])
old = scan(["results/repool_small/*_match_mip_p*.json",
            "results/exact_big/*_match_mip_p*.json",
            "results/repool_small/*_cheat_mip.json",
            "results/tariff_matrix/*_cheat_mip.json"])
print(f"repaired cells:   {len(rrz['closed'])} closed / {len(rrz['open'])} open")
print(f"pre-repair cells: {len(old['closed'])} closed / {len(old['open'])} open")
if rrz["open"]:
    print("\nOPEN repaired cells (these are the real frontier):")
    for k, b, s, n in sorted(rrz["open"]):
        print(f"  k={k:<3} buses={b} {s:<12} {n}")
PY

echo
echo "=== Live CG trajectories (last iters.csv line per active run) ==="
for f in $(find results/tariff_big results/exact_big results/tariff_matrix \
             -name '*.iters.csv' -newermt '1 hour ago' 2>/dev/null | sort); do
  printf "%-58s %s\n" "$(basename "$f" .iters.csv)" \
    "$(tail -1 "$f" | awk -F, '{printf "t=%.1fh it=%s obj=%.0f w=%.3f art=%.1f rc=%.0f", $1/3600, $2, $3, $4, $5, $6}')"
done

echo
echo "=== Stall/certificate outcomes in tariff_big ==="
python3 - <<'PY'
import glob, json
for f in sorted(glob.glob("results/tariff_big/*.json")):
    if ".snapshot." in f:
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    fl = d.get("final_lp") or {}
    w = fl.get("route_weight")
    print(f"{f.split('/')[-1]:<52} {d.get('stop_reason', 'running'):<28} "
          f"weight={w if w is None else round(w, 4)}")
PY

echo
echo "=== Error excerpts (files w/ Traceback|FATAL|rc!=0 since $SINCE) ==="
FILES="$OUTS $(find . -maxdepth 1 -name '*.err' -newermt "$SINCE" -size +0 2>/dev/null)"
if [ -n "${FILES// /}" ]; then
  for f in $(grep -El 'Traceback|FATAL|MIP rc=[1-9]' $FILES 2>/dev/null | sort -u); do
    echo "--- $f"
    grep -m2 -n 'Traceback\|FATAL\|MIP rc=[1-9]' "$f"
  done
fi
echo "=== done ==="
