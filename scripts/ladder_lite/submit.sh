#!/bin/bash

main() {
  [ "$#" -ge 1 ] || { echo "usage: submit.sh GROUP [options]" >&2; return 2; }
  GROUP=$1; shift
  case "$GROUP" in PREFLIGHT|SEED|CG|CG_SENSITIVITY|MIP_RAW|MIP_KNOWN) ;; *) echo "invalid group" >&2; return 2;; esac
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || return 1; REPO=$(cd "$SCRIPT_DIR/../.." && pwd) || return 1; LL_ROOT=${LL_ROOT:-"$HOME/ladder-lite"}; PYTHON=${LL_PYTHON:-/home/nc437/evsp_env/bin/python3.12}; CAMPAIGN=${LL_CAMPAIGN:-"ll_$(date -u +%Y%m%d)"}
  PLAN="$LL_ROOT/campaign/$CAMPAIGN/approved-plan.json"; CONC=16; SCALES=""; PART=""; MEM=""; DRY=0
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --scales) SCALES=$2; shift 2;; --concurrency) CONC=$2; shift 2;;
      --partition) PART=$2; shift 2;; --mem) MEM=$2; shift 2;;
      --dry-run) DRY=1; shift;; *) echo "unknown option: $1" >&2; return 2;;
    esac
  done
  [ -s "$PLAN" ] || { echo "missing plan: $PLAN" >&2; return 1; }
  [[ "$CONC" =~ ^[1-9][0-9]*$ ]] || { echo "invalid concurrency" >&2; return 2; }
  mkdir -p "$LL_ROOT/logs" || return 1
  total=0; ids=()
  while IFS=$'\t' read -r BUDGET PLAN_PART THREADS MAX_SCALE INDICES COUNT SCALE_TAG; do
    [ -n "$INDICES" ] || continue
    USE_PART=${PART:-$PLAN_PART}; USE_MEM=$MEM
    if [ -z "$USE_MEM" ]; then
      if [[ "$GROUP" == MIP_* ]] && [ "$MAX_SCALE" -ge 20 ]; then USE_MEM=48G
      elif [[ "$GROUP" == MIP_* ]] || { [[ "$GROUP" == CG* ]] && [ "$MAX_SCALE" -ge 20 ]; }; then USE_MEM=24G
      else USE_MEM=16G; fi
    fi
    WALL=$((BUDGET + 1800)); LIMIT=${LL_MAX_TIME_S:-0}
    if [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] && [ "$WALL" -gt "$LIMIT" ]; then
      echo "WARNING: clamping $GROUP scales=$SCALE_TAG from ${WALL}s to ${LIMIT}s" >&2; WALL=$LIMIT
    fi
    TIME=$(printf '%d:%02d:%02d' $((WALL/3600)) $(((WALL%3600)/60)) $((WALL%60)))
    cmd=(sbatch --requeue --parsable "--array=$INDICES%$CONC" "--partition=$USE_PART"
      -c "$THREADS" "--mem=$USE_MEM" "--time=$TIME" --signal=B:USR1@180 --open-mode=append
      -J "ll_${GROUP}_k${SCALE_TAG//,/x}" -o "$LL_ROOT/logs/ll_${GROUP}_%A_%a.out"
      -e "$LL_ROOT/logs/ll_${GROUP}_%A_%a.err"
      --export="ALL,LL_PYTHON=$PYTHON,LL_REPO=$REPO" "$SCRIPT_DIR/run_cell.sh" "$PLAN" "$GROUP")
    if [ "$DRY" -eq 1 ]; then printf '%q ' "${cmd[@]}"; echo; else
      raw=$("${cmd[@]}"); rc=$?; [ "$rc" -eq 0 ] || { echo "sbatch failed: $raw" >&2; return "$rc"; }
      id=${raw%%;*}; ids+=("$id")
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$GROUP" "$id" "$BUDGET" "$USE_PART" "$USE_MEM" "$COUNT" "$INDICES" >>"$LL_ROOT/submitted.tsv"
    fi
    total=$((total + COUNT))
  done < <("$PYTHON" -B - "$PLAN" "$GROUP" "$SCALES" <<'PY'
import collections,json,sys
p=json.load(open(sys.argv[1])); group=sys.argv[2]; wanted={int(x) for x in sys.argv[3].split(",") if x}
jobs={j["job_key"]:j for j in p["jobs"]}; buckets=collections.defaultdict(list)
for i,key in enumerate(p["task_groups"][group]):
 j=jobs[key]
 if not wanted or int(j["scale"]) in wanted: buckets[(int(j["budget_s"]),j["partition"],int(j["threads"]))].append((i,j))
for (budget,part,threads),rows in sorted(buckets.items()):
 scales=sorted({int(j["scale"]) for _,j in rows})
 print(budget,part,threads,max(scales),",".join(str(i) for i,_ in rows),len(rows),",".join(map(str,scales)),sep="\t")
PY
  ) || return 2
  [ "$total" -gt 0 ] || { echo "no tasks selected" >&2; return 1; }
  [ "$DRY" -eq 1 ] || echo "array_ids=${ids[*]}"
  echo "total_tasks=$total"
}
main "$@"
