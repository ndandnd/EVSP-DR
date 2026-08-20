#!/bin/bash

main() {
  LL_ROOT=${LL_ROOT:-"$HOME/ladder-lite"}; PYTHON=${LL_PYTHON:-/home/nc437/evsp_env/bin/python3.12}
  PLAN="$LL_ROOT/campaign/approved-plan.json"; FILTER=${1:-}
  [ -s "$PLAN" ] || { echo "missing plan: $PLAN" >&2; return 1; }
  QUEUE=$(squeue -r -h -u "${USER:-$(id -un)}" -o '%i|%j|%P|%t|%M|%L|%R' 2>/dev/null)
  export LL_QUEUE="$QUEUE"
  "$PYTHON" -B - "$PLAN" "$FILTER" <<'PY' || return 1
import json,os,pathlib,sys
p=json.load(open(sys.argv[1])); filt=sys.argv[2]
jobs={j["job_key"]:j for j in p["jobs"]}
names=[f.split("|")[1] for f in os.environ.get("LL_QUEUE","").splitlines() if "|" in f]
for group,keys in p["task_groups"].items():
 if filt and group!=filt: continue
 counts={"done":0,"failed":0,"blocked":0}
 for key in keys:
  out=pathlib.Path(jobs[key]["output"])
  for state in counts: counts[state]+=int(pathlib.Path(str(out)+"."+state).exists())
 running=sum(name.startswith("ll_"+group+"_") for name in names)
 missing=len(keys)-sum(counts.values())-running
 print(group,len(keys),counts["done"],counts["failed"],counts["blocked"],running,max(0,missing))
PY
  echo "$QUEUE"
  ACCT=$(sacct -nP -S now-7days -o JobIDRaw,State 2>/dev/null)
  export LL_ACCT="$ACCT"
  "$PYTHON" -B - "$PLAN" "$LL_ROOT" "$FILTER" <<'PY' || return 1
import json,os,pathlib,re,sys
p=json.load(open(sys.argv[1])); root=pathlib.Path(sys.argv[2]); filt=sys.argv[3]
jobs={j["job_key"]:j for j in p["jobs"]}
group_by={key:group for group,keys in p["task_groups"].items() for key in keys}
iters=[]
for group,keys in p["task_groups"].items():
 if filt and group!=filt: continue
 for key in keys:
  path=pathlib.Path(str(jobs[key]["output"])+".iters.csv")
  if path.is_file(): iters.append((path.stat().st_mtime,path))
for _,path in sorted(iters,reverse=True)[:3]:
 lines=path.read_text().splitlines(); print("ITERS",path,lines[-1] if lines else "")
for job in p["jobs"]:
 if filt and group_by[job["job_key"]]!=filt: continue
 marker=pathlib.Path(str(job["output"])+".failed")
 if marker.is_file():
  print("FAILED",job["job_key"]); print("\n".join(marker.read_text().splitlines()[:5]))
submitted={}
path=root/"submitted.tsv"
if path.is_file():
 for line in path.read_text().splitlines():
  f=line.split("\t"); submitted[f[2]]=f[1]
oom=set()
for line in os.environ.get("LL_ACCT","").splitlines():
 f=line.split("|"); base=f[0].split("_",1)
 if len(f)>1 and f[1].startswith("OUT_OF_MEMORY") and len(base)==2 and base[0] in submitted and (not filt or submitted[base[0]]==filt):
  try: oom.add(jobs[p["task_groups"][submitted[base[0]]][int(base[1])]]["job_key"])
  except (KeyError,IndexError,ValueError): pass
for err in (root/"logs").glob("*.err"):
 if "Exceeded job memory limit" in err.read_text(errors="replace"):
  m=re.search(r"_(\d+)_(\d+)\.err$",err.name)
  if m and m.group(1) in submitted and (not filt or submitted[m.group(1)]==filt):
   try: oom.add(jobs[p["task_groups"][submitted[m.group(1)]][int(m.group(2))]]["job_key"])
   except (KeyError,IndexError): pass
print("OOM",len(oom),",".join(sorted(oom)))
PY
}
main "$@"
