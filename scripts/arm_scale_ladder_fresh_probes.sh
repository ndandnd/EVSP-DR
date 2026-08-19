#!/bin/bash
set +e
set +u
set +o pipefail

main() {
  ROOT="$HOME/EVSP-DR-scale-ladder-ba09d46"
  CROOT="$ROOT/src/results/scale_ladder/slad_20260819_031808_ba09d46"
  PLAN="$CROOT/approved-plan.json"
  PLAN_SHA="bcea6b9391cf49515de2dc8ae06ee6bdc70186fde2c80243a2eeaec62b2c083b"
  PREFIX=${PLAN_SHA:0:20}
  GATE=218102
  ATTEMPT=2
  DOUT="$CROOT/probes/default_partition.attempt2.json"
  SOUT="$CROOT/probes/scaglione.attempt2.json"
  LOGS="$CROOT/logs"
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)

  [ -s "$PLAN" ] || { echo "Missing approved plan; nothing changed."; return 1; }
  [ "$(sha256sum "$PLAN" | awk '{print $1}')" = "$PLAN_SHA" ] || {
    echo "Plan hash mismatch; nothing changed."; return 1;
  }
  [ ! -e "$DOUT" ] && [ ! -e "$DOUT.sha256" ] &&
  [ ! -e "$SOUT" ] && [ ! -e "$SOUT.sha256" ] || {
    echo "Attempt-2 output already exists; refusing duplicate submission."; return 1;
  }

  SBATCH=$(jq -r '.sbatch.path' "$PLAN")
  SCONTROL=$(jq -r '.scontrol.path' "$PLAN")
  PYTHON=$(jq -r '.python.path' "$PLAN")
  PYTHON_SHA=$(jq -r '.python.sha256' "$PLAN")
  WORKER="$ROOT/src/submit_scale_ladder_probe.sub"
  WORKER_SHA=$(jq -r '.probe_worker_sha256' "$PLAN")
  PLAN_HOME=$(jq -r '.runtime_environment.HOME' "$PLAN")

  for spec in \
    "$SBATCH:$(jq -r '.sbatch.sha256' "$PLAN")" \
    "$SCONTROL:$(jq -r '.scontrol.sha256' "$PLAN")" \
    "$PYTHON:$PYTHON_SHA" \
    "$WORKER:$WORKER_SHA"
  do
    path=${spec%%:*}
    expected=${spec##*:}
    [ -f "$path" ] && [ "$(sha256sum "$path" | awk '{print $1}')" = "$expected" ] || {
      echo "Executable/worker identity mismatch: $path"; return 1;
    }
  done

  GATE_INFO=$("$SCONTROL" show job -o "$GATE" 2>/dev/null) || {
    echo "Cannot inspect gate $GATE; nothing changed."; return 1;
  }
  case "$GATE_INFO" in
    *"JobState=PENDING"*"Reason=JobHeldUser"*"Comment=SLADG:$PREFIX"*) ;;
    *) echo "Gate is not the exact pending user-held gate; nothing changed."; return 1;;
  esac

  mkdir -p "$CROOT/probes" "$LOGS" || return 1
  DRAW=$("$SBATCH" --parsable --partition=default_partition --no-requeue \
    --time=00:10:00 --cpus-per-task=1 --mem=4G \
    --job-name="LDPDE2${PLAN_SHA:0:3}" \
    --comment="SLADP:$PREFIX:default:2" \
    --output="$LOGS/probe_default_a2_%j.out" \
    --error="$LOGS/probe_default_a2_%j.err" --export=NONE \
    "$WORKER" "$PLAN" "$PLAN_SHA" default "$ATTEMPT" \
    "$PYTHON" "$PYTHON_SHA" "$ROOT" "$PLAN_HOME" "$DOUT" "$WORKER_SHA")
  DSTATE=$?
  DJOB=${DRAW%%;*}
  if [ "$DSTATE" -ne 0 ] || [[ ! "$DJOB" =~ ^[0-9]+$ ]]; then
    echo "Default attempt-2 probe submission failed: $DRAW"; return 1
  fi

  SRAW=$("$SBATCH" --parsable --partition=scaglione --no-requeue \
    --time=00:10:00 --cpus-per-task=1 --mem=4G \
    --job-name="LDPSC2${PLAN_SHA:0:3}" \
    --comment="SLADP:$PREFIX:scaglione:2" \
    --output="$LOGS/probe_scaglione_a2_%j.out" \
    --error="$LOGS/probe_scaglione_a2_%j.err" --export=NONE \
    "$WORKER" "$PLAN" "$PLAN_SHA" scaglione "$ATTEMPT" \
    "$PYTHON" "$PYTHON_SHA" "$ROOT" "$PLAN_HOME" "$SOUT" "$WORKER_SHA")
  SSTATE=$?
  SJOB=${SRAW%%;*}
  if [ "$SSTATE" -ne 0 ] || [[ ! "$SJOB" =~ ^[0-9]+$ ]]; then
    echo "Scaglione attempt-2 probe submission failed: $SRAW"
    echo "Gate remains held; harmless default probe job=$DJOB"
    return 1
  fi

  AUDIT="$CROOT/operator_probe_recovery_$STAMP.txt"
  {
    echo "schema=evsp-dr-controller-probe-recovery-v1"
    echo "plan_sha256=$PLAN_SHA"
    echo "gate_job_id=$GATE"
    echo "default_probe_job_id=$DJOB"
    echo "scaglione_probe_job_id=$SJOB"
    echo "default_output=$DOUT"
    echo "scaglione_output=$SOUT"
    echo "worker_sha256=$WORKER_SHA"
  } >"$AUDIT" || return 1

  "$SCONTROL" update JobId="$GATE" Dependency="afterok:${DJOB}:${SJOB}" || {
    echo "Could not attach probe dependencies; gate remains held."; return 1;
  }
  UPDATED=$("$SCONTROL" show job -o "$GATE" 2>/dev/null) || return 1
  case "$UPDATED" in
    *"Dependency="*"$DJOB"*"$SJOB"*) ;;
    *) echo "Both new dependencies were not recorded; gate remains held."; return 1;;
  esac

  "$SCONTROL" release "$GATE" || {
    echo "Gate release failed; afterok dependencies remain safe."; return 1;
  }
  printf '%s\n' "$UPDATED" >>"$AUDIT"
  sha256sum "$AUDIT" >"$AUDIT.sha256"

  echo "OVERNIGHT_SCALE_LADDER_ARMED=true"
  echo "Fresh probes: default=$DJOB scaglione=$SJOB gate=$GATE"
  echo "The 138 tasks start only if both fresh probes exit 0."
  squeue --me -o '%.14i %.18j %.2t %.10M %R'
}

main
