#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# == 2 ]] || evsp_die "usage: $0 PANEL_A_ROOT PANEL_B_ROOT"
A_ROOT=$(cd "$1" && pwd)
B_ROOT=$(cd "$2" && pwd)
REPO=$(evsp_repo_root)
BRANCH=$(git -C "$REPO" branch --show-current)
evsp_verify_remote_head "$REPO" "$BRANCH" >/dev/null
if squeue --me -h | grep -q .; then
  evsp_die "queue is not empty; audit before adding an overnight portfolio"
fi

echo "=== audit completed 24-hour HiGHS work ==="
bash "$SCRIPT_DIR/audit_highs_unresolved_retry86400.sh" "$A_ROOT" "$B_ROOT"

set +e
echo "=== submit only validated unresolved pools for 48 hours on scaglione ==="
bash "$SCRIPT_DIR/submit_highs_unresolved_retry172800.sh" "$A_ROOT" "$B_ROOT"
MIP_RC=$?

echo "=== launch 18 medium event-CG scale probes on default_partition ==="
bash "$SCRIPT_DIR/submit_medium_event_legacy_overnight.sh"
MEDIUM_RC=$?
set -e

echo "=== resulting queue ==="
squeue -r --me -h -o '%j|%P|%T|%M|%l|%R' | sort
[[ $MIP_RC == 0 ]] || echo "WARNING: 48-hour MIP submission returned $MIP_RC" >&2
[[ $MEDIUM_RC == 0 ]] || echo "WARNING: medium event submission returned $MEDIUM_RC" >&2
[[ $MIP_RC == 0 && $MEDIUM_RC == 0 ]] \
  || evsp_die "one overnight component failed; successful submissions remain queued"
