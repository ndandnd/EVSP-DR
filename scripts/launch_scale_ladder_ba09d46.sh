#!/bin/bash

# Retired compatibility entry point.  The old script assumed that scientific
# arrays existed when the top-level launcher returned, which is false under
# the reviewed probe-first protocol.
echo "This launcher is retired. Use scripts/launch_scale_ladder_probe_first.sh." >&2
exit 2
