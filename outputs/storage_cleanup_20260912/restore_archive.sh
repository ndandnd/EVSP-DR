#!/bin/bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: $0 ARCHIVE.gz DESTINATION SOURCE_SHA256" >&2
    exit 2
fi
archive=$1
destination=$2
expected=$3

[ -f "$archive" ] || { echo "archive not found: $archive" >&2; exit 3; }
[ ! -e "$destination" ] && [ ! -L "$destination" ] || { echo "refusing to overwrite existing destination: $destination" >&2; exit 4; }
mkdir -p "$(dirname "$destination")"
tmp="$destination.restore.part.$$"
trap 'rm -f -- "$tmp"' EXIT
gzip -cd -- "$archive" > "$tmp"
actual=$(sha256sum "$tmp" | cut -d' ' -f1)
[ "$actual" = "$expected" ] || { echo "restored hash mismatch: $actual" >&2; exit 5; }
python3 - "$tmp" <<'PY_SYNC'
import os,sys
fd=os.open(sys.argv[1],os.O_RDONLY)
os.fsync(fd);os.close(fd)
PY_SYNC
# ln creates the destination exclusively, refusing any concurrent collision.
ln -- "$tmp" "$destination"
rm -- "$tmp"
python3 - "$(dirname "$destination")" <<'PY_SYNC'
import os,sys
fd=os.open(sys.argv[1],os.O_RDONLY)
os.fsync(fd);os.close(fd)
PY_SYNC
trap - EXIT
echo "restored=$destination"
echo "sha256=$actual"
