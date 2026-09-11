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
[ ! -e "$destination" ] || { echo "refusing to overwrite existing destination: $destination" >&2; exit 4; }
mkdir -p "$(dirname "$destination")"
tmp="$destination.restore.part.$$"
trap 'rm -f -- "$tmp"' EXIT
gzip -cd -- "$archive" > "$tmp"
actual=$(sha256sum "$tmp" | cut -d' ' -f1)
[ "$actual" = "$expected" ] || { echo "restored hash mismatch: $actual" >&2; exit 5; }
mv -- "$tmp" "$destination"
trap - EXIT
echo "restored=$destination"
echo "sha256=$actual"
