#!/usr/bin/env bash
# Install compressed weight scales into an M3 checkout: copy the new files and apply patches/ (written against
# deepseek-v4.1-flash/m3 of the dgx-station-gb300-research fork).
#   ./install.sh [--check] [M3_DIR]      M3_DIR defaults to ../m3; --check only tests that the patches apply.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CHECK=0
if [ "${1:-}" = --check ]; then CHECK=1; shift; fi
M3=$(cd "${1:-$HERE/../m3}" && pwd)
for f in hook/mega_peer_hook.py sidecar/peer_server2.py launch-m3.sh; do
  [ -f "$M3/$f" ] || { echo "$M3/$f not found; is $M3 an M3 checkout?"; exit 1; }
done
for p in "$HERE"/patches/*.diff; do
  if ! patch -p3 -d "$M3" --dry-run -s -f < "$p" >/dev/null; then
    echo "$(basename "$p") does not apply to $M3 (already installed, or the file has changed)"; exit 1
  fi
done
if [ "$CHECK" = 1 ]; then echo "all patches apply to $M3"; exit 0; fi
for p in "$HERE"/patches/*.diff; do patch -p3 -d "$M3" -s < "$p"; done
cp "$HERE/hook/sf_compress.py" "$HERE/hook/rowmap-mix-h254.json" "$HERE/hook/rowmap-mix-h258.json" "$M3/hook/"
cp "$HERE/sidecar/csf_encode.py" "$M3/sidecar/"
echo "installed into $M3 (MEGA_SF_COMPRESS=1 and PEER_CSF=1 are the defaults; ROWMAP=rowmap-mix-h254.json for 254/130)"
