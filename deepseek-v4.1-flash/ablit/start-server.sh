#!/usr/bin/env bash
# Full lane boot with the ablit switch installed: sidecar plus container, m3 or the image untouched.
# Use this after a host reboot (the sidecar is gone and has to re-prepare: ~160 s for 40 layers).
. "$(dirname "$0")/lane-mounts.sh"
M3=${M3:?} ; HERE=$(cd "$(dirname "$0")" && pwd)
BOOT=$M3/start-server.sh ; [ -x "$BOOT" ] || BOOT=$M3/swap-to-m3v2.sh
echo "ablit: booting via $BOOT with switching ${MEGA_ABLIT}"
exec "$BOOT" "$@"
