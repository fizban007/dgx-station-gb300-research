#!/usr/bin/env bash
# Print or export the DOCKER_MOUNTS and DOCKER_ENV that add the ablit switch to a lane boot, in the style of
# m3/engram-nvfp4/lane-mounts.sh:
#
#   DOCKER_MOUNTS="$(lane-mounts.sh)"        # print mode, for launch-m3.sh / swap-to-m3v2.sh
#   . lane-mounts.sh && ./start-server.sh    # source mode, for the wrappers in this directory
#
# A non-empty DOCKER_MOUNTS also makes swap-to-m3v2.sh skip its own vllm#58132 mounts, which is why they are
# included here (REPLAY_OVERLAY=0 leaves them out and skips the staleness check).
#
# Two reasons this exists:
#   * the lane launcher builds its own vllm#58132 overlay mounts only when DOCKER_MOUNTS is empty, and the
#     ablit paths have to preset it (that is what keeps m3 unmodified), so the overlay list is repeated here.
#   * the injection .pth must be mounted into a site directory *itself*: site.py only reads *.pth files that sit
#     directly in a site dir, not inside a package. Mounting it at $SITE/vllm/ablit.pth silently does nothing.
#
# Keep REPLAY_FILES in step with m3/start-server.sh and m3/overlay-58132/tree; the check below warns about
# overlay files that would go unmounted, and the caller warns if the boot log has no
# "Decoder SWA bounded replay" line (that overlay is worth +57-61% prefill).
#
# Executed (DOCKER_MOUNTS="$(container-mounts.sh)"): prints the value for command substitution.
# Sourced (`. container-mounts.sh`): stays quiet and just exports.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ABLIT=${ABLIT:-$HERE}
M3=${M3:-$(cd "$HERE/../m3" 2>/dev/null && pwd || true)}
if [ -z "${M3:-}" ] || [ ! -f "$M3/launch-m3.sh" ]; then M3=${M3_STATION:-${HOME}/ai/llm/ds4-station/m3}; fi

# The vllm#58132 overlay tree is not committed in this fork (only overlay.diff is), so point OVERLAY at it
# or at the station copy, the way m3/engram-nvfp4/lane-mounts.sh points at its overlay.
OVERLAY=${OVERLAY:-$M3/overlay-58132/tree/vllm}
if [ ! -d "$OVERLAY" ]; then OVERLAY=${OVERLAY_STATION:-${HOME}/ai/llm/ds4-station/m3/overlay-58132/tree/vllm}; fi
SITE=${SITE:-/usr/local/lib/python3.12/dist-packages}
VLLM_PKG=$SITE/vllm

REPLAY_FILES=(config/cache.py models/deepseek_v41/decoder_replay_layers.py models/deepseek_v41/nvidia/model.py
              models/deepseek_v41/nvidia/model_state.py models/deepseek_v41/nvidia/vl_model.py)

[ -f "$ABLIT/ablit.pth" ] || { echo "ablit: $ABLIT/ablit.pth is missing" >&2; exit 1; }
[ -f "$ABLIT/hook/ablit_switch.py" ] || { echo "ablit: $ABLIT/ablit_switch.py is missing" >&2; exit 1; }

mounts=""
if [ "${REPLAY_OVERLAY:-1}" = 1 ]; then
  for f in "${REPLAY_FILES[@]}"; do
    [ -f "$OVERLAY/$f" ] || { echo "ablit: overlay file $OVERLAY/$f is missing" >&2; exit 1; }
    mounts="$mounts $OVERLAY/$f:$VLLM_PKG/$f:ro"
  done
  if [ -d "$OVERLAY" ]; then
    while read -r f; do
      case " ${REPLAY_FILES[*]} " in
        *" $f "*) ;;
        *) echo "ablit: WARNING $OVERLAY/$f is not in REPLAY_FILES, so it would not be mounted" >&2 ;;
      esac
    done < <(cd "$OVERLAY" && find . -name '*.py' -printf '%P\n' | sort)
  fi
fi
# The data directory is mounted read-write (mode.json, state.json, selftest.json land there) and the hook only
# needs $SITE plus that directory, so the ablit tree itself does not have to be mounted at all.
DATA=${DATA:-$ABLIT/data}
mounts="$mounts $ABLIT/ablit.pth:$SITE/ablit.pth:ro $ABLIT/hook/ablit_switch.py:$SITE/ablit_switch.py:ro $DATA:/ablit/data"
case "$mounts" in
  *"$SITE/ablit.pth:ro"*"$SITE/ablit_switch.py:ro"*) ;;
  *) echo "ablit: refusing to boot: the injection is not landing in $SITE" >&2; exit 1 ;;
esac
export DOCKER_MOUNTS="$mounts"
# Defaults for a portable checkout: 127.0.0.1. The station this was written on binds vLLM to a tailnet address
# instead (the launcher's --host), and localhost does not answer when the server is bound to one non-loopback
# interface, so there set HOST (or a tool's --base). The wrappers below read these.
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}
export SITE VLLM_PKG ABLIT M3 HERE DATA HOST PORT

# Engram host layout. With the default torch pinned allocator the two tables land in power-of-two chunks
# (2 x 128 GiB + 2 x 4 GiB, 264 GiB resident); with the packed THP path they are exactly rows*264 B
# (190 GiB resident, 189 GiB of it AnonHugePages). Measured 2026-10-08/09; see notes/engram-host-memory.md.
# Defaulted here because an unset ENGRAM_CONFIG silently brings the chunked layout back after a reboot.
if [ -z "${ENGRAM_CONFIG:-}" ]; then
  ENGRAM_CONFIG='{"cpu_offload": true, "use_thp": true}'
fi
export ENGRAM_CONFIG

if [ "${1:-}" = --print ] || [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  printf '%s\n' "$DOCKER_MOUNTS"
fi
