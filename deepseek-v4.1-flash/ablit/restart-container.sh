#!/usr/bin/env bash
# Recreate only the vLLM container, leaving the RTX PRO 6000 sidecar running.
#
# m3/start-server.sh restarts the sidecar as part of every boot (kill the pid, start a fresh one, wait for
# "serving"), which costs minutes of cold-expert preparation and CUDA-graph capture. The container does not
# need that:
#
#   * the sidecar is a separate host process, it outlives the container;
#   * the shared buffer is opened, never recreated (peer_tier2.open_shared only grows a short file), so the
#     sidecar's mapping and its captured graphs stay valid;
#   * the GB300 side zeroes the sequence words and header at init (PeerTier2.__init__), and the sidecar reads
#     words[0] fresh each poll, so it re-syncs its own counter on the first poll after the restart;
#   * the abliteration touches only dense tensors, so the sidecar's cold-expert weights are still correct.
#
# What must agree between the two sides is the rowmap (the hot/cold split) and VLLM_EXP_PEER2_SMALL. This script
# derives the rowmap from the container it replaces and refuses to run if the live sidecar was started with a
# different one (FORCE=1 overrides, and then you probably want a full boot instead).
#
# Usage:
#   ./restart-container.sh              # recreate the container with the previous container's own settings
#   ./restart-container.sh --dry-run     # print the exact docker command it would issue; change nothing
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
M3=${M3:-$(cd "$HERE/../m3" 2>/dev/null && pwd || true)}
if [ -z "${M3:-}" ] || [ ! -f "$M3/launch-m3.sh" ]; then M3=${M3_STATION:-${HOME}/ai/llm/ds4-station/m3}; fi
NAME=${NAME:-dsv41-flash-megamoe}
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

command -v docker >/dev/null || { echo "ablit: docker not found" >&2; exit 1; }
docker inspect "$NAME" >/dev/null 2>&1 || {
  echo "ablit: no container named $NAME to copy settings from; do a full boot: $ABLIT/start-server.sh" >&2
  exit 1
}

# --- settings of the container being replaced ---------------------------------------------------------
OLD_ENV=$(docker inspect "$NAME" --format '{{range .Config.Env}}{{println .}}{{end}}')
OLD_CMD=$(docker inspect "$NAME" --format '{{json .Config.Cmd}}')
env_val() { printf '%s\n' "$OLD_ENV" | sed -n "s/^$1=//p" | head -1; }

ROWMAP=${ROWMAP:-$(env_val MEGA_ROWMAP | sed 's#.*/##')}
MEGA_PEER=${MEGA_PEER:-$(env_val MEGA_PEER)}
MEGA_PEER=${MEGA_PEER:-2}
MEGA_COLD_TRT=${MEGA_COLD_TRT-$(env_val MEGA_COLD_TRT)}
MEGA_COUNT=${MEGA_COUNT-$(env_val MEGA_COUNT)}
GPU_UTIL=${GPU_UTIL:-$(python3 -c '
import json, sys
c = json.loads(sys.argv[1])
print(c[c.index("--gpu-memory-utilization") + 1] if "--gpu-memory-utilization" in c else "0.832")' "$OLD_CMD")}
# m3/start-server.sh folds the KV-offload settings and prompt-token details into EXTRA; copy them verbatim.
EXTRA=${EXTRA:-$(python3 -c '
import json, sys
c = json.loads(sys.argv[1])
out = []
if "--enable-prompt-tokens-details" in c:
    out.append("--enable-prompt-tokens-details")
if "--kv-offloading-size" in c:
    i = c.index("--kv-offloading-size")
    out += ["--kv-offloading-size", c[i + 1]]
    if "--kv-offloading-backend" in c:
        j = c.index("--kv-offloading-backend")
        out += ["--kv-offloading-backend", c[j + 1]]
print(" ".join(out))' "$OLD_CMD")}
# Everything the *run* passed with -e, not what the image bakes in: subtract the image's own environment so
# only the lane's settings are carried over. launch-m3.sh reads most of these with ${VAR:-default} itself, so
# duplicates with identical values are harmless (docker takes the last occurrence).
IMG=$(docker inspect "$NAME" --format '{{.Config.Image}}')
IMAGE_ENV=$(docker image inspect "$IMG" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null || true)
ENV_PASS=$(comm -23 <(printf '%s\n' "$OLD_ENV" | sort -u) <(printf '%s\n' "$IMAGE_ENV" | sort -u) \
  | grep -vE '^(MEGA_HOOK|MEGA_ROWMAP|MEGA_PEER|MEGA_COUNT|MEGA_COLD_TRT|MEGA_ABLIT|MEGA_ABLIT_DIR|VLLM_SERVER_DEV_MODE|VLLM_LOGGING_LEVEL|CUDA_LOG_FILE)=' \
  | paste -sd' ' -)
export DOCKER_ENV="CUDA_LOG_FILE=${CUDA_LOG_FILE:-$(env_val CUDA_LOG_FILE)} $ENV_PASS MEGA_ABLIT=${MEGA_ABLIT:-auto} MEGA_ABLIT_DIR=/ablit/data"
# An m3-plain boot carries no VLLM_SERVER_DEV_MODE at all; keep the value explicit rather than empty.
dev_mode=${VLLM_SERVER_DEV_MODE:-$(env_val VLLM_SERVER_DEV_MODE)}
export DOCKER_ENV="$DOCKER_ENV VLLM_SERVER_DEV_MODE=${dev_mode:-0}"
export DOCKER_ENV="${DOCKER_ENV// VLLM_SERVER_DEV_MODE=/ VLLM_SERVER_DEV_MODE=}"   # dedupe if it came through

# --- the sidecar must be alive, and split the same way -------------------------------------------------
PIDFILE=$M3/sidecar/peer_server.pid
[ -f "$PIDFILE" ] || { echo "ablit: no $PIDFILE; the sidecar is not running, do a full boot" >&2; exit 1; }
SP=$(cat "$PIDFILE")
kill -0 "$SP" 2>/dev/null || { echo "ablit: sidecar pid $SP is not alive; do a full boot" >&2; exit 1; }
SIDE_ROWMAP=$(tr '\0' ' ' < "/proc/$SP/cmdline" | sed -n 's/.*--rowmap \([^ ]*\).*/\1/p' | sed 's#.*/##')
if [ "$SIDE_ROWMAP" != "$ROWMAP" ]; then
  echo "ablit: sidecar $SP uses rowmap ${SIDE_ROWMAP:-unknown} but this container would use $ROWMAP" >&2
  echo "ablit: the hot/cold split must match; restart both with $ABLIT/start-server.sh (FORCE=1 overrides)" >&2
  [ "${FORCE:-0}" = 1 ] || exit 1
fi
SIDE_SMALL=$(tr '\0' '\n' < "/proc/$SP/environ" | sed -n 's/^VLLM_EXP_PEER2_SMALL=//p')
CONT_SMALL=$(env_val VLLM_EXP_PEER2_SMALL)
if [ -n "$SIDE_SMALL" ] && [ -n "$CONT_SMALL" ] && [ "$SIDE_SMALL" != "$CONT_SMALL" ]; then
  echo "ablit: VLLM_EXP_PEER2_SMALL differs (sidecar $SIDE_SMALL, container was $CONT_SMALL); the two sides" >&2
  echo "ablit: must agree on which buckets are copied whole; do a full boot" >&2
  exit 1
fi

# shellcheck source=lane-mounts.sh
. "$HERE/lane-mounts.sh"

echo "ablit: sidecar $SP alive (rowmap $SIDE_ROWMAP); recreating container $NAME only"
echo "ablit: ROWMAP=$ROWMAP MEGA_PEER=$MEGA_PEER GPU_UTIL=$GPU_UTIL MEGA_COLD_TRT=${MEGA_COLD_TRT:-} MEGA_COUNT=${MEGA_COUNT:-}"
echo "ablit: EXTRA='$EXTRA'"
echo "ablit: DOCKER_ENV='$DOCKER_ENV'"

if [ "$DRY" = 1 ]; then
  # Show the exact docker command without touching anything: stub docker just for launch-m3.sh.
  STUB=$(mktemp -d)
  printf '#!/usr/bin/env bash\nprintf "docker %%s\\n" "$*"\n' > "$STUB/docker"
  chmod +x "$STUB/docker"
  echo "ablit: dry run, launch-m3.sh with the command above (docker stubbed):"
  PATH="$STUB:$PATH" NAME=$NAME MEGA_PEER=$MEGA_PEER ROWMAP=$ROWMAP GPU_UTIL=$GPU_UTIL \
    MEGA_COLD_TRT=$MEGA_COLD_TRT MEGA_COUNT=$MEGA_COUNT EXTRA="$EXTRA" "$M3/launch-m3.sh"
  rm -rf "$STUB"
  exit 0
fi

docker stop -t 20 "$NAME" >/dev/null 2>&1 || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
# Let the driver finish tearing down the old context (the freed ~217 GiB takes a moment): launching into that
# window produced "CUDA error: operation not permitted" at EngineCore init on 2026-10-08. m3/start-server.sh
# avoids it by stopping gracefully *and* restarting the sidecar first; here it is a settle plus one retry below.
sleep 10

started=0
for attempt in 1 2; do
  NAME=$NAME MEGA_PEER=$MEGA_PEER ROWMAP=$ROWMAP GPU_UTIL=$GPU_UTIL \
    MEGA_COLD_TRT=$MEGA_COLD_TRT MEGA_COUNT=$MEGA_COUNT EXTRA="$EXTRA" "$M3/launch-m3.sh"
  sleep 20
  if docker ps -q -f "name=^$NAME$" | grep -q .; then
    started=1
    break
  fi
  echo "ablit: $NAME exited right after launch (attempt $attempt of 2); log tail:" >&2
  docker logs "$NAME" 2>&1 | grep -E "AcceleratorError|Engine core initialization failed|Error" | tail -3 >&2
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  sleep 15
done
[ "$started" = 1 ] || { echo "ablit: giving up after 2 attempts; do a full boot: $ABLIT/start-server.sh" >&2; exit 1; }

echo "ablit: container up; waiting for $HOST:$PORT/v1/models (up to 25 min)"
deadline=$((SECONDS + 1500))
until curl -s -m 3 "http://$HOST:$PORT/v1/models" | grep -q local-model; do
  if [ $SECONDS -ge $deadline ]; then
    echo "ablit: timed out waiting for the API; see: docker logs $NAME" >&2
    exit 1
  fi
  if ! docker ps -q -f "name=^$NAME$" | grep -q .; then
    echo "ablit: $NAME exited during startup" >&2
    docker logs "$NAME" 2>&1 | grep -E "Error:|EngineCore failed" | tail -3 >&2
    exit 1
  fi
  sleep 10
done
echo "ablit: serving on $HOST:$PORT"
if [ "${REPLAY_OVERLAY:-1}" = 1 ] && ! docker logs "$NAME" 2>&1 | grep -q "Decoder SWA bounded replay"; then
  echo "ablit: WARNING the boot log has no 'Decoder SWA bounded replay' line; prefill runs without the overlay" >&2
fi
echo "ablit: now check: docker logs $NAME 2>&1 | grep ABLIT   (expect 'ABLIT on: ...' twice, then the self-test)"
