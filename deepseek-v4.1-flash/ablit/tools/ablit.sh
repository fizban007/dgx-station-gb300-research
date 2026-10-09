#!/usr/bin/env bash
# Switch the M3 server between the official and abliterated DeepSeek-V4.1-Flash weights, online.
# See ../PLAN-abliterated-switch.md. The hook (../ablit_switch.py) does the work; this tool sequences it:
#
#   drain (so no request spans the switch)  ->  request the mode  ->  wait for the hook to apply it  ->
#   flush the prefix cache (KV computed with the other weights must not be reused)
#
# The server address defaults to 127.0.0.1:8001; set HOST/PORT if the lane binds elsewhere.
#
# Usage:
#   tools/ablit.sh status
#   tools/ablit.sh selftest
#   tools/ablit.sh drain [seconds]
#   tools/ablit.sh set abliterated [--force] [--timeout 120] [--no-drain] [--no-flush]
#   tools/ablit.sh set official
#
# The prefix-cache flush needs vLLM's dev endpoints (VLLM_SERVER_DEV_MODE=1 in launch-m3.sh); without them this
# tool says so and you should pass a fresh `cache_salt` on any request whose answer you care about.
set -euo pipefail

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}
HERE=$(cd "$(dirname "$0")" && pwd)
ABLIT_DIR=${ABLIT_DIR:-$(cd "$HERE/.." && pwd)/data}
MODE_FILE="$ABLIT_DIR/mode.json"
STATE_FILE="$ABLIT_DIR/state.json"
SELFTEST_FILE="$ABLIT_DIR/selftest.json"
SERVED=${SERVED:-local-model}
BASE="http://$HOST:$PORT"

die() { echo "ablit: $*" >&2; exit 1; }

running_requests() {
  local body
  body=$(curl -s -m 5 "$BASE/metrics" 2>/dev/null || true)
  [ -z "$body" ] && { echo ""; return; }
  printf '%s\n' "$body" | awk '/^vllm:num_requests_(running|waiting)\{/ {s+=$NF} END {printf "%d", s+0}'
}

drain() {
  local timeout=${1:-60}
  local deadline=$((SECONDS + timeout)) n
  while :; do
    n=$(running_requests)
    if [ -z "$n" ]; then echo "ablit: cannot read $BASE/metrics; skipping the drain check" >&2; return 1; fi
    if [ "$n" = "0" ]; then echo "ablit: drained (0 running, 0 waiting)"; return 0; fi
    if [ $SECONDS -ge $deadline ]; then echo "ablit: $n request(s) still in flight after ${timeout}s" >&2; return 1; fi
    sleep 2
  done
}

poke() {
  # One throwaway token: the hook polls for a mode change inside a step, so an idle engine needs a step to notice.
  curl -s -m 60 -X POST "$BASE/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"1\",\"max_tokens\":1,\"temperature\":0}" >/dev/null 2>&1 || true
}

flush_cache() {
  local code
  code=$(curl -s -m 30 -o /tmp/ablit-flush.$$ -w '%{http_code}' -X POST "$BASE/reset_prefix_cache" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    echo "ablit: prefix cache flushed"
  else
    echo "ablit: prefix cache NOT flushed (HTTP $code). Reused prefixes would carry KV from the previous" >&2
    echo "       weights. Either boot with VLLM_SERVER_DEV_MODE=1 (launch-m3.sh) or send a fresh cache_salt" >&2
    echo "       on requests whose output matters (tools/longctx_check.py already does this)." >&2
  fi
  rm -f /tmp/ablit-flush.$$
}

gate_selftest() {
  [ -f "$STATE_FILE" ] || die "no $STATE_FILE in $ABLIT_DIR: either the hook is off (MEGA_ABLIT=0 or no delta) or the server is still booting"
  python3 - "$STATE_FILE" <<'PY' || exit 1
import json, sys
s = json.load(open(sys.argv[1]))
st = s.get("selftest") or {}
if not st:
    print("ablit: no self-test result yet; the server is probably still booting", file=sys.stderr)
    sys.exit(1)
if st.get("failed"):
    print(f"ablit: self-test failed for {st['failed']} tensor(s); refusing to switch", file=sys.stderr)
    for f in st.get("failed_tensors") or []:
        print(f"  {f['name']}: {f['why']}", file=sys.stderr)
    sys.exit(1)
PY
}

cmd_status() {
  [ -f "$STATE_FILE" ] || die "no $STATE_FILE in $ABLIT_DIR: either the hook is off (MEGA_ABLIT=0 or no delta) or the server is still booting; check: docker logs dsv41-flash-megamoe 2>&1 | grep ABLIT"
  python3 - "$STATE_FILE" "$MODE_FILE" <<'PY'
import json, os, sys
s = json.load(open(sys.argv[1]))
print(f"enabled          {s.get('enabled')}")
print(f"reason           {s.get('reason')}")
print(f"applied          {s.get('applied_mode')} (seq {s.get('applied_seq')}), "
      f"{s.get('switches')} switch(es), last {s.get('last_switch_s')}s at {s.get('last_switch_at')}")
st = s.get("selftest") or {}
if st:
    print(f"self-test        {st.get('verified')}/{st.get('checked')} verified, {st.get('failed')} failed, "
          f"{st.get('seconds')}s, mode {st.get('mode')}")
    print(f"recipes          {st.get('recipes')}")
    delta = st.get("delta") or {}
    if delta:
        print(f"delta            {delta.get('tensors')} tensors from {delta.get('repo')}"
              + ("  [SYNTHETIC - a test fixture, not the real abliteration]" if delta.get("synthetic") else ""))
    layers = st.get("layers") or []
    if layers:
        print(f"layers           {layers[0]}-{layers[-1]} ({len(layers)})")
    for f in st.get("failed_tensors") or []:
        print(f"FAILED           {f['name']}: {f['why']}")
if os.path.exists(sys.argv[2]):
    req = json.load(open(sys.argv[2]))
    print(f"requested        {req.get('mode')} (seq {req.get('seq')})")
else:
    print("requested        none (no mode.json: the model is official)")
for e in (s.get("errors") or [])[-5:]:
    print(f"error            {e['t']} {e['msg']}")
PY
  echo "running requests $(running_requests)"
}

cmd_set() {
  local mode=$1 force=$2 timeout=$3 do_drain=$4 do_flush=$5
  case "$mode" in official|abliterated) ;; *) die "mode must be official or abliterated" ;; esac
  gate_selftest
  if [ "$do_drain" = 1 ]; then
    drain "${DRAIN_TIMEOUT:-60}" || [ "$force" = 1 ] || die "drain timed out; pass --force to switch anyway"
  fi
  local seq
  seq=$(python3 -c "
import json, os
p = '$MODE_FILE'
s = json.load(open(p))['seq'] if os.path.exists(p) else 0
print(s + 1)")
  python3 - "$MODE_FILE" "$mode" "$seq" <<'PY'
import json, os, sys, time
path, mode, seq = sys.argv[1], sys.argv[2], int(sys.argv[3])
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump({"mode": mode, "seq": seq, "requested_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f, indent=1)
os.replace(tmp, path)
print(f"ablit: requested {mode} (seq {seq})")
PY
  local deadline=$((SECONDS + timeout)) applied=0
  while [ $SECONDS -lt $deadline ]; do
    applied=$(python3 -c "
import json
try:
    s = json.load(open('$STATE_FILE'))
except Exception:
    print(0); raise SystemExit
print(1 if s.get('applied_mode') == '$mode' and int(s.get('applied_seq', 0)) >= $seq else 0)")
    [ "$applied" = 1 ] && break
    poke
    sleep 2
  done
  if [ "$applied" != 1 ]; then
    echo "ablit: not applied within ${timeout}s; state follows" >&2
    cmd_status >&2 || true
    die "switch did not take effect"
  fi
  echo "ablit: $mode applied"
  [ "$do_flush" = 1 ] && flush_cache
  cmd_status
}

cmd=${1:-status}
shift || true
case "$cmd" in
  status) cmd_status ;;
  selftest) [ -f "$SELFTEST_FILE" ] || die "no $SELFTEST_FILE yet"; cat "$SELFTEST_FILE" ;;
  drain) drain "${1:-60}" ;;
  set)
    mode=${1:-}
    shift || true
    force=0 timeout=120 do_drain=1 do_flush=1
    while [ $# -gt 0 ]; do
      case "$1" in
        --force) force=1 ;;
        --timeout) timeout=${2:-120}; shift ;;
        --no-drain) do_drain=0 ;;
        --no-flush) do_flush=0 ;;
        *) die "unknown option $1" ;;
      esac
      shift
    done
    cmd_set "$mode" "$force" "$timeout" "$do_drain" "$do_flush" ;;
  *) die "usage: ablit.sh {status|selftest|drain|set official|set abliterated} [options]" ;;
esac
