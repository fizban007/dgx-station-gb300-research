# Sourced by the queued DS41 test scripts. wait_idle <ds41|h3|both> <reason> blocks until the named LAN services have had
# no running or waiting requests for IDLE_SECS (default 120), so restarts and timed benchmarks never land on user traffic.
# Our own benchmarks run sequentially, so at a gate none of ours are in flight. A stopped container counts as idle.
inflight() {  # inflight <metrics url> <metric prefix>: running + waiting requests (a failed scrape counts as busy)
  curl -s --max-time 5 "$1" | awk -v p="^$2:num_requests_(running|waiting)[{]" '$0 ~ p {s += $2; n++} END {print (n ? int(s) : 1)}'
}
busy() {
  local d=0 h=0
  local up; up=$(docker ps --format '{{.Names}}')
  if [ "$1" != h3 ] && grep -qx dsv41-flash-megamoe <<<"$up"; then d=$(inflight localhost:30006/metrics vllm); fi
  if [ "$1" != ds41 ] && grep -qx minimax-h3 <<<"$up"; then h=$(inflight localhost:8091/metrics vllm_omni); fi
  echo $((d + h))
}
wait_idle() {
  local need=${IDLE_SECS:-120} quiet=0 t0 n
  t0=$(date +%s)
  while [ "$quiet" -lt "$need" ]; do
    n=$(busy "$1")
    if [ "${n:-1}" -eq 0 ]; then quiet=$((quiet + 10)); else quiet=0; fi
    if [ "$quiet" -lt "$need" ]; then sleep 10; fi
  done
  echo "  [$(date +%H:%M)] $1 idle for ${need}s before: $2 (waited $(( ($(date +%s) - t0) / 60 )) min)"
}
