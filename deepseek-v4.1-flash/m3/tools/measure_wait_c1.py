"""GB300 wait on the 6000 per forward pass at low concurrency (C1/2/4), default buckets, 2 runs each."""
import json
import sys
import time

sys.path.insert(0, "/home/jasonc/research/megamoe")
import measure_wait as mw  # noqa: E402

open(mw.OVERRIDE, "w").write("{}")
time.sleep(4)
mw.run(1, 50)
for c in (1, 2, 4):
    for rep in range(2):
        print(json.dumps({"C": c, "rep": rep, **mw.run(c, 3000 + 10 * c + rep)}), flush=True)
