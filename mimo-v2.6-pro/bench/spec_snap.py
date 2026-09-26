import json, re, sys, urllib.request
txt = urllib.request.urlopen("http://127.0.0.1:30007/metrics", timeout=30).read().decode()
print(json.dumps({k: sum(float(x) for x in re.findall(rf"^vllm:spec_decode_{k}_total(?:\{{[^}}]*\}})? ([0-9.e+]+)", txt, re.M)) for k in ("num_drafts", "num_draft_tokens", "num_accepted_tokens")}))
