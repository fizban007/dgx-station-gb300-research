"""Print cumulative spec-decode counters from vLLM /metrics as JSON (drafts, draft tokens, accepted tokens)."""
import json, os, re, urllib.request
txt = urllib.request.urlopen(f"http://127.0.0.1:{os.environ.get('PORT', '30006')}/metrics", timeout=30).read().decode()
out = {}
for k in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
    out[k] = sum(float(x) for x in re.findall(rf"^vllm:spec_decode_{k}_total(?:\{{[^}}]*\}})? ([0-9.e+]+)", txt, re.M))
print(json.dumps(out))
