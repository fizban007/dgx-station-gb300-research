"""One long streamed chat generation with exact per-chunk token counts (continuous_usage_stats).
Reports TTFT, e2e tok/s (tokens / request wall time), decode tok/s (after first token), peak tok/s over sliding
1 s and 5 s windows, and spec-decode acceptance from /metrics deltas. Stops client-side at MAX_CHARS generated
characters (reasoning + content). Saves reasoning, content, per-chunk timeline, and the first ```html block.
Canonical prompt: /home/jasonc/research/longgen/tetris-ps4-prompt.txt (PS4 Tetris long-coding test; keep verbatim, typos included).
Usage: long-gen.py <prompt-file> <out-dir> [max_chars=100000] [max_tokens=60000] [cap_on=all|content]
cap_on=content applies MAX_CHARS to the final answer only, leaving thinking uncapped (max_tokens still bounds it)."""
import json, os, re, sys, time, urllib.request
PORT = os.environ.get("PORT", "30006"); MODEL = os.environ.get("MODEL_NAME", "mimo-v26-flash")
prompt_file, out = sys.argv[1], sys.argv[2]
max_chars = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 60_000
cap_on = sys.argv[5] if len(sys.argv) > 5 else "all"
os.makedirs(out, exist_ok=True)

def spec():
    txt = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=30).read().decode()
    out = {k: sum(float(x) for x in re.findall(rf"^vllm:spec_decode_{k}_total(?:\{{[^}}]*\}})? ([0-9.e+]+)", txt, re.M))
           for k in ("num_drafts", "num_draft_tokens", "num_accepted_tokens")}
    for pos, v in re.findall(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\} ([0-9.e+]+)', txt, re.M):
        out[f"pos{pos}"] = out.get(f"pos{pos}", 0.0) + float(v)
    return out

def phase(a, b):
    """Spec stats between two snapshots: accepted/draft, P(accept length >= i), projected tokens/step for each k."""
    d = {k: b.get(k, 0.0) - a.get(k, 0.0) for k in b}
    n = d.get("num_drafts", 0.0)
    if not n:
        return None
    p = [d.get(f"pos{i}", 0.0) / n for i in range(sum(1 for k in d if k.startswith("pos")))]
    return {"drafts": int(n), "accepted_per_draft": round(d["num_accepted_tokens"] / n, 2),
            "p_len_ge": [round(x, 3) for x in p],
            "tokens_per_step_if_k": {k: round(1 + sum(p[:k]), 2) for k in range(1, len(p) + 1)}}

body = {"model": MODEL, "messages": [{"role": "user", "content": open(prompt_file).read().rstrip("\n")}],
        "temperature": 1.0, "top_p": 0.95, "max_tokens": max_tokens, "stream": True,
        "chat_template_kwargs": {"enable_thinking": True},
        "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
s0 = spec()
reasoning, content, timeline = [], [], []  # timeline: (t_since_start, cumulative completion tokens)
finish, stopped_by, nchars, t_content, s_mid = None, None, 0, None, None
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=3600) as r:
    for raw in r:
        line = raw.strip()
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        ev = json.loads(line[6:]); now = time.perf_counter() - t0
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            rs = d.get("reasoning") or d.get("reasoning_content") or ""
            ct = d.get("content") or ""
            reasoning.append(rs); content.append(ct)
            nchars += len(ct) if cap_on == "content" else len(rs) + len(ct)
            if ct and t_content is None:
                t_content = now; s_mid = spec()
            finish = ch.get("finish_reason") or finish
        u = ev.get("usage")
        if u and u.get("completion_tokens"):
            if not timeline or u["completion_tokens"] != timeline[-1][1]:
                timeline.append((now, u["completion_tokens"]))
        if nchars >= max_chars:
            stopped_by = f"client cap {max_chars:,} chars ({cap_on})"; break
t_end = time.perf_counter() - t0
s1 = spec()

ttft, toks, t_last = timeline[0][0], timeline[-1][1], timeline[-1][0]
def peak(win):
    best, j = 0.0, 0
    for i in range(len(timeline)):
        while timeline[i][0] - timeline[j][0] > win:
            j += 1
        # window ending at chunk i, anchored at the chunk just before j so that tokens/elapsed are both counted
        a = timeline[j - 1] if j > 0 else (0.0, 0)
        if timeline[i][0] - a[0] >= win * 0.9:
            best = max(best, (timeline[i][1] - a[1]) / (timeline[i][0] - a[0]))
    return best
d = {k: s1[k] - s0[k] for k in s0}
R, C = "".join(reasoning), "".join(content)
summary = {
    "completion_tokens": toks, "chars_reasoning": len(R), "chars_content": len(C),
    "finish_reason": finish, "stopped_by": stopped_by or f"server ({finish})",
    "ttft_s": round(ttft, 3), "wall_s": round(t_end, 2),
    "answer_starts_s": round(t_content, 2) if t_content is not None else None,
    "e2e_tok_s": round(toks / t_end, 1), "decode_tok_s": round((toks - timeline[0][1]) / (t_last - ttft), 1),
    "peak_1s_tok_s": round(peak(1.0), 1), "peak_5s_tok_s": round(peak(5.0), 1),
    "spec_drafts": int(d["num_drafts"]),
    "accepted_per_draft": round(d["num_accepted_tokens"] / d["num_drafts"], 2) if d["num_drafts"] else None,
    "accept_rate_pct": round(100 * d["num_accepted_tokens"] / d["num_draft_tokens"], 1) if d["num_draft_tokens"] else None,
}
open(f"{out}/reasoning.txt", "w").write(R); open(f"{out}/content.md", "w").write(C)
json.dump({"summary": summary, "timeline": timeline}, open(f"{out}/run.json", "w"))
m = re.search(r"```html\s*\n(.*?)(?:```|\Z)", C, re.S)
if m:
    open(f"{out}/tetris.html", "w").write(m.group(1))
    summary["html_chars"] = len(m.group(1)); summary["html_closed"] = "</html>" in m.group(1).lower()
summary["spec_all"] = phase(s0, s1)
if s_mid is not None:
    summary["spec_thinking"] = phase(s0, s_mid); summary["spec_answer"] = phase(s_mid, s1)
json.dump({"summary": summary, "timeline": timeline}, open(f"{out}/run.json", "w"))
print(json.dumps(summary, indent=1))
