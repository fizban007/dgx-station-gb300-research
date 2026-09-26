#!/usr/bin/env python3
"""bfcl_gate.py — tool-call correctness gate for the :30006 lane (BFCL v4 simple_python + multiple, our grader).

Usage: SUITE=dev|heldout python3 bfcl_gate.py <tag> <data_dir> [out_dir]
  dev = simple_python+multiple (600; used during development). heldout = live_simple+live_multiple (1,311; the frozen
  sibling suite — run ONLY at promotion, never during a campaign). data_dir holds the BFCL_v4_*.json files + possible_answer/.
Writes <out>/bfcl-<tag>.jsonl (one line per case: model call, verdict, spec-decode metrics if present)
and <out>/bfcl-<tag>-summary.json.

Grader = BFCL AST-check semantics, simplified: exactly one tool call expected; function name must match the ground-truth
key (dotted names compared verbatim); every ground-truth param whose allowed list does NOT contain "" is required and the
call's value must be in the allowed list; params the call supplies that are in the ground truth must be in the allowed list;
extra params not in the ground truth fail. Values compared after JSON-normalisation (int/float/str/bool/list/dict);
numeric 5 == 5.0; strings compared stripped. This is a same-grader A/B between two servers, not a leaderboard number.

Env: BASE_URL MODEL API_KEY CONC(8) MAXTOK(2048) THINKING(0) LIMIT(0=all) PER_REQ_METRICS(0)
Only urllib (box host python has no openai/numpy).
"""
from __future__ import annotations
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import urllib.request

BASE = os.getenv("BASE_URL", "http://127.0.0.1:30006/v1")
MODEL = os.getenv("MODEL", "dsv41-flash-uva")
KEY = os.getenv("API_KEY", "none")
CONC = int(os.getenv("CONC", "8"))
MAXTOK = int(os.getenv("MAXTOK", "2048"))
THINKING = os.getenv("THINKING", "0") == "1"
LIMIT = int(os.getenv("LIMIT", "0"))
PERREQ = os.getenv("PER_REQ_METRICS", "0") == "1"
CTK = json.loads(os.getenv("CTK")) if os.getenv("CTK") else None  # GLM: {"reasoning_effort":"low"}
STRICT = os.getenv("STRICT", "0") == "1"  # per-tool strict:true -> xgrammar structural tag


def bfcl_to_openai_tools(funcs):
    tools = []
    for f in funcs:
        p = json.loads(json.dumps(f.get("parameters", {})))
        def fix(o):
            if isinstance(o, dict):
                if o.get("type") == "dict": o["type"] = "object"
                if o.get("type") == "tuple": o["type"] = "array"
                if o.get("type") == "float": o["type"] = "number"
                for v in o.values(): fix(v)
            elif isinstance(o, list):
                for v in o: fix(v)
        fix(p)
        fn = {"name": f["name"], "description": f.get("description", ""), "parameters": p}
        if STRICT: fn["strict"] = True
        tools.append({"type": "function", "function": fn})
    return tools


def norm(v):
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, str):
        s = v.strip()
        try:
            return float(s) if s.replace(".", "", 1).replace("-", "", 1).isdigit() else s
        except Exception:
            return s
    if isinstance(v, list): return [norm(x) for x in v]
    if isinstance(v, dict): return {k: norm(x) for k, x in v.items()}
    return v


def grade(call, gt_list):
    """call: {"name":..., "arguments": dict}. gt_list: list of {fname: {param: [allowed...]}}; any one ground truth may match."""
    if call is None: return False, "no_call"
    for gt in gt_list:
        (fname, params), = gt.items()
        if call["name"] != fname and call["name"].replace(".", "_") != fname.replace(".", "_"):
            continue
        args = call["arguments"]; ok = True; why = ""
        for p, allowed in params.items():
            required = "" not in allowed
            if p not in args:
                if required: ok = False; why = f"missing:{p}"; break
                continue
            av = norm(args[p]); al = [norm(x) for x in allowed]
            if av not in al and not (isinstance(av, str) and any(isinstance(x, str) and av.lower() == x.lower() for x in al)):
                ok = False; why = f"value:{p}={args[p]!r}"; break
        if ok:
            extra = [k for k in args if k not in params]
            if extra: ok = False; why = f"extra:{extra}"
        if ok: return True, "ok"
        last = why
    return False, locals().get("last", f"name:{call['name']}")


def ask(case):
    tools = bfcl_to_openai_tools(case["function"])
    msgs = case["question"][0]
    body = {"model": MODEL, "messages": msgs, "tools": tools, "tool_choice": "auto", "temperature": 0,
            "max_tokens": MAXTOK, "chat_template_kwargs": CTK if CTK is not None else {"thinking": THINKING}}
    if PERREQ: body["per_request_spec_decode_metrics"] = True
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r: j = json.load(r)
    except Exception as ex:  # noqa
        return {"id": case["id"], "error": f"{type(ex).__name__}: {str(ex)[:200]}", "secs": round(time.time() - t0, 1)}
    ch = j["choices"][0]; msg = ch["message"]
    tcs = msg.get("tool_calls") or []
    call = None; parse_err = None
    if tcs:
        try:
            a = tcs[0]["function"]["arguments"]
            call = {"name": tcs[0]["function"]["name"], "arguments": json.loads(a) if isinstance(a, str) else a}
        except Exception as ex:  # noqa
            parse_err = f"args_json:{str(ex)[:80]}"
    ok, why = grade(call, case["gt"])
    if parse_err: why = parse_err
    rec = {"id": case["id"], "ok": ok, "why": why, "n_calls": len(tcs), "call": call,
           "content": (msg.get("content") or "")[:200], "finish": ch.get("finish_reason"),
           "usage": j.get("usage"), "secs": round(time.time() - t0, 1)}
    sd = (j.get("usage") or {}).get("speculative_decoding") or (j.get("metrics") or {}).get("speculative_decoding")
    if sd: rec["spec"] = sd
    return rec


def load(data_dir, name):
    cases = [json.loads(l) for l in open(f"{data_dir}/BFCL_v4_{name}.json")]
    gts = {json.loads(l)["id"]: json.loads(l)["ground_truth"] for l in open(f"{data_dir}/possible_answer/BFCL_v4_{name}.json")}
    for c in cases: c["gt"] = gts[c["id"]]; c["cat"] = name
    return cases


SUITES = {"dev": ("simple_python", "multiple"), "heldout": ("live_simple", "live_multiple")}


def main():
    tag, data_dir = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "."
    suite = os.getenv("SUITE", "dev")
    cases = [c for name in SUITES[suite] for c in load(data_dir, name)]
    if LIMIT: cases = cases[:LIMIT]
    t0 = time.time(); recs = []
    with open(f"{out}/bfcl-{tag}.jsonl", "w") as f, ThreadPoolExecutor(CONC) as ex:
        for i, r in enumerate(ex.map(ask, cases)):
            r["tag"] = tag; r["cat"] = cases[i]["cat"]; recs.append(r); f.write(json.dumps(r) + "\n"); f.flush()
            if i % 50 == 0: print(f"[{tag}] {i}/{len(cases)} {time.time()-t0:.0f}s", flush=True)
    summ = {"tag": tag, "model": MODEL, "n": len(recs), "wall_s": round(time.time() - t0),
            "errors": sum(1 for r in recs if r.get("error"))}
    summ["suite"] = suite; summ["ctk"] = CTK; summ["strict"] = STRICT; summ["maxtok"] = MAXTOK; summ["conc"] = CONC
    for cat in SUITES[suite] + ("all",):
        rs = [r for r in recs if not r.get("error") and (cat == "all" or r["cat"] == cat)]
        n = len(rs); ok = sum(1 for r in rs if r["ok"])
        whys = {}
        for r in rs:
            if not r["ok"]: whys[r["why"].split(":")[0]] = whys.get(r["why"].split(":")[0], 0) + 1
        summ[cat] = {"n": n, "ok": ok, "acc": round(ok / n, 4) if n else None, "no_call": sum(1 for r in rs if r["n_calls"] == 0),
                     "multi_call": sum(1 for r in rs if r["n_calls"] > 1), "fail_kinds": whys}
    specs = [r["spec"] for r in recs if r.get("spec")]
    if specs:
        summ["spec_metrics_n"] = len(specs); summ["spec_sample"] = specs[0]
    json.dump(summ, open(f"{out}/bfcl-{tag}-summary.json", "w"), indent=1)
    a, b = SUITES[suite]
    print(f"BFCL {tag} [{suite}]: all {summ['all']['ok']}/{summ['all']['n']} = {summ['all']['acc']} | {a} {summ[a]['acc']} | {b} {summ[b]['acc']} | errors {summ['errors']} | {summ['wall_s']}s")


if __name__ == "__main__":
    main()
