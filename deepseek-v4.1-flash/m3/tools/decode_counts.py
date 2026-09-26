"""Routing counts from decode only: flush, run C8/C16 prose generation (short prompts), flush, diff."""
import json, sys, time, urllib.request, concurrent.futures as cf, random
sys.path.insert(0, "/home/jasonc/research/megamoe")
from calibrate_routes import flush_snapshot, diff, post
from build_rowmap import cold_share, load_counts, E
def one(i, n=384):
    topics = ["the number", "the city of", "the history of", "how to cook", "the physics of", "a short story about"]
    body = {"model": "dsv41-flash-uva", "messages": [{"role": "user", "content": f"Write about {topics[i % 6]} {i}. Use full paragraphs."}],
            "max_tokens": n, "temperature": 0.7, "chat_template_kwargs": {"thinking": i % 3 == 0}, "ignore_eos": True}
    return post("/v1/chat/completions", body)["usage"]["completion_tokens"]
a = flush_snapshot()
with cf.ThreadPoolExecutor(12) as ex: toks = sum(ex.map(one, range(200, 296)))
b = flush_snapshot()
d = diff(a, b)
json.dump({"layers": d}, open("/home/jasonc/research/megamoe/prof/counts-D.json", "w"))
print("decode tokens generated", toks)
D = {int(k): v for k, v in d.items()}
for name in ("rowmap-static-v1.json", "rowmap-cal-v2.json", "rowmap-cal-v2-h285.json"):
    m = {int(k): v for k, v in json.load(open(f"/home/jasonc/research/megamoe/hook/{name}"))["layers"].items()}
    print(f"{name:>28}: decode cold routes {100 * cold_share(m, D)[0]:.2f}%")
