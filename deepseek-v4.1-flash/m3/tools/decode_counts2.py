"""Second, differently-prompted decode set (code, Q&A, reasoning) for an out-of-sample decode check. Writes counts-E."""
import json, sys, concurrent.futures as cf
sys.path.insert(0, "/home/jasonc/research/megamoe")
from calibrate_routes import flush_snapshot, diff, post
from build_rowmap import cold_share, load_counts, E
prompts = []
for i in range(40):
    prompts.append(f"Write a Python function that solves problem #{i}: given a list of integers, return the {['longest increasing run', 'k most frequent values', 'running median', 'pairs summing to a target'][i % 4]}. Explain it briefly.")
for i in range(28):
    prompts.append(f"Explain {['how TCP congestion control works', 'why the sky is blue', 'what a mortgage amortization schedule is', 'how vaccines train the immune system', 'the causes of the French Revolution', 'how a transistor amplifies'][i % 6]} to a smart {12 + i}-year-old.")
for i in range(28):
    prompts.append(f"A train leaves at {i % 12 + 1}:15 going {40 + i} mph; another leaves an hour later at {55 + i} mph. When does the second catch up? Think it through.")
def one(i_p):
    i, p = i_p
    body = {"model": "dsv41-flash-uva", "messages": [{"role": "user", "content": p}], "max_tokens": 384,
            "temperature": 0.7, "chat_template_kwargs": {"thinking": i % 2 == 0}, "ignore_eos": True}
    return post("/v1/chat/completions", body)["usage"]["completion_tokens"]
a = flush_snapshot()
with cf.ThreadPoolExecutor(12) as ex: toks = sum(ex.map(one, enumerate(prompts)))
b = flush_snapshot()
json.dump({"layers": diff(a, b)}, open("/home/jasonc/research/megamoe/prof/counts-E.json", "w"))
print("decode tokens generated", toks)
