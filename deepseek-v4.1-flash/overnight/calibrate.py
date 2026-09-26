"""Routing-count calibration traffic, unrelated to the benchmark prompts.

Mixes code, shell, math, prose, Chinese, JSON and chat requests with long
generations at moderate concurrency. The benchmark uses an architecture
reference document and a history-of-mathematics article; neither appears here.
"""
import argparse
import concurrent.futures as cf
import json
import time
import urllib.request

PROMPTS = [
    "Write a Python module implementing a thread-safe LRU cache with TTL expiry, type hints, and pytest tests.",
    "Implement a lock-free single-producer single-consumer ring buffer in C++20. Explain the memory ordering of every atomic.",
    "Write a Rust CLI that tails a log file, parses JSON lines, and prints per-minute error counts. Include Cargo.toml.",
    "Write a bash script that backs up a PostgreSQL database nightly, rotates backups older than 14 days, and alerts on failure.",
    "Explain step by step how to solve the integral of x^3 * e^(2x) dx, then verify by differentiation.",
    "Prove that there are infinitely many primes congruent to 3 mod 4. Be rigorous.",
    "A train leaves at 9:40 at 72 km/h; another leaves the same station at 10:15 at 96 km/h on the same track. When and where does the second catch the first? Show your work.",
    "Write a short story (about 800 words) about a lighthouse keeper who discovers the light is signalling to something beneath the sea.",
    "Write a persuasive essay arguing for four-day work weeks, with counterarguments and rebuttals.",
    "用中文详细介绍长城的建造历史、主要结构特点以及它在今天的文化意义。",
    "请用中文写一篇关于人工智能对教育影响的议论文，至少五个段落。",
    "Produce a JSON array of 25 fictional products with fields id, name, category, price, tags, and a nested inventory object per warehouse.",
    "Translate the following into French and German, then explain any idioms: 'It is raining cats and dogs, so let's call it a day and hit the sack early.'",
    "You are a helpful assistant. A user says: 'My sourdough starter smells like nail polish remover and isn't rising. What should I do?' Answer thoroughly.",
    "Compare TCP congestion control algorithms Reno, CUBIC and BBR. Include a table and discuss fairness.",
    "Write SQL to find, per customer, the longest streak of consecutive days with at least one order. Explain the window functions.",
    "Design a REST API for a library management system: resources, endpoints, status codes, pagination and auth. Then write an OpenAPI snippet.",
    "Explain how a transformer's attention works to a high-school student, then again to a graduate student with equations.",
    "Write a Go HTTP server with graceful shutdown, structured logging, a /healthz endpoint and middleware for request IDs.",
    "Debug this: a React component re-renders infinitely when it calls setState inside useEffect with an object dependency. Explain and fix.",
    "Summarize the causes and consequences of the 2008 financial crisis in a structured outline with at least 20 points.",
    "Write a CUDA kernel for a tiled matrix transpose using shared memory that avoids bank conflicts, and explain the indexing.",
    "Plan a 10-day trip to Japan for a family with two kids, with daily itineraries, budget estimates and transit tips.",
    "Given a list of intervals, merge overlapping ones. Provide solutions in Python, Java and JavaScript with complexity analysis.",
]


def post(host, port, prompt, max_tokens):
    body = json.dumps({
        "model": "DeepSeek-V4.1-Flash", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.7, "top_p": 0.95,
    }).encode()
    request = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.load(response)["usage"]["completion_tokens"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--rounds", type=int, default=1)
    args = p.parse_args()
    start, total = time.time(), 0
    jobs = PROMPTS * args.rounds
    with cf.ThreadPoolExecutor(args.concurrency) as pool:
        for tokens in pool.map(lambda q: post(args.host, args.port, q, args.max_tokens), jobs):
            total += tokens
    print(f"{len(jobs)} requests, {total} completion tokens in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
