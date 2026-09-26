"""RTX PRO 6000 sidecar for the experimental peer expert tier.

Loads every layer's cold experts (the complement of the profile's hot set, in
the same ascending order the GB300 uses for its permutation) from the
checkpoint into VRAM, then serves per-layer requests published by the GB300
through the shared pinned buffer. Run it before (or while) the server starts:

  CUDA_VISIBLE_DEVICES=<6000> python peer_server.py --profile hot.json --model <dir>
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/jasonc/vllm-karmic/vllm/model_executor/layers/fused_moe/experts")
import peer_tier as pt  # noqa: E402
from cold_moe_triton import ColdExperts  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--profile", default=None)
p.add_argument("--rowmap", default=None, help="Al-ENGR rowmap: load each layer's cold list in its order")
p.add_argument("--model", default="/home/jasonc/models/DeepSeek-V4.1-Flash")
p.add_argument("--experts", type=int, default=384)
p.add_argument("--limit", type=float, default=10.0)
p.add_argument("--layers", type=int, default=40)
a = p.parse_args()

dev = torch.device("cuda", 0)
torch.cuda.set_device(dev)
index = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
headers = {}


def tensor(name):
    file = os.path.join(a.model, index[name])
    if file not in headers:
        with open(file, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            headers[file] = (8 + size, json.loads(f.read(size)))
    base, header = headers[file]
    meta = header[name]
    start, end = meta["data_offsets"]
    with open(file, "rb") as f:
        f.seek(base + start)
        data = f.read(end - start)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).view(meta["shape"])


profile = json.load(open(a.profile)) if a.profile else None
rowmap = json.load(open(a.rowmap))["layers"] if a.rowmap else None
experts = {}
start = time.time()
total = 0
for layer in range(a.layers):
    if rowmap is not None:
        cold = list(rowmap[str(layer)]["cold"])
    else:
        name = f"language_model.model.layers.{layer}.ffn.experts"
        hot = set(profile[name]["hot"])
        cold = [e for e in range(a.experts) if e not in hot]
    parts = {k: [] for k in ("w13", "s13", "w2", "s2")}
    for e in cold:
        prefix = f"layers.{layer}.ffn.experts.{e}"
        parts["w13"].append(torch.cat([tensor(f"{prefix}.w1.weight"), tensor(f"{prefix}.w3.weight")]))
        parts["s13"].append(torch.cat([tensor(f"{prefix}.w1.scale"), tensor(f"{prefix}.w3.scale")]))
        parts["w2"].append(tensor(f"{prefix}.w2.weight"))
        parts["s2"].append(tensor(f"{prefix}.w2.scale"))
    stacked = {k: torch.stack(v).to(dev) for k, v in parts.items()}
    total += sum(t.numel() for t in stacked.values())
    experts[layer] = ColdExperts(stacked["w13"], stacked["s13"], stacked["w2"], stacked["s2"],
                                 cold_start=0, limit=a.limit, gate_first=True)
    print(f"layer {layer}: {len(cold)} cold experts", flush=True)
print(f"loaded {total / 1e9:.1f} GB of cold experts in {time.time() - start:.0f}s", flush=True)

buf = pt.open_shared(create=True)
host, _ = pt.register(buf, dev)
words = np.frombuffer(buf, dtype=np.int64, count=8)
header = np.frombuffer(buf, dtype=np.int64, count=pt.HEADER, offset=pt.SLOT)
x_host = host[pt.X_OFF:pt.XS_OFF]
xs_host = host[pt.XS_OFF:pt.IDS_OFF]
ids_host = host[pt.IDS_OFF:pt.W_OFF].view(torch.int32)
w_host = host[pt.W_OFF:pt.OUT_OFF].view(torch.float32)
out_host = host[pt.OUT_OFF:pt.OUT_OFF + pt.MAX_TOKENS * pt.HIDDEN * 2].view(torch.bfloat16)
H = pt.HIDDEN
x = torch.empty(pt.MAX_TOKENS, H, dtype=torch.float8_e4m3fn, device=dev)
xs = torch.empty(pt.MAX_TOKENS, H // 32, dtype=torch.uint8, device=dev)
ids = torch.empty(pt.MAX_TOKENS, pt.TOPK, dtype=torch.int32, device=dev)
w = torch.empty(pt.MAX_TOKENS, pt.TOPK, dtype=torch.float32, device=dev)
out = torch.empty(pt.MAX_TOKENS, H, dtype=torch.bfloat16, device=dev)
def work(layer, tokens):
    n = tokens * H
    x[:tokens].view(torch.uint8).reshape(-1)[:n].copy_(x_host[:n], non_blocking=True)
    xs[:tokens].reshape(-1)[: n // 32].copy_(xs_host[: n // 32], non_blocking=True)
    ids[:tokens].reshape(-1)[: tokens * pt.TOPK].copy_(ids_host[: tokens * pt.TOPK], non_blocking=True)
    w[:tokens].reshape(-1)[: tokens * pt.TOPK].copy_(w_host[: tokens * pt.TOPK], non_blocking=True)
    out[:tokens].zero_()
    experts[layer].add_to(out[:tokens], x[:tokens], xs[:tokens], ids[:tokens], w[:tokens])
    out_host[:n].copy_(out[:tokens].reshape(-1), non_blocking=True)


graphs = {}
capture_stream = torch.cuda.Stream(dev)
torch.cuda.set_stream(capture_stream)
print("serving", flush=True)
last, served, busy = int(words[0]), 0, 0.0
while True:
    seq = int(words[0])
    if seq == last:
        continue
    if int(header[0]) != seq:
        continue  # header not yet visible for this sequence
    layer, tokens, count = int(header[1]), int(header[2]), int(header[3])
    if count > 0:
        t0 = time.perf_counter()
        key = (layer, tokens)
        graph = graphs.get(key)
        if graph is None:
            work(layer, tokens)  # compile and warm outside capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                work(layer, tokens)
            graphs[key] = graph
        graph.replay()
        torch.cuda.synchronize()
        busy += time.perf_counter() - t0
        served += 1
        if served % 500 == 0:
            print(f"served {served} cold layer calls, mean {busy / served * 1e6:.1f} us", flush=True)
    words[1] = seq
    last = seq
