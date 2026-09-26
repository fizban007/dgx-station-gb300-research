"""RTX PRO 6000 sidecar for peer tier v2: cold experts in VRAM, computed with b12x's SM120 fused MoE.

Loads each layer's cold experts (rowmap order, the GB300's local cold ids) from the checkpoint, prepares
them for b12x w4a8_mx (MXFP4 weights, MXFP8 activations; preparation rewrites the weights in place), and
captures one CUDA graph per (layer, row bucket). Per published sequence it copies the compacted rows in,
pads the bucket with masked routes, replays, and copies the rows' outputs back.

  CUDA_VISIBLE_DEVICES=<6000> python peer_server2.py --rowmap rowmap.json
"""
import argparse
import gc
import json
import os
import struct
import sys
import time
from dataclasses import replace

import numpy as np
import torch
import triton
import triton.language as tl

sys.path.insert(0, "/home/jasonc/research/megamoe/hook")
import peer_tier2 as pt  # noqa: E402

import b12x.moe.fused_moe as fm  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402

H, I, K, LIMIT = pt.HIDDEN, 2304, pt.TOPK, 10.0
BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


@triton.jit
def _dequant(X, XS, A, H: tl.constexpr, BLOCK: tl.constexpr):
    """MXFP8 row (E4M3 + E8M0 per 32) -> BF16."""
    t = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    # X holds raw E4M3 bytes (uint8, so host copies move bytes): reinterpret, don't convert the integers.
    x = tl.load(X + t * H + cols).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    e = tl.load(XS + t * (H // 32) + cols // 32).to(tl.int32) - 127
    tl.store(A + t * H + cols, (x * tl.exp2(e.to(tl.float32))).to(tl.bfloat16))


def tensor_reader(model):
    index = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    headers = {}

    def tensor(name):
        file = os.path.join(model, index[name])
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

    return tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rowmap", required=True)
    p.add_argument("--model", default="/home/jasonc/models/DeepSeek-V4.1-Flash")
    p.add_argument("--layers", type=int, default=40)
    a = p.parse_args()

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    sys.path.insert(0, "/home/jasonc/b12x-exp")
    from benchmarks.moe_preparation import request_for_capacity

    tensor = tensor_reader(a.model)
    rowmap = json.load(open(a.rowmap))["layers"]
    rows_max = max(BUCKETS)
    x_buf = torch.empty(rows_max, H, dtype=torch.uint8, device=dev)
    xs_buf = torch.empty(rows_max, H // 32, dtype=torch.uint8, device=dev)
    a_buf = torch.zeros(rows_max, H, dtype=torch.bfloat16, device=dev)
    ids_buf = torch.full((rows_max, K), -1, dtype=torch.int32, device=dev)
    w_buf = torch.zeros(rows_max, K, dtype=torch.float32, device=dev)
    out_buf = torch.zeros(rows_max, H, dtype=torch.bfloat16, device=dev)
    # Tune on realistic compacted rows: one or two live cold routes per row, the rest masked. (With every
    # route masked, the autotuner times empty work and some buckets end up with configurations 2-3x slower.)
    g = torch.Generator(device=dev).manual_seed(0)
    a_buf.normal_(0.0, 0.3, generator=g)
    live = torch.rand(rows_max, K, device=dev, generator=g) < (1.4 / K)
    live[:, 0] = True
    n_cold_guess = len(rowmap["0"]["cold"])
    ids_buf.copy_(torch.where(live, torch.randint(0, n_cold_guess, (rows_max, K), device=dev, generator=g,
                                                  dtype=torch.int32), -1))
    w_buf.copy_(torch.where(live, torch.rand(rows_max, K, device=dev, generator=g) * 0.3, 0.0))

    plan = None
    experts, decls, keep = {}, {}, []
    shared_scratch, shared_src = {}, {}
    start = time.time()
    for layer in range(a.layers):
        cold = list(rowmap[str(layer)]["cold"])
        parts = {k: [] for k in ("w13", "s13", "w2", "s2")}
        for e in cold:
            prefix = f"layers.{layer}.ffn.experts.{e}"
            parts["w13"].append(torch.cat([tensor(f"{prefix}.w1.weight"), tensor(f"{prefix}.w3.weight")]))
            parts["s13"].append(torch.cat([tensor(f"{prefix}.w1.scale"), tensor(f"{prefix}.w3.scale")]))
            parts["w2"].append(tensor(f"{prefix}.w2.weight"))
            parts["s2"].append(tensor(f"{prefix}.w2.scale"))
        W = {k: torch.stack(v).to(dev) for k, v in parts.items()}
        del parts
        ones = torch.ones(len(cold), dtype=torch.float32, device=dev)
        one = torch.ones((), dtype=torch.float32, device=dev)
        if plan is None:
            plan = fm.plan_weights(
                source=fm.PackedSource(format=fm.PackedSourceFormat("fp4_e8m0_k32"),
                                       w13_layout=fm.W13Layout("w31")),
                activation=fm.ActivationSpec(mode=fm.ActivationMode.A8, nonlinearity="silu",
                                             io_dtype=torch.bfloat16, swiglu_limit=LIMIT,
                                             swiglu_alpha=None, swiglu_beta=None),
                geometry=fm.MoEGeometry(num_experts=len(cold), hidden_size=H, intermediate_size=I))
        # b12x's DS-V4.1 contract: E8M0 scales clamped to 247, [w1, w3] rows as layout "w31".
        ex = fm.prepare_weights(plan=plan, weights=fm.PackedWeights(
            w13=W["w13"], w2=W["w2"],
            w13_block_scales=W["s13"].clamp_(max=247).view(torch.float8_e8m0fnu),
            w2_block_scales=W["s2"].clamp_(max=247).view(torch.float8_e8m0fnu),
            w13_global_scales=ones, w2_global_scales=ones, input_scale=one, intermediate_scale=one,
            immutable_input_scales=True))
        decl = fm.plan_execution(experts=ex, capacity=fm.ExecutionCapacity(
            max_tokens=rows_max, top_k=K, warmup_token_counts=tuple(BUCKETS), route_num_experts=0))
        def call(st, m, ex=ex):
            # Preparation keeps its trial calls alive, so they share one scratch set and one input copy
            # per bucket across layers instead of allocating ~1 GB per layer.
            # Candidate configurations need different scratch sizes; b12x accepts any buffer at least as
            # large as required, so grow the shared buffer to the largest request seen.
            specs = st.scratch.scratch_specs()
            cur = shared_scratch.get(m)
            if cur is None or len(cur) != len(specs) or any(
                    t.numel() < int(sp.shape[0]) or t.dtype != sp.dtype for t, sp in zip(cur, specs)):
                shared_scratch[m] = tuple(
                    torch.empty(max(int(sp.shape[0]), cur[i].numel() if cur and i < len(cur) else 0),
                                dtype=sp.dtype, device=sp.device)
                    for i, sp in enumerate(specs))
            if m not in shared_src:
                shared_src[m] = a_buf[:m].clone()
            binding = st.bind(scratch=shared_scratch[m], a=a_buf[:m], experts=ex, topk_ids=ids_buf[:m],
                              topk_weights=w_buf[:m], output=out_buf[:m], input_scales_static=True)
            return PreparedCall(run=binding.run, output=out_buf[:m], owners=(binding,))

        calls = {m: (lambda st, m=m: call(st, m)) for m in decl.token_counts}

        def race(st, m):
            trial = call(st, m)
            return replace(trial, produce=lambda: a_buf[:m].copy_(shared_src[m]))

        # A session per layer: it keeps its trial bindings and scratch alive until it is dropped.
        session = PreparationSession(device=dev)
        session.prepare((request_for_capacity(
            decl, name=f"peer.cold.l{layer}", calls=calls,
            benchmark_calls={m: (lambda st, m=m: race(st, m)) for m in calls}),))
        del session, calls, race
        gc.collect()
        torch.cuda.empty_cache()
        experts[layer], decls[layer] = ex, decl
        keep.append(W)  # prepare_weights rewrote these in place; the prepared experts use their storage
        print(f"layer {layer}: {len(cold)} cold experts prepared, "
              f"{torch.cuda.memory_allocated() / 1e9:.1f} GB allocated", flush=True)
    print(f"prepared {a.layers} layers in {time.time() - start:.0f}s; "
          f"{torch.cuda.memory_allocated() / 1e9:.1f} GB allocated", flush=True)

    # One scratch set per bucket, shared by every layer (layers run one at a time).
    from b12x.preparation.types import require_prepared
    scratch = {}
    for m in BUCKETS:
        variant = decls[0].variants[m]
        state = require_prepared(variant, variant.component_id)
        specs = state.scratch.scratch_specs()
        assert all(t.numel() >= int(s.shape[0]) and t.dtype == s.dtype for t, s in zip(shared_scratch[m], specs))
        scratch[m] = shared_scratch[m]

    buf = pt.open_shared(create=True)
    host, _ = pt.register(buf, dev)
    words = np.frombuffer(buf, dtype=np.int64, count=8)
    header = np.frombuffer(buf, dtype=np.int64, count=pt.HEADER, offset=pt.SLOT)
    x_host = host[pt.X_OFF:pt.XS_OFF]
    xs_host = host[pt.XS_OFF:pt.IDS_OFF]
    ids_host = host[pt.IDS_OFF:pt.W_OFF].view(torch.int32)
    w_host = host[pt.W_OFF:pt.OUT_OFF].view(torch.float32)
    out_host = host[pt.OUT_OFF:pt.OUT_OFF + rows_max * H * 2].view(torch.bfloat16)
    stream = torch.cuda.Stream(dev)
    torch.cuda.set_stream(stream)
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    ids_buf[:, 0] = 0
    w_buf[:, 0] = 0.1
    for layer in range(a.layers):
        for m in BUCKETS:
            binding = fm.bind(decls[layer].variants[m], scratch=scratch[m], a=a_buf[:m], experts=experts[layer],
                              topk_ids=ids_buf[:m], topk_weights=w_buf[:m], output=out_buf[:m],
                              input_scales_static=True)

            def work(m=m, binding=binding):
                small = m <= pt.SMALL_ROWS
                if small:
                    # Small buckets copy whole: one replay per decode layer. The GB300 masks rows past its count.
                    x_buf[:m].view(-1).copy_(x_host[: m * H], non_blocking=True)
                    xs_buf[:m].view(-1).copy_(xs_host[: m * (H // 32)], non_blocking=True)
                    ids_buf[:m].view(-1).copy_(ids_host[: m * K], non_blocking=True)
                    w_buf[:m].view(-1).copy_(w_host[: m * K], non_blocking=True)
                _dequant[(m, H // 1024)](x_buf, xs_buf, a_buf, H=H, BLOCK=1024)
                fm.run(binding=binding)
                if small:
                    out_host[: m * H].copy_(out_buf[:m].view(-1), non_blocking=True)

            work()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                work()
            graphs[(layer, m)] = graph
    ids_buf.fill_(-1)
    torch.cuda.synchronize()
    print(f"captured {len(graphs)} graphs; {torch.cuda.memory_allocated() / 1e9:.1f} GB allocated", flush=True)

    print("serving", flush=True)
    last, served, busy, rows_total, stale = int(words[0]), 0, 0.0, 0, 0
    # Per-bucket latency stats (graph replay through sync), dumped every ~2 s; touch STATS_RESET to clear.
    STATS_FILE = "/home/jasonc/research/megamoe/logs/peer_stats.json"
    STATS_RESET = "/home/jasonc/research/megamoe/logs/peer_stats.reset"
    stats, last_dump = {}, 0.0
    base_bucket_of = [0] + [next(m for m in BUCKETS if m >= r) for r in range(1, rows_max + 1)]
    bucket_of = base_bucket_of
    BUCKET_OVERRIDE = "/home/jasonc/research/megamoe/logs/peer_buckets.json"
    while True:
        seq = int(words[0])
        if seq == last:
            continue
        if int(header[0]) != seq:
            continue  # header not yet visible for this sequence
        layer, rows = int(header[1]), int(header[2])
        if rows > 0:
            t0 = time.perf_counter()
            m = bucket_of[rows]
            if m <= pt.SMALL_ROWS:
                graphs[(layer, m)].replay()
            else:
                x_buf[:rows].view(-1).copy_(x_host[: rows * H], non_blocking=True)
                xs_buf[:rows].view(-1).copy_(xs_host[: rows * (H // 32)], non_blocking=True)
                ids_buf[:rows].view(-1).copy_(ids_host[: rows * K], non_blocking=True)
                w_buf[:rows].view(-1).copy_(w_host[: rows * K], non_blocking=True)
                if rows < m:
                    ids_buf[rows:m].fill_(-1)
                graphs[(layer, m)].replay()
                out_host[: rows * H].copy_(out_buf[:rows].view(-1), non_blocking=True)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            busy += dt
            served += 1
            rows_total += rows
            st = stats.setdefault(m, [0, 0.0, 0])
            st[0] += 1
            st[1] += dt
            st[2] += rows
            if served % 256 == 0:
                now = time.time()
                if now - last_dump > 2:
                    last_dump = now
                    # Live bucket override: {"<rows>": <bucket>} remaps row counts to another captured bucket.
                    try:
                        override = json.load(open(BUCKET_OVERRIDE))
                        bucket_of = list(base_bucket_of)
                        for r, b in override.items():
                            bucket_of[int(r)] = int(b)
                    except (OSError, ValueError):
                        bucket_of = base_bucket_of
                    if os.path.exists(STATS_RESET):
                        stats.clear()
                        os.remove(STATS_RESET)
                    with open(STATS_FILE + ".tmp", "w") as f:
                        json.dump({str(k): {"calls": v[0], "mean_us": round(v[1] / v[0] * 1e6, 1),
                                            "mean_rows": round(v[2] / v[0], 1)} for k, v in sorted(stats.items())}, f)
                    os.replace(STATS_FILE + ".tmp", STATS_FILE)
            if served % 2000 == 0:
                print(f"served {served} cold layer calls, mean {busy / served * 1e6:.1f} us, "
                      f"mean rows {rows_total / served:.1f}", flush=True)
        words[1] = seq
        last = seq


if __name__ == "__main__":
    main()
