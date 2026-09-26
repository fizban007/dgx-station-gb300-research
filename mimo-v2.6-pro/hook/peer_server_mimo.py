"""RTX PRO 6000 sidecar for MiMo-V2.6-Pro: the rowmap's "peer" experts in VRAM, computed with b12x's SM120 fused MoE.

Port of ds41f-exp/peer/peer_server2.py (DeepSeek-V4.1) to MiMo: H 6144, I 2048, top-8, plain SiLU (no clamp), 69 MoE
layers (1..69), checkpoint names model.layers.L.mlp.experts.E.{gate,up,down}_proj.{weight,weight_scale}.

Per layer it loads the peer experts from the checkpoint (local id = position in the rowmap's "peer" list, the same order
the GB300's hotsplit uses), prepares them for b12x w4a8_mx (MXFP4 weights, MXFP8 activations; preparation rewrites the
weights in place) and captures one CUDA graph per (layer, row bucket). Per published sequence it copies the compacted
rows in, pads the bucket with masked routes, replays, and copies the rows' outputs back.

Before printing "serving" it self-tests the exact serving path on a few layers: MXFP8-quantized rows written to the
shared buffer the way the GB300 writes them, the same copy/dequant/replay code, against a float32 reference computed
from freshly read checkpoint bytes. It exits non-zero if any layer's cosine is below --min-cos.

  CUDA_VISIBLE_DEVICES=<6000 UUID> python peer_server_mimo.py --rowmap rowmap.json
"""
import argparse
import gc
import json
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peer_tier_mimo as pt  # noqa: E402

import b12x.moe.fused_moe as fm  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402

H, I, K = pt.HIDDEN, 2048, pt.TOPK
BUCKETS = [m for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384) if m <= pt.MAX_ROWS]
LOG_DIR = "/home/jasonc/research/mimo-pro/logs"


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


def expert_bytes(tensor, layer, e):
    p = f"model.layers.{layer}.mlp.experts.{e}"
    g, u, d = (tensor(f"{p}.{n}.weight") for n in ("gate_proj", "up_proj", "down_proj"))
    gs, us, ds = (tensor(f"{p}.{n}.weight_scale") for n in ("gate_proj", "up_proj", "down_proj"))
    # [gate, up] rows as layout "w31": the same contract the DS-V4.1 sidecar used for [w1, w3] (validated there).
    return torch.cat([g, u]), torch.cat([gs, us]), d, ds


# ---------------------------------------------------------------------------------------------------------------------
# Reference math for the self-test (float32, independent of b12x)
# ---------------------------------------------------------------------------------------------------------------------
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def dequant_mxfp4(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Packed E2M1 [N, K/2] (low nibble = even element) + E8M0 [N, K/32] -> float32 [N, K]."""
    lut = _E2M1.to(w.device)
    lo, hi = lut[(w & 0xF).long()], lut[(w >> 4).long()]
    v = torch.stack([lo, hi], dim=-1).reshape(w.shape[0], -1)
    return v * torch.exp2(s.float() - 127.0).repeat_interleave(32, dim=1)


def quant_mxfp8(x: torch.Tensor):
    """BF16 [T, H] -> (E4M3 bytes [T, H], E8M0 bytes [T, H/32]); per-32 scale 2^ceil(log2(amax / 448))."""
    xb = x.float().view(x.shape[0], -1, 32)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(2.0**-126)
    exp = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    q = (xb / torch.exp2(exp)).to(torch.float8_e4m3fn).view(x.shape[0], -1)
    return q.view(torch.uint8), (exp.squeeze(-1) + 127).to(torch.uint8)


def reference(x_hat, ids, w, raw):
    """sum_k w[t,k] * down(silu(gate x) * up x) over live routes, on the CPU (the 6000 is full by now): each used
    expert is dequantized once and applied to every (row, slot) routed to it. raw[local id] = (w13, s13, w2, s2)."""
    x_hat, ids, w = x_hat.cpu(), ids.cpu(), w.float().cpu()
    out = torch.zeros(x_hat.shape[0], H, dtype=torch.float32)
    for e, (w13, s13, w2, s2) in raw.items():
        t_idx, k_idx = (ids == e).nonzero(as_tuple=True)
        if t_idx.numel() == 0:
            continue
        gu = x_hat[t_idx] @ dequant_mxfp4(w13, s13).T
        h = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
        out.index_add_(0, t_idx, w[t_idx, k_idx].unsqueeze(1) * (h @ dequant_mxfp4(w2, s2).T))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rowmap", required=True)
    p.add_argument("--model", default="/home/jasonc/models/MiMo-V2.6-Pro-RL")
    p.add_argument("--layers", default="1-69", help="inclusive range of MoE layers to serve")
    p.add_argument("--selftest-layers", default="1,35,69")
    p.add_argument("--min-cos", type=float, default=0.995)
    p.add_argument("--read-threads", type=int, default=16)
    a = p.parse_args()

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    from benchmarks.moe_preparation import request_for_capacity  # b12x checkout on sys.path via the editable install

    lo, hi = (int(x) for x in a.layers.split("-"))
    layers = list(range(lo, hi + 1))
    tensor = tensor_reader(a.model)
    rowmap = json.load(open(a.rowmap))["layers"]
    peer = {li: [int(e) for e in rowmap[str(li)]["peer"]] for li in layers}
    layers = [li for li in layers if peer[li]]
    rows_max = max(BUCKETS)
    x_buf = torch.empty(rows_max, H, dtype=torch.uint8, device=dev)
    xs_buf = torch.empty(rows_max, H // 32, dtype=torch.uint8, device=dev)
    a_buf = torch.zeros(rows_max, H, dtype=torch.bfloat16, device=dev)
    ids_buf = torch.full((rows_max, K), -1, dtype=torch.int32, device=dev)
    w_buf = torch.zeros(rows_max, K, dtype=torch.float32, device=dev)
    out_buf = torch.zeros(rows_max, H, dtype=torch.bfloat16, device=dev)
    # Tune on realistic compacted rows: one or two live peer routes per row, the rest masked. (With every route
    # masked the autotuner times empty work and picks configurations 2-3x too slow; see megamoe-m3-results.)
    g = torch.Generator(device=dev).manual_seed(0)
    a_buf.normal_(0.0, 0.3, generator=g)
    live = torch.rand(rows_max, K, device=dev, generator=g) < (1.4 / K)
    live[:, 0] = True
    n_guess = min(len(v) for v in peer.values())
    ids_buf.copy_(torch.where(live, torch.randint(0, n_guess, (rows_max, K), device=dev, generator=g,
                                                  dtype=torch.int32), -1))
    w_buf.copy_(torch.where(live, torch.rand(rows_max, K, device=dev, generator=g) * 0.3, 0.0))

    # b12x allocates a fresh 2x-L2 (256 MiB) flush buffer per PreparationSession to time race candidates. With the card
    # nearly full, the last layers' sessions cannot get one. Share a single buffer for the whole preparation (the flush
    # is stateless: it sums the buffer to evict L2) and free it before graph capture.
    import b12x.preparation._measurement as _meas
    orig_flush, shared_flush = _meas._l2_flush_fn, {}

    def l2_flush_once(device, *, enabled):
        if not enabled:
            return None
        if "fn" not in shared_flush:
            shared_flush["fn"] = orig_flush(device, enabled=True)
        return shared_flush["fn"]

    _meas._l2_flush_fn = l2_flush_once

    plan = None
    experts, decls, keep = {}, {}, []
    shared_scratch, shared_src = {}, {}
    start = time.time()
    pool = ThreadPoolExecutor(a.read_threads)
    dbg = os.environ.get("PEER_DEBUG_MEM") == "1"

    def mem(tag, li):
        if dbg:
            torch.cuda.synchronize()
            print(f"  mem l{li} {tag}: {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

    # Largest layer first: b12x scratch grows with the expert count, and each growth leaves the previous scratch set
    # (~2.5 GiB at 16K rows) pinned by earlier layers' prepared calls. Sized once for the largest layer, it never grows.
    for li in sorted(layers, key=lambda x: -len(peer[x])):
        parts = list(pool.map(lambda e, li=li: expert_bytes(tensor, li, e), peer[li]))
        W = {k: torch.stack([pp[i] for pp in parts]).to(dev) for i, k in enumerate(("w13", "s13", "w2", "s2"))}
        del parts
        mem("weights", li)
        n = len(peer[li])
        ones = torch.ones(n, dtype=torch.float32, device=dev)
        one = torch.ones((), dtype=torch.float32, device=dev)
        # A plan is geometry-specific (num_experts varies per layer).
        plan = fm.plan_weights(
            source=fm.PackedSource(format=fm.PackedSourceFormat("fp4_e8m0_k32"), w13_layout=fm.W13Layout("w31")),
            activation=fm.ActivationSpec(mode=fm.ActivationMode.A8, nonlinearity="silu", io_dtype=torch.bfloat16,
                                         swiglu_limit=None, swiglu_alpha=None, swiglu_beta=None),
            geometry=fm.MoEGeometry(num_experts=n, hidden_size=H, intermediate_size=I))
        ex = fm.prepare_weights(plan=plan, weights=fm.PackedWeights(
            w13=W["w13"], w2=W["w2"],
            w13_block_scales=W["s13"].view(torch.float8_e8m0fnu), w2_block_scales=W["s2"].view(torch.float8_e8m0fnu),
            w13_global_scales=ones, w2_global_scales=ones, input_scale=one, intermediate_scale=one,
            immutable_input_scales=True))
        mem("prepare_weights", li)
        decl = fm.plan_execution(experts=ex, capacity=fm.ExecutionCapacity(
            max_tokens=rows_max, top_k=K, warmup_token_counts=tuple(BUCKETS), route_num_experts=0))
        mem("plan_execution", li)
        ids_buf.remainder_(n).masked_fill_(~live, -1)  # keep tuning ids inside this layer's expert count

        def call(st, m, ex=ex):
            # Trial calls share one scratch set and one input copy per bucket across layers. Candidate
            # configurations need different scratch sizes; grow the shared buffer to the largest request seen.
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

        session = PreparationSession(device=dev)
        session.prepare((request_for_capacity(
            decl, name=f"peer.mimo.l{li}", calls=calls,
            benchmark_calls={m: (lambda st, m=m: race(st, m)) for m in calls}),))
        mem("session.prepare", li)
        del session, calls, race
        gc.collect()
        torch.cuda.empty_cache()
        mem("after gc", li)
        if dbg:
            print("  shared_scratch:", {m: [round(t.numel() * t.element_size() / 2**20) for t in v]
                                       for m, v in shared_scratch.items()}, flush=True)
        experts[li], decls[li] = ex, decl
        keep.append(W)  # prepare_weights rewrote these in place; the prepared experts use their storage
        if li in (layers[0], layers[-1]) or li % 10 == 0 or dbg:
            print(f"layer {li}: {n} peer experts prepared, {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated, "
                  f"{time.time() - start:.0f}s", flush=True)
    pool.shutdown()
    _meas._l2_flush_fn = orig_flush
    shared_flush.clear()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"prepared {len(layers)} layers ({sum(len(peer[li]) for li in layers)} experts) in "
          f"{time.time() - start:.0f}s; {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated", flush=True)

    # One scratch set per bucket, shared by every layer (layers run one at a time).
    from b12x.preparation.types import require_prepared
    scratch = {}
    for m in BUCKETS:
        need = {}
        for li in layers:
            variant = decls[li].variants[m]
            specs = require_prepared(variant, variant.component_id).scratch.scratch_specs()
            for i, sp in enumerate(specs):
                need[i] = (max(need.get(i, (0, sp.dtype))[0], int(sp.shape[0])), sp.dtype)
        cur = shared_scratch.get(m, ())
        scratch[m] = tuple(cur[i] if i < len(cur) and cur[i].numel() >= n_ and cur[i].dtype == dt
                           else torch.empty(n_, dtype=dt, device=dev) for i, (n_, dt) in sorted(need.items()))

    buf = pt.open_shared(create=True)
    os.chmod(pt.PATH, 0o666)
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
    gpool = torch.cuda.graph_pool_handle()
    graphs = {}
    ids_buf.fill_(-1)
    ids_buf[:, 0] = 0
    w_buf[:, 0] = 0.1
    for li in layers:
        for m in BUCKETS:
            binding = fm.bind(decls[li].variants[m], scratch=scratch[m], a=a_buf[:m], experts=experts[li],
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
            with torch.cuda.graph(graph, pool=gpool, stream=stream):
                work()
            graphs[(li, m)] = graph
    ids_buf.fill_(-1)
    torch.cuda.synchronize()
    print(f"captured {len(graphs)} graphs; {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated, "
          f"{torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB free", flush=True)

    base_bucket_of = [0] + [next(m for m in BUCKETS if m >= r) for r in range(1, rows_max + 1)]

    def serve_one(li, rows, bucket_of):
        """The serving step: exactly what the loop below runs for one published sequence."""
        m = bucket_of[rows]
        if m <= pt.SMALL_ROWS:
            graphs[(li, m)].replay()
        else:
            x_buf[:rows].view(-1).copy_(x_host[: rows * H], non_blocking=True)
            xs_buf[:rows].view(-1).copy_(xs_host[: rows * (H // 32)], non_blocking=True)
            ids_buf[:rows].view(-1).copy_(ids_host[: rows * K], non_blocking=True)
            w_buf[:rows].view(-1).copy_(w_host[: rows * K], non_blocking=True)
            if rows < m:
                ids_buf[rows:m].fill_(-1)
            graphs[(li, m)].replay()
            out_host[: rows * H].copy_(out_buf[:rows].view(-1), non_blocking=True)
        torch.cuda.synchronize()
        return m

    # ----- self-test through serve_one(), against freshly read checkpoint bytes -----
    worst = 1.0
    for li in [int(x) for x in a.selftest_layers.split(",") if int(x) in peer and peer[int(x)]]:
        n = len(peer[li])
        for rows in (3, 100):
            gen = torch.Generator(device=dev).manual_seed(1000 * li + rows)
            x = (torch.randn(rows, H, device=dev, generator=gen) * 0.5).to(torch.bfloat16)
            ids = torch.full((rows, K), -1, dtype=torch.int32, device=dev)
            nlive = torch.randint(1, 4, (rows,), device=dev, generator=gen)
            for t in range(rows):
                ids[t, : int(nlive[t])] = torch.randperm(n, device=dev, generator=gen)[: int(nlive[t])].int()
            w = torch.where(ids >= 0, torch.rand(rows, K, device=dev, generator=gen) * 0.3, 0.0)
            xq, xs = quant_mxfp8(x)
            pad = base_bucket_of[rows] if base_bucket_of[rows] <= pt.SMALL_ROWS else rows
            ids_pad = torch.full((pad, K), -1, dtype=torch.int32, device=dev)
            ids_pad[:rows] = ids  # the GB300's _publish masks rows past its count below SMALL_ROWS
            x_host[: rows * H].copy_(xq.reshape(-1).cpu())
            xs_host[: rows * (H // 32)].copy_(xs.reshape(-1).cpu())
            ids_host[: pad * K].copy_(ids_pad.reshape(-1).cpu())
            w_host[: rows * K].copy_(w.float().reshape(-1).cpu())
            serve_one(li, rows, base_bucket_of)
            got = out_host[: rows * H].view(rows, H).float().clone()
            used = sorted({int(e) for e in ids.flatten().tolist() if e >= 0})
            raw = {e: expert_bytes(tensor, li, peer[li][e]) for e in used}  # CPU tensors
            x_hat = (xq.view(torch.float8_e4m3fn).float().view(rows, -1, 32)
                     * torch.exp2(xs.float() - 127.0).unsqueeze(-1)).view(rows, H)
            ref = reference(x_hat, ids, w, raw)
            cos = float(torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0))
            rel = float((got - ref).norm() / ref.norm().clamp_min(1e-9))
            worst = min(worst, cos)
            print(f"selftest layer {li} rows {rows}: cos {cos:.5f} rel {rel:.4f}", flush=True)
    out_host.zero_()
    ids_host.fill_(-1)
    if worst < a.min_cos:
        print(f"SELFTEST FAILED: worst cosine {worst:.5f} < {a.min_cos}", flush=True)
        sys.exit(3)

    print("serving", flush=True)
    last, served, busy, rows_total = int(words[0]), 0, 0.0, 0
    # Per-bucket latency stats (graph replay through sync), dumped every ~2 s; touch STATS_RESET to clear.
    stats_file, stats_reset = f"{LOG_DIR}/peer_stats.json", f"{LOG_DIR}/peer_stats.reset"
    stats, last_dump = {}, 0.0
    bucket_of = base_bucket_of
    bucket_override = f"{LOG_DIR}/peer_buckets.json"
    while True:
        seq = int(words[0])
        if seq == last:
            continue
        if int(header[0]) != seq:
            continue  # header not yet visible for this sequence
        li, rows = int(header[1]), int(header[2])
        if rows > 0:
            t0 = time.perf_counter()
            m = serve_one(li, rows, bucket_of)
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
                        override = json.load(open(bucket_override))
                        bucket_of = list(base_bucket_of)
                        for r, b in override.items():
                            bucket_of[int(r)] = int(b)
                    except (OSError, ValueError):
                        bucket_of = base_bucket_of
                    if os.path.exists(stats_reset):
                        stats.clear()
                        os.remove(stats_reset)
                    with open(stats_file + ".tmp", "w") as f:
                        json.dump({str(k): {"calls": v[0], "mean_us": round(v[1] / v[0] * 1e6, 1),
                                            "mean_rows": round(v[2] / v[0], 1)} for k, v in sorted(stats.items())}, f)
                    os.replace(stats_file + ".tmp", stats_file)
            if served % 2000 == 0:
                print(f"served {served} peer layer calls, mean {busy / served * 1e6:.1f} us, "
                      f"mean rows {rows_total / served:.1f}", flush=True)
        words[1] = seq
        last = seq


if __name__ == "__main__":
    main()
