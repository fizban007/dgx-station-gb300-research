"""Phase 2 offline test: MegaMoE with in-kernel compressed-SF decode vs the stock TMA path, one real layer.

Runs inside the vLLM image with the patched header mounted over deep_gemm's. Builds a layer's hot experts exactly as
DeepseekV4MegaMoEExperts.finalize_weights does, encodes the transformed SF with hook/sf_compress.py, and runs
deep_gemm.fp8_fp4_mega_moe twice per input: raw SF (null table, stock path), then CSF (table pointer passed as
cumulative_local_expert_recv_stats, raw SF tensors poisoned so any stray read shows). Outputs must be bit-identical.

  python test_megamoe_csf.py --layer 0 --rowmap /w/rowmap-mix-h254.json --tokens 1 8 32 128 512 2048
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, "/w")
import sf_compress as sfc  # noqa: E402

I, H, TOPK = 2304, 5120, 6

p = argparse.ArgumentParser()
p.add_argument("--model", default="/model")
p.add_argument("--rowmap", default="/w/rowmap-mix-h254.json")
p.add_argument("--layer", type=int, default=0)
p.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 128, 512, 2048])
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--max-tokens", type=int, default=8192)
p.add_argument("--no-timing", action="store_true")
p.add_argument("--experts", type=int, default=0, help="use only the first N hot experts (smaller footprint)")
a = p.parse_args()

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29533")
dist.init_process_group("gloo", rank=0, world_size=1)
dev = torch.device("cuda", 0)
torch.cuda.set_device(dev)
from vllm.third_party import deep_gemm  # noqa: E402
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs  # noqa: E402

index = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
headers = {}


def get(name):
    path = os.path.join(a.model, index[name])
    if path not in headers:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            headers[path] = (8 + n, json.loads(f.read(n)))
    base, hdr = headers[path]
    meta = hdr[name]
    s, e = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + s)
        return torch.from_numpy(np.frombuffer(f.read(e - s), dtype=np.uint8).reshape(meta["shape"]).copy())


hot = json.load(open(a.rowmap))["layers"][str(a.layer)]["hot"]
if a.experts:
    hot = hot[:a.experts]
E = len(hot)
t0 = time.time()
w13 = torch.empty(E, 2 * I, H // 2, dtype=torch.uint8, device=dev)
s13 = torch.empty(E, 2 * I, H // 32, dtype=torch.uint8, device=dev)
w2 = torch.empty(E, H, I // 2, dtype=torch.uint8, device=dev)
s2 = torch.empty(E, H, I // 32, dtype=torch.uint8, device=dev)
for i, e in enumerate(hot):
    pre = f"layers.{a.layer}.ffn.experts.{e}"
    w13[i, :I].copy_(get(f"{pre}.w1.weight")); w13[i, I:].copy_(get(f"{pre}.w3.weight"))
    s13[i, :I].copy_(get(f"{pre}.w1.scale")); s13[i, I:].copy_(get(f"{pre}.w3.scale"))
    w2[i].copy_(get(f"{pre}.w2.weight")); s2[i].copy_(get(f"{pre}.w2.scale"))
print(f"loaded layer {a.layer}: {E} hot experts in {time.time() - t0:.0f}s", flush=True)


def ue8m0(u8):
    return (u8.to(torch.int32) << 23).view(torch.float32)


sf13 = deep_gemm.transform_sf_into_required_layout(ue8m0(s13).contiguous(), 2 * I, H, (1, 32), E)
sf2 = deep_gemm.transform_sf_into_required_layout(ue8m0(s2).contiguous(), H, I, (1, 32), E)
del s13, s2
l1, l2 = deep_gemm.transform_weights_for_mega_moe((w13.view(torch.int8), sf13), (w2.view(torch.int8), sf2))
del w13, sf13, sf2, w2
torch.cuda.empty_cache()
print("L1 SF", tuple(l1[1].shape), l1[1].stride(), "L2 SF", tuple(l2[1].shape), l2[1].stride(), flush=True)

c1, c2 = sfc.encode(l1[1]), sfc.encode(l2[1])
print(f"encoded: L1 {c1.nbytes() / 2**20:.1f} MiB ({c1.overflow.shape[0]} spill), "
      f"L2 {c2.nbytes() / 2**20:.1f} MiB ({c2.overflow.shape[0]} spill)", flush=True)
table64 = torch.tensor([c1.slots.data_ptr(), c1.overflow.data_ptr(), c2.slots.data_ptr(), c2.overflow.data_ptr()],
                       dtype=torch.int64)
table = torch.zeros(max(E, 8), dtype=torch.int32, device=dev)
table[:8].copy_(table64.view(torch.int32))
# Poisoned SF for the CSF runs: E8M0 0 = 2^-127 would zero (or wildly change) any output that reads it.
poison1 = torch.zeros_like(l1[1])
poison2 = torch.zeros_like(l2[1])
l1_csf, l2_csf = (l1[0], poison1), (l2[0], poison2)

buf = deep_gemm.get_symm_buffer_for_mega_moe(dist.group.WORLD, E, a.max_tokens, TOPK, H, I)
ok = True
for T in a.tokens:
    g = torch.Generator(device=dev).manual_seed(T)
    x = (torch.randn(T, H, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev, generator=g)[:TOPK] for _ in range(T)]).to(torch.int64)
    wts = torch.softmax(torch.randn(T, TOPK, device=dev, generator=g), -1)
    outs = {}

    def run(mode):
        prepare_megamoe_inputs(x, wts, ids, buf.x[:T], buf.x_sf[:T], buf.topk_idx[:T], buf.topk_weights[:T])
        y = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
        if mode == "raw":
            deep_gemm.fp8_fp4_mega_moe(y, l1, l2, buf)
        else:
            deep_gemm.fp8_fp4_mega_moe(y, l1_csf, l2_csf, buf, cumulative_local_expert_recv_stats=table)
        return y

    raws = [run("raw") for _ in range(a.repeats)]
    csfs = [run("csf") for _ in range(a.repeats)]
    torch.cuda.synchronize()
    raw_rep = all(torch.equal(r.view(torch.int16), raws[0].view(torch.int16)) for r in raws)
    same = all(torch.equal(c.view(torch.int16), raws[0].view(torch.int16)) for c in csfs)
    finite = bool(torch.isfinite(raws[0]).all()) and float(raws[0].float().abs().mean()) > 0
    diff = max((c.float() - raws[0].float()).abs().max().item() for c in csfs)
    ok &= same and finite
    line = f"T={T}: raw repeatable={raw_rep} csf bit-identical={same} (max diff {diff:.3g}) |y|={raws[0].float().abs().mean():.3g}"
    if not a.no_timing:
        times = {}
        for mode in ("raw", "csf"):
            for _ in range(3):
                run(mode)
            torch.cuda.synchronize()
            st = time.perf_counter()
            for _ in range(20):
                run(mode)
            torch.cuda.synchronize()
            times[mode] = (time.perf_counter() - st) / 20 * 1e6
        line += f"  raw {times['raw']:.0f} us  csf {times['csf']:.0f} us (incl. staging)"
    print(line, flush=True)
print("PASS" if ok else "FAIL")
dist.destroy_process_group()
