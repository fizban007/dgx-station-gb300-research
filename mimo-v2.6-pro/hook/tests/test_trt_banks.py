"""TRT banks (hot HBM + staged/slab Grace) vs one TRT call over all experts in HBM, on real MiMo-V2.6-Pro experts.
Run in the vLLM image, checkpoint at /model, on the GB300."""
import json
import os
import struct
import sys

import torch

sys.path.insert(0, "/w")
from stage_grace import GraceStager  # noqa: E402
from trt_banks import TrtLayer, mxfp8, trt  # noqa: E402

dev = torch.device("cuda", 0)
torch.cuda.set_device(dev)
E, H, I, K, LAYER = 96, 6144, 2048, 8, 5
index = json.load(open("/model/model.safetensors.index.json"))["weight_map"]
headers = {}


def tensor(name):
    f = os.path.join("/model", index[name])
    if f not in headers:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            headers[f] = (8 + n, json.loads(fh.read(n)))
    base, h = headers[f]
    s, e = h[name]["data_offsets"]
    with open(f, "rb") as fh:
        fh.seek(base + s)
        return torch.frombuffer(bytearray(fh.read(e - s)), dtype=torch.uint8).view(h[name]["shape"])


parts = {k: [] for k in ("w13", "s13", "w2", "s2")}
for e in range(E):
    p = f"model.layers.{LAYER}.mlp.experts.{e}"
    parts["w13"].append(torch.cat([tensor(f"{p}.gate_proj.weight"), tensor(f"{p}.up_proj.weight")]))
    parts["s13"].append(torch.cat([tensor(f"{p}.gate_proj.weight_scale"), tensor(f"{p}.up_proj.weight_scale")]))
    parts["w2"].append(tensor(f"{p}.down_proj.weight"))
    parts["s2"].append(tensor(f"{p}.down_proj.weight_scale"))
raw = {k: torch.stack(v).to(dev) for k, v in parts.items()}
del parts
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (  # noqa: E402
    Mxfp4MoeBackend, convert_weight_to_mxfp4_moe_kernel_format)
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor as uva  # noqa: E402

with set_current_vllm_config(VllmConfig()):
    w13, w2, s13, s2, _, _ = convert_weight_to_mxfp4_moe_kernel_format(
        mxfp4_backend=Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8, layer=None, w13_weight=raw["w13"],
        w2_weight=raw["w2"], w13_weight_scale=raw["s13"], w2_weight_scale=raw["s2"], _cache_permute_indices={})
del raw
torch.cuda.empty_cache()
print("TRT layout:", tuple(w13.shape), w13.dtype, tuple(s13.shape), s13.dtype, tuple(w2.shape), tuple(s2.shape))
g = torch.Generator(device=dev).manual_seed(0)
perm = torch.randperm(E, device=dev, generator=g)
hot_ids, grace_ids = perm[:32].sort().values, perm[32:].sort().values
hot = tuple(t.index_select(0, hot_ids).contiguous() for t in (w13, s13, w2, s2))
host = [torch.empty((grace_ids.numel(),) + tuple(t.shape[1:]), dtype=t.dtype, device="cpu") for t in (w13, w2)]
gw13, gw2 = uva(host[0]), uva(host[1])
gw13.copy_(w13.index_select(0, grace_ids)); gw2.copy_(w2.index_select(0, grace_ids))
gs13, gs2 = s13.index_select(0, grace_ids).contiguous(), s2.index_select(0, grace_ids).contiguous()
grace = (gw13, gw2, gs13, gs2)
layers = {}
for S in (64, 32):
    st = GraceStager(gw13, gw2, gs13, gs2, slots=S, num_experts=E)
    layers[S] = TrtLayer(E, I, hot, hot_ids, grace, grace_ids, st)


def reference(x, topk, wts):
    xq, xs = mxfp8(x)
    out = torch.zeros(x.shape[0], H, dtype=torch.bfloat16, device=dev)
    return trt(xq, xs, topk.int(), wts.to(torch.bfloat16), w13, s13, w2, s2, num_experts=E, offset=0, local=E,
               inter=I, out=out)


def compare(tag, got, ref):
    a, b = got.float(), ref.float()
    cos = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    rel = float((a - b).norm() / b.norm())
    ratio = float(a.norm() / b.norm())
    ok = cos > 0.9999 and abs(ratio - 1) < 1e-3
    print(f"{tag}: cos {cos:.6f} rel {rel:.2e} norm ratio {ratio:.5f} {'OK' if ok else 'FAIL'}", flush=True)
    assert ok, tag


with set_current_vllm_config(VllmConfig()):
    for T in (1, 4, 16):
        x = (torch.randn(T, H, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        topk = torch.stack([torch.randperm(E, device=dev, generator=g)[:K] for _ in range(T)]).int()
        wts = torch.softmax(torch.randn(T, K, device=dev, generator=g), -1)
        ref = reference(x, topk, wts)
        compare(f"staged T={T:4d}", layers[64].forward(x, topk, wts, path="staged"), ref)
        compare(f"slab   T={T:4d}", layers[32].forward(x, topk, wts, path="slab"), ref)
    for T in (20, 300, 2048):
        x = (torch.randn(T, H, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        topk = torch.stack([torch.randperm(E, device=dev, generator=g)[:K] for _ in range(T)]).int()
        wts = torch.softmax(torch.randn(T, K, device=dev, generator=g), -1)
        r = reference(x, topk, wts)
        compare(f"slab   T={T:4d}", layers[32].forward(x, topk, wts, path="slab"), r)
        compare(f"auto   T={T:4d}", layers[64].forward(x, topk, wts, path="auto"), r)
    # graph capture of the staged decode path, replayed with new routing
    T = 16
    x = (torch.randn(T, H, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    topk = torch.stack([torch.randperm(E, device=dev, generator=g)[:K] for _ in range(T)]).int()
    wts = torch.softmax(torch.randn(T, K, device=dev, generator=g), -1)
    layers[64].forward(x, topk, wts, path="staged"); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream(dev)
    with torch.cuda.graph(gr, stream=s):
        out = layers[64].forward(x, topk, wts, path="staged")
    for trial in range(3):
        topk.copy_(torch.stack([torch.randperm(E, device=dev, generator=g)[:K] for _ in range(T)]).int())
        x.copy_((torch.randn(T, H, device=dev, generator=g) * 0.5).to(torch.bfloat16))
        gr.replay(); torch.cuda.synchronize()
        compare(f"staged graph replay {trial}", out, reference(x, topk, wts))
print("TRT BANKS OK")
