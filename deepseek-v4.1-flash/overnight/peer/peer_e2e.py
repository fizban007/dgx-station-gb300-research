"""GB300-side check of the peer tier against the running 6000 sidecar.

Compares the peer's cold contribution for layer 0 with the same Triton kernel
run locally on the GB300, then times one layer (hot stand-in + peer round trip)
inside a CUDA graph at several batch sizes.
"""
import json, os, statistics, struct, sys, torch
sys.path.insert(0, "/home/jasonc/vllm-karmic/vllm/model_executor/layers/fused_moe/experts")
import peer_tier as pt
from cold_moe_triton import ColdExperts

dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
model = "/home/jasonc/models/DeepSeek-V4.1-Flash"
profile = json.load(open("/home/jasonc/ds41f-exp/profiles/hot-calib1-u305.json"))
index = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]
def tensor(name):
    f = f"{model}/{index[name]}"
    with open(f, "rb") as h:
        size = struct.unpack("<Q", h.read(8))[0]; hdr = json.loads(h.read(size))
        s, e = hdr[name]["data_offsets"]; h.seek(8 + size + s); data = h.read(e - s)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).view(hdr[name]["shape"])
layer = 0
hot = set(profile[f"language_model.model.layers.{layer}.ffn.experts"]["hot"])
cold = [e for e in range(384) if e not in hot]
w13 = torch.stack([torch.cat([tensor(f"layers.{layer}.ffn.experts.{e}.w1.weight"), tensor(f"layers.{layer}.ffn.experts.{e}.w3.weight")]) for e in cold]).to(dev)
s13 = torch.stack([torch.cat([tensor(f"layers.{layer}.ffn.experts.{e}.w1.scale"), tensor(f"layers.{layer}.ffn.experts.{e}.w3.scale")]) for e in cold]).to(dev)
w2 = torch.stack([tensor(f"layers.{layer}.ffn.experts.{e}.w2.weight") for e in cold]).to(dev)
s2 = torch.stack([tensor(f"layers.{layer}.ffn.experts.{e}.w2.scale") for e in cold]).to(dev)
H = len(hot)
local = ColdExperts(w13, s13, w2, s2, cold_start=H, limit=10.0, gate_first=True)
peer = pt.PeerTier(dev)
g = torch.Generator(device=dev).manual_seed(0)
for tokens in (1, 4, 8, 32):
    x = torch.randn(tokens, 5120, device=dev, generator=g)
    xs = torch.full((tokens, 160), 124, dtype=torch.uint8, device=dev)
    xq = (x / 2 ** (124 - 127)).clamp(-448, 448).to(torch.float8_e4m3fn)
    ids = torch.randint(0, H, (tokens, 6), dtype=torch.int32, device=dev, generator=g)
    ids[:, 0] = H + torch.randint(0, len(cold), (tokens,), dtype=torch.int32, device=dev, generator=g)
    wts = torch.rand(tokens, 6, device=dev, generator=g).to(torch.bfloat16)
    ref = torch.zeros(tokens, 5120, dtype=torch.bfloat16, device=dev)
    local.add_to(ref, xq, xs, ids, wts)
    out = torch.empty(tokens, 5120, dtype=torch.bfloat16, device=dev)
    peer.run(out, xq, xs, ids, wts, cold_start=H, layer=layer, hot_call=lambda: out.zero_())
    torch.cuda.synchronize()
    err = float((out.float() - ref.float()).norm() / ref.float().norm())
    timeouts = int(peer.words[2].item())
    hot_buf = torch.empty(300 * 2**20, dtype=torch.uint8, device=dev)
    def step():
        peer.run(out, xq, xs, ids, wts, cold_start=H, layer=layer,
                 hot_call=lambda: out.copy_(hot_buf[: out.numel() * 2].view(torch.bfloat16).view_as(out)))
    graph = torch.cuda.CUDAGraph()
    step(); torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(10):
            step()
    times = []
    for _ in range(20):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); graph.replay(); b.record(); b.synchronize(); times.append(a.elapsed_time(b) * 100)
    print(f"T={tokens:2d} cold routes {tokens}: rel err vs local {err:.2e}, timeouts {timeouts}, "
          f"graph step {statistics.median(times):.1f} us per layer", flush=True)
