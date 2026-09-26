"""Does FlashInfer TRT-LLM (as the hook calls it) match DeepSeek's reference expert math on real layer-0 experts?"""
import json, os, struct, torch
from flashinfer import trtllm_fp4_block_scale_routed_moe
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend, convert_weight_to_mxfp4_moe_kernel_format
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import _mxfp8_e4m3_quantize_impl as quant
M = "/model"; H, I, K, LIM = 5120, 2304, 6, 10.0
idx = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]; hd = {}
def t(n):
    f = f"{M}/{idx[n]}"
    if f not in hd:
        with open(f, "rb") as fh: s = struct.unpack("<Q", fh.read(8))[0]; hd[f] = (8 + s, json.loads(fh.read(s)))
    b, h = hd[f]; m = h[n]; a, e = m["data_offsets"]
    with open(f, "rb") as fh: fh.seek(b + a); d = fh.read(e - a)
    return torch.frombuffer(bytearray(d), dtype=torch.uint8).view(m["shape"])
dev = torch.device("cuda"); E = 8; experts = [13, 19, 93, 376, 233, 244, 202, 218]
w13 = torch.stack([torch.cat([t(f"layers.0.ffn.experts.{e}.w1.weight"), t(f"layers.0.ffn.experts.{e}.w3.weight")]) for e in experts]).to(dev)
s13 = torch.stack([torch.cat([t(f"layers.0.ffn.experts.{e}.w1.scale"), t(f"layers.0.ffn.experts.{e}.w3.scale")]) for e in experts]).to(dev)
w2 = torch.stack([t(f"layers.0.ffn.experts.{e}.w2.weight") for e in experts]).to(dev)
s2 = torch.stack([t(f"layers.0.ffn.experts.{e}.w2.scale") for e in experts]).to(dev)
LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
def dq(w, s): return torch.stack([LUT[(w & 15).long()], LUT[(w >> 4).long()]], -1).flatten(-2) * torch.exp2(s.float() - 127).repeat_interleave(32, -1)
T = 16
torch.manual_seed(0)
x = (torch.randn(T, H, device=dev) * 0.5).to(torch.bfloat16)
xq, xs = quant(x, is_sf_swizzled_layout=False)
xd = (xq.float() * torch.exp2(xs.float() - 127).repeat_interleave(32, 1))
ids = torch.full((T, K), -1, dtype=torch.int32, device=dev); ids[:, 0] = torch.arange(T, device=dev) % E
wts = torch.zeros(T, K, device=dev); wts[:, 0] = 1.0
def dense(gate_first):
    out = torch.zeros(T, H, device=dev)
    for i in range(T):
        e = int(ids[i, 0]); h = dq(w13[e], s13[e]) @ xd[i]
        g, u = (h[:I], h[I:]) if gate_first else (h[I:], h[:I])
        out[i] = dq(w2[e], s2[e]) @ (torch.nn.functional.silu(g.clamp(max=LIM)) * u.clamp(-LIM, LIM))
    return out
fw13, fw2, fs13, fs2, _, _ = convert_weight_to_mxfp4_moe_kernel_format(mxfp4_backend=Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8, layer=None,
    w13_weight=w13.clone(), w2_weight=w2.clone(), w13_weight_scale=s13.clone(), w2_weight_scale=s2.clone(), _cache_permute_indices={})
out = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
trtllm_fp4_block_scale_routed_moe(topk_ids=(ids, wts.to(torch.bfloat16)), routing_bias=None, hidden_states=xq, hidden_states_scale=xs,
    gemm1_weights=fw13, gemm1_weights_scale=fs13, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None,
    gemm1_clamp_limit=torch.full((E,), LIM, device=dev), gemm2_weights=fw2, gemm2_weights_scale=fs2, gemm2_bias=None,
    output1_scale_scalar=None, output1_scale_gate_scalar=None, output2_scale_scalar=None, num_experts=E, top_k=K,
    n_group=None, topk_group=None, intermediate_size=I, local_expert_offset=0, local_num_experts=E, routed_scaling_factor=None,
    routing_method_type=RoutingMethodType.Renormalize, do_finalize=True, enable_pdl=True, output=out, tune_max_num_tokens=64)
torch.cuda.synchronize()
cos = lambda a, b: round(float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0)), 5)
print("TRT vs dense(gate=w1)", cos(out, dense(True)), "| TRT vs dense(gate=w3)", cos(out, dense(False)))
