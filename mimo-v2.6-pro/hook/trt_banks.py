"""TRT-LLM banks for hotsplit (FlashInfer routed MXFP4 weights x MXFP8 activations); every kernel reads HBM.

Measured on MiMo-V2.6-Pro experts in HBM (2026-09-25, bench_trt_vs_marlin.py): TRT-LLM is 1.7-1.8x Marlin for decode
batches and 2-6.4x for prefill batches; untuned heuristics match tuned tactics within 1%, so no autotune is needed.

vLLM's modular TRT wrapper ignores expert_map (it uses contiguous expert-parallel ranges), so each bank calls the routed
kernel directly with bank-local ids and -1 for routes outside the bank:
  hot bank     HBM-resident experts, ids = HMAP[topk]
  staged bank  decode (T * top_k <= slots): the step's Grace experts copied into the shared staging slots (stage_grace),
               ids = slot of each route
  slab bank    prefill: the Grace bank copied through the staging slots one slab at a time; ids = Grace-local ids with
               local_expert_offset = slab start, so the kernel skips routes outside the slab
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _map_ids(IDS, MAP, OUT, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    e = tl.load(IDS + i, mask=m, other=-1)
    v = tl.where(e >= 0, tl.load(MAP + tl.maximum(e, 0).to(tl.int64), mask=m, other=-1), -1)
    tl.store(OUT + i, v.to(tl.int32), mask=m)


def map_ids(topk_ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    out = torch.empty(topk_ids.shape, dtype=torch.int32, device=topk_ids.device)
    n = topk_ids.numel()
    _map_ids[(triton.cdiv(n, 1024),)](topk_ids, table, out, n, BLOCK=1024)
    return out


_ACT = None


def _act() -> int:
    global _ACT
    if _ACT is None:
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import activation_to_flashinfer_int
        _ACT = activation_to_flashinfer_int(MoEActivation.SILU)
    return _ACT


def trt(xq, xs, ids, w_bf16, w13, s13, w2, s2, *, num_experts: int, offset: int, local: int, inter: int,
        out: torch.Tensor) -> torch.Tensor:
    from flashinfer import trtllm_fp4_block_scale_routed_moe
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    trtllm_fp4_block_scale_routed_moe(
        topk_ids=(ids, w_bf16), routing_bias=None, hidden_states=xq, hidden_states_scale=xs,
        gemm1_weights=w13, gemm1_weights_scale=s13, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None,
        gemm1_clamp_limit=None, gemm2_weights=w2, gemm2_weights_scale=s2, gemm2_bias=None,
        output1_scale_scalar=None, output1_scale_gate_scalar=None, output2_scale_scalar=None,
        num_experts=num_experts, top_k=ids.shape[1], n_group=None, topk_group=None, intermediate_size=inter,
        local_expert_offset=offset, local_num_experts=local, routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize, do_finalize=True, enable_pdl=True,
        activation_type=_act(), output=out, tune_max_num_tokens=8192)
    return out


def mxfp8(x: torch.Tensor):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import _mxfp8_e4m3_quantize_impl
    return _mxfp8_e4m3_quantize_impl(x.contiguous(), is_sf_swizzled_layout=False)


class TrtLayer:
    """One MoE layer's GB300 banks. hot: (w13, s13, w2, s2) in HBM; grace: (w13, w2) UVA views + (s13, s2) in HBM."""

    def __init__(self, E: int, inter: int, hot, hot_ids: torch.Tensor, grace, grace_ids: torch.Tensor, stager):
        dev = torch.device("cuda", torch.cuda.current_device())
        self.inter, self.stager = inter, stager
        self.hot, self.n_h = hot, hot_ids.numel()
        self.hmap = torch.full((E,), -1, dtype=torch.int32, device=dev)
        if self.n_h:
            self.hmap[hot_ids] = torch.arange(self.n_h, dtype=torch.int32, device=dev)
        self.grace, self.n_c = grace, grace_ids.numel()
        self.cmap = torch.full((E,), -1, dtype=torch.int32, device=dev)
        if self.n_c:
            from stage_grace import _words
            self.cmap[grace_ids] = torch.arange(self.n_c, dtype=torch.int32, device=dev)
            self.src = tuple(_words(t) for t in grace)       # (w13, w2, s13, s2) as int64 words for the copies
            S = stager.slots
            self.n_c_pad = (self.n_c + S - 1) // S * S

    def forward(self, x, topk_ids, topk_weights, path: str = "auto") -> torch.Tensor:
        T = x.shape[0]
        xq, xs = mxfp8(x)
        w = topk_weights.to(torch.bfloat16)
        out = torch.zeros(T, x.shape[1], dtype=torch.bfloat16, device=x.device)
        if self.n_h:
            w13, s13, w2, s2 = self.hot
            trt(xq, xs, map_ids(topk_ids, self.hmap), w, w13, s13, w2, s2, num_experts=self.n_h, offset=0,
                local=self.n_h, inter=self.inter, out=out)
        if self.n_c:
            g = self.grace_out(xq, xs, topk_ids, w, path)
            out.add_(g)
        return out

    def grace_out(self, xq, xs, topk_ids, w, path: str = "auto") -> torch.Tensor:
        st = self.stager
        S = st.slots
        T = xq.shape[0]
        tmp = torch.zeros(T, xq.shape[1], dtype=torch.bfloat16, device=xq.device)
        staged = path == "staged" or (path == "auto" and topk_ids.numel() <= S)
        mapped = False
        if (path == "auto" and not staged and T <= 64  # past ~64 tokens the check (1.4 ms/layer at 8K) never fits
                and not torch.cuda.is_current_stream_capturing()):
            # Eager prefill: short prompts (tool results) often route to <= S distinct Grace experts; copying just those
            # beats sweeping the whole Grace bank through the slabs. One host sync, eager only.
            st.map(topk_ids, self.cmap)
            mapped = staged = int(st.nused) <= S
        if staged:
            emap = st.stage(topk_ids, self.cmap, self.src, mapped=mapped)
            trt(xq, xs, map_ids(topk_ids, emap), w, st.w13, st.s13, st.w2, st.s2, num_experts=S, offset=0,
                local=S, inter=self.inter, out=tmp)
            return tmp
        cl = map_ids(topk_ids, self.cmap)
        part = torch.empty_like(tmp)
        for start in range(0, self.n_c, S):
            st.stage_rows(start, min(S, self.n_c - start), self.src)
            trt(xq, xs, cl, w, st.w13, st.s13, st.w2, st.s2, num_experts=self.n_c_pad, offset=start, local=S,
                inter=self.inter, out=part)
            tmp.add_(part)
        return tmp
