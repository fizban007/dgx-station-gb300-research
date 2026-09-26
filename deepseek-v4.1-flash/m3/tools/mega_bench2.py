"""Phase 1: DeepGEMM MegaMoE vs FlashInfer TRT-LLM at DeepSeek-V4.1 geometry on one GB300.

MegaMoE weights are prepared exactly as vLLM v0.30.0's DeepseekV4MegaMoEExperts.finalize_weights
(pad intermediate 2304 -> 2560, UE8M0 scales to DeepGEMM layout, transform_weights_for_mega_moe);
inputs are staged with vLLM's prepare_megamoe_inputs. FlashInfer runs the same MXFP4 experts
through vLLM's FLASHINFER_TRTLLM_MXFP4_MXFP8 conversion. Cases:
  all:    E=384 experts, every route live.
  hot:    E=295 experts (a hot subset), routes to the 89 cold experts masked to -1.
"""
import json, os, statistics, sys
import torch
import vllm._custom_ops  # noqa: F401  (registers torch.ops._C.silu_and_mul)
import torch.distributed as dist

E, H, I, K, LIMIT = 384, 5120, 2304, 6, 10.0
HOT = 295
PAD = os.environ.get("PAD", "1") == "1"   # v0.30.0 pads intermediate 2304 -> 2560; later nightlies do not
SHARED = os.environ.get("SHARED", "1") == "1"
TOKENS = [int(v) for v in os.environ.get("TOKENS", "1 4 8 16 32 64 256 1024 4096 8192").split()]
MAXT = max(TOKENS)
dev = torch.device("cuda", 0)
torch.cuda.set_device(dev)
dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29511", rank=0, world_size=1)
from vllm.utils.deep_gemm import _import_deep_gemm
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs
dg = _import_deep_gemm()
g = torch.Generator(device=dev).manual_seed(0)

w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev, generator=g)
w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev, generator=g)
s13 = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev, generator=g)
s2 = torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8, device=dev, generator=g)


def ue8m0(sf):
    return (sf.to(torch.int32) << 23).view(torch.float32)


def mega_weights(n):
    """vLLM v0.30.0 finalize_weights for the first n experts."""
    pad = (I + 511) // 512 * 512 if PAD else I
    a = w13[:n].unflatten(1, (2, I)); a = torch.nn.functional.pad(a, (0, 0, 0, pad - I)).flatten(1, 2)
    sa = s13[:n].unflatten(1, (2, I)); sa = torch.nn.functional.pad(sa, (0, 0, 0, pad - I)).flatten(1, 2)
    b = torch.nn.functional.pad(w2[:n], (0, (pad - I) // 2))
    sb = torch.nn.functional.pad(s2[:n], (0, (pad - I) // 32))
    fa = dg.transform_sf_into_required_layout(ue8m0(sa).contiguous(), 2 * pad, H, (1, 32), n)
    fb = dg.transform_sf_into_required_layout(ue8m0(sb).contiguous(), H, pad, (1, 32), n)
    l1, l2 = dg.transform_weights_for_mega_moe((a.view(torch.int8).contiguous(), fa), (b.view(torch.int8).contiguous(), fb))
    return l1, l2, pad


def mega_shared(width):
    """Shared expert as vLLM prepares it for MegaMoE fusion: FP8 [32,32] blocks -> 1x32 -> DeepGEMM layout."""
    gu = (torch.randn(2 * width, H, device=dev, generator=g) * 0.02).to(torch.float8_e4m3fn)
    dn = (torch.randn(H, width, device=dev, generator=g) * 0.02).to(torch.float8_e4m3fn)
    def sf(mn, k):
        blocks = torch.randint(120, 126, ((mn + 31) // 32, (k + 31) // 32), dtype=torch.uint8, device=dev, generator=g)
        f = ue8m0(blocks).repeat_interleave(32, 0).repeat_interleave(1, 1)[:mn]
        f = f.repeat_interleave(1, 1)[:, : k // 32].contiguous()
        return dg.transform_sf_into_required_layout(f.unsqueeze(0), mn, k, (1, 32), 1).squeeze(0)
    return dg.transform_weights_for_mega_moe((gu, sf(2 * width, H)), (dn, sf(H, width)))


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    t = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); graph.replay(); b.record(); b.synchronize(); t.append(a.elapsed_time(b) * 1000)
    graph.reset()
    return statistics.median(t)


# FlashInfer TRT-LLM on the same experts.
fi_ok = True
try:
    from flashinfer import trtllm_fp4_block_scale_routed_moe
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend, convert_weight_to_mxfp4_moe_kernel_format
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import _mxfp8_e4m3_quantize_impl
    fw13, fw2, fs13, fs2, _, _ = convert_weight_to_mxfp4_moe_kernel_format(
        mxfp4_backend=Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8, layer=None,
        w13_weight=w13.clone(), w2_weight=w2.clone(), w13_weight_scale=s13.clone(), w2_weight_scale=s2.clone(),
        _cache_permute_indices={})
    clamp = torch.full((E,), LIMIT, dtype=torch.float32, device=dev)
except Exception as exc:  # report and continue with MegaMoE only
    fi_ok = False
    print(json.dumps({"flashinfer": f"unavailable: {type(exc).__name__}: {exc}"}), flush=True)


_SHARED_FI = None


def shared_fi():
    """Shared expert weights in FlashInfer MXFP8 layout (weights as [N, K], swizzled scales)."""
    global _SHARED_FI
    if _SHARED_FI is None:
        import flashinfer
        gu = torch.randn(2 * I, H, device=dev, generator=g, dtype=torch.bfloat16) * 0.02
        dn = torch.randn(H, I, device=dev, generator=g, dtype=torch.bfloat16) * 0.02
        gq, gsf = flashinfer.mxfp8_quantize(gu, True)
        dq, dsf = flashinfer.mxfp8_quantize(dn, True)
        _SHARED_FI = (gq.t(), gsf, dq.t(), dsf)
    return _SHARED_FI


def fi_call(xq, xs, ids, wts, out, n):
    trtllm_fp4_block_scale_routed_moe(
        topk_ids=(ids, wts), routing_bias=None, hidden_states=xq, hidden_states_scale=xs,
        gemm1_weights=fw13[:n], gemm1_weights_scale=fs13[:n], gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None,
        gemm1_clamp_limit=clamp[:n], gemm2_weights=fw2[:n], gemm2_weights_scale=fs2[:n], gemm2_bias=None,
        output1_scale_scalar=None, output1_scale_gate_scalar=None, output2_scale_scalar=None, num_experts=n,
        top_k=K, n_group=None, topk_group=None, intermediate_size=I, local_expert_offset=0, local_num_experts=n,
        routed_scaling_factor=None, routing_method_type=1, do_finalize=True, enable_pdl=True, output=out,
        tune_max_num_tokens=MAXT)


results = []
for case, n in (("all", E), ("hot", HOT)):
    l1, l2, pad = mega_weights(n)
    sh1 = sh2 = None
    if SHARED:
        sh1, sh2 = mega_shared(pad)
    buf = dg.get_symm_buffer_for_mega_moe(dist.group.WORLD, n, MAXT, K, H, pad,
                                          num_shared_experts=1 if SHARED else 0)
    for m in TOKENS:
        x = torch.randn(m, H, device=dev, generator=g, dtype=torch.bfloat16) * 0.5
        ids = torch.randint(0, E, (m, K), device=dev, generator=g)
        wts = torch.softmax(torch.rand(m, K, device=dev, generator=g), -1)
        if case == "hot":
            # Routes to experts >= HOT are cold: 2.3% of routes, as with profile placement.
            cold = torch.rand(m, K, device=dev, generator=g) < 0.023
            ids = torch.where(cold, torch.randint(HOT, E, (m, K), device=dev, generator=g), ids % HOT)
            mids = torch.where(ids >= HOT, -1, ids)
        else:
            mids = ids
        y = torch.empty(m, H, device=dev, dtype=torch.bfloat16)

        sbm = (dg.get_block_m_for_mega_moe(1, n, buf.num_max_tokens_per_rank, m, K, "fp8xfp4")
               if SHARED else None)

        def mega():
            prepare_megamoe_inputs(x, wts, mids.to(torch.int64), buf.x[:m], buf.x_sf[:m], buf.topk_idx[:m],
                                   buf.topk_weights[:m], is_padding=None,
                                   shared_x_sf=buf.shared_l1_acts_sf if SHARED else None, shared_block_m=sbm)
            if SHARED:
                dg.fp8_fp4_mega_moe(y, l1, l2, buf, shared_l1_weights=sh1, shared_l2_weights=sh2,
                                    activation_clamp=LIMIT, fast_math=True)
            else:
                dg.fp8_fp4_mega_moe(y, l1, l2, buf, activation_clamp=LIMIT, fast_math=True)
        row = {"case": case, "experts": n, "tokens": m}
        try:
            row["mega_us"] = round(timed(mega), 1)
        except Exception as exc:
            row["mega_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        if fi_ok:
            xq, xs = _mxfp8_e4m3_quantize_impl(x, is_sf_swizzled_layout=False)
            out = torch.empty(m, H, device=dev, dtype=torch.bfloat16)
            fids = torch.where(ids >= n, -1, ids).to(torch.int32) if case == "hot" else ids.to(torch.int32)
            fw = wts.to(torch.bfloat16)
            if SHARED:
                import flashinfer
                sw = shared_fi()
                act = torch.empty(m, I, device=dev, dtype=torch.bfloat16)
                best = None
                for backend in ("cute-dsl", "cutlass", "trtllm", "cudnn"):
                    def fi_all(backend=backend):
                        # Upstream's shared expert: MXFP8 quantize, block-scaled GEMM, fused SiLU-mul,
                        # quantize, GEMM, add.
                        fi_call(xq, xs, fids, fw, out, n)
                        aq, asf = flashinfer.mxfp8_quantize(x, True)
                        hgu = flashinfer.mm_mxfp8(aq, sw[0], asf, sw[1], out_dtype=torch.bfloat16, backend=backend)
                        torch.ops._C.silu_and_mul(act, hgu)
                        bq, bsf = flashinfer.mxfp8_quantize(act, True)
                        out.add_(flashinfer.mm_mxfp8(bq, sw[2], bsf, sw[3], out_dtype=torch.bfloat16, backend=backend))
                    try:
                        us = timed(fi_all)
                    except Exception:
                        continue
                    if best is None or us < best[0]:
                        best = (us, backend)
                row["fi_us"], row["fi_shared_backend"] = round(best[0], 1), best[1]
                row["fi_routed_only_us"] = round(timed(lambda: fi_call(xq, xs, fids, fw, out, n)), 1)
            else:
                row["fi_us"] = round(timed(lambda: fi_call(xq, xs, fids, fw, out, n)), 1)
            if "mega_us" in row and not SHARED:
                mega(); fi_call(xq, xs, fids, fw, out, n); torch.cuda.synchronize()
                row["rel_diff"] = round(float((y.float() - out.float()).norm() / out.float().norm().clamp_min(1e-9)), 4)
        results.append(row)
        print(json.dumps(row), flush=True)
    del l1, l2, buf
    torch.cuda.empty_cache()
json.dump(results, open(os.environ.get("OUT", "/megamoe/phase1.json"), "w"), indent=1)
