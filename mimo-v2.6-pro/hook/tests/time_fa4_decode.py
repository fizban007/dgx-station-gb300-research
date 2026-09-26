"""FA4 DiffKV decode time for one query token vs context: FP8 KV forced single split vs BF16 KV auto splits (graphed)."""
import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func
dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
BS, HKV, HQ, DK, DV = 16, 8, 128, 192, 128
g = torch.Generator(device=dev).manual_seed(0)
def timed(fn, reps=50):
    fn(); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(dev)
    with torch.cuda.graph(gr, stream=s):
        fn()
    gr.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps): gr.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / reps
for ctx in (8192, 65536, 262144, 1048576):
    nblk = ctx // BS
    res = {}
    for kind in ("fp8-auto", "bf16-auto"):
        dt = torch.float8_e4m3fn if kind.startswith("fp8") else torch.bfloat16
        cache = (torch.randn(nblk, HKV, BS, DK + DV, device=dev, generator=g) * 0.5).to(dt)
        kc, vc = cache.transpose(1, 2).split(DK, dim=-1)
        q = (torch.randn(1, HQ, DK, device=dev, generator=g) * 0.5).to(dt)
        out = torch.empty(1, HQ, DV, dtype=torch.bfloat16, device=dev)
        bt = torch.arange(nblk, dtype=torch.int32, device=dev)[None]
        ones = torch.ones(1, HKV, dtype=torch.float32, device=dev)
        kw = dict(q_descale=ones, k_descale=ones, v_descale=ones) if kind.startswith("fp8") else {}
        splits = 1 if kind == "fp8-split1" else 0
        cu = torch.tensor([0, 1], dtype=torch.int32, device=dev); sk = torch.tensor([ctx], dtype=torch.int32, device=dev)
        fn = lambda: flash_attn_varlen_func(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=sk,
                                           max_seqlen_k=ctx, softmax_scale=DK ** -0.5, causal=True, block_table=bt,
                                           fa_version=4, num_splits=splits, **kw)
        res[kind] = timed(fn)
        del cache, kc, vc
        torch.cuda.empty_cache()
    print(f"ctx {ctx:8d}: per full-attn layer  FP8 auto {res['fp8-auto']:8.1f} us   BF16 auto {res['bf16-auto']:8.1f} us"
          f"   -> x10 full layers/token: {res['fp8-auto'] * 10 / 1e3:6.2f} ms vs {res['bf16-auto'] * 10 / 1e3:6.2f} ms", flush=True)
