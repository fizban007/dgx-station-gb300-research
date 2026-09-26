"""FA4 DiffKV (192/128) decode over an FP8 paged KV cache the size of the MiMo-Pro FP8 pool, with the request's blocks
at low ids vs at the tail of the pool (the AGENTS.md big-pid test), against a BF16-cache reference on the same values.
Mirrors vLLM's FlashAttentionDiffKVImpl call: cache [NB, H, bs, 192+128] -> transpose -> split; q/k/v descales."""
import os
import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

dev = torch.device("cuda", 0)
torch.cuda.set_device(dev)
NB = int(os.environ.get("NB", "164715"))
BS, HKV, HQ, DK, DV = 16, 8, 128, 192, 128
CTX = int(os.environ.get("CTX", "60000"))
nblk = (CTX + BS - 1) // BS
g = torch.Generator(device=dev).manual_seed(0)
cache = torch.empty(NB, HKV, BS, DK + DV, dtype=torch.float8_e4m3fn, device=dev)
print(f"FP8 cache {cache.numel() * cache.element_size() / 1e9:.2f} GB, {NB} blocks; request {CTX} tokens = {nblk} blocks")


def fill(ids):
    vals = (torch.randn(ids.numel(), HKV, BS, DK + DV, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
    cache[ids.long()] = vals
    return vals


def run(block_ids, window=None, sinks=None, kv_dtype="fp8"):
    kc, vc = cache.transpose(1, 2).split(DK, dim=-1)
    if kv_dtype == "bf16":  # reference: the same values in a compact BF16 cache, low ids
        compact = cache[block_ids.long()].to(torch.bfloat16)
        kc, vc = compact.transpose(1, 2).split(DK, dim=-1)
        bt = torch.arange(nblk, dtype=torch.int32, device=dev)[None]
    else:
        bt = block_ids[None].int()
    q = (torch.randn(1, HQ, DK, device=dev, generator=torch.Generator(device=dev).manual_seed(1)) * 0.5)
    qd = q.to(torch.float8_e4m3fn) if kv_dtype == "fp8" else q.to(torch.float8_e4m3fn).to(torch.bfloat16)
    out = torch.empty(1, HQ, DV, dtype=torch.bfloat16, device=dev)
    ones = torch.ones(1, HKV, dtype=torch.float32, device=dev)
    kw = dict(q_descale=ones, k_descale=ones, v_descale=ones) if kv_dtype == "fp8" else {}
    flash_attn_varlen_func(q=qd, k=kc, v=vc, out=out, cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=dev),
                           max_seqlen_q=1, seqused_k=torch.tensor([CTX], dtype=torch.int32, device=dev),
                           max_seqlen_k=CTX, softmax_scale=DK ** -0.5, causal=True, alibi_slopes=None,
                           window_size=window, block_table=bt, softcap=0.0, fa_version=4, num_splits=int(os.environ.get("SPLITS", "0")),
                           s_aux=sinks, **kw)
    torch.cuda.synchronize()
    return out.float()


low = torch.arange(nblk, dtype=torch.int32, device=dev)
high = torch.arange(NB - nblk, NB, dtype=torch.int32, device=dev)
fill(low)
cache[high.long()] = cache[low.long()]   # identical values at the tail
sinks = torch.randn(HQ, device=dev, generator=g).to(torch.bfloat16)
for name, window, s in (("full attention", None, None), ("SWA 128 + sinks", [127, 0], sinks)):
    ref = run(low, window, s, kv_dtype="bf16")
    for where, ids in (("low ids", low), ("tail ids", high)):
        try:
            got = run(ids, window, s)
            cos = float(torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0))
            print(f"{name:16s} {where:9s}: cos vs BF16 ref {cos:.5f}  {'OK' if cos > 0.99 else 'WRONG'}", flush=True)
        except Exception as e:
            print(f"{name:16s} {where:9s}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
            raise
