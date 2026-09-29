"""Run the mega attention decode kernel on fixed inputs; save outputs (for byte comparison across .so builds) and time it."""
import json, sys, torch
sys.path.insert(0, "/s")
import mega_attn_bench as mb
tag = sys.argv[1]
dev = torch.device("cuda")
res = {}
for s_q in (1, 6, 12, 24, 48, 96):
    torch.manual_seed(1234 + s_q)
    captured = {}
    orig = torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_decode
    # run bench_decode's setup + timing; grab the output buffers by wrapping alloc
    import vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn as fm
    real_alloc = mb.alloc_mega_attn_output
    def alloc(*a, **k):
        o = real_alloc(*a, **k); captured["out"] = o; return o
    mb.alloc_mega_attn_output = alloc
    m_us, k_us = mb.bench_decode(s_q, dev, 512, mb.V41_BYTES, 64, 64)
    mb.alloc_mega_attn_output = real_alloc
    o = captured["out"]
    torch.cuda.synchronize()
    torch.save({"data": o.data.view(torch.uint8).cpu(), "scale": o.scale.contiguous().cpu()}, f"/s/attn_out_{tag}_{s_q}.pt")
    res[s_q] = (round(m_us, 2), round(k_us, 2))
print(tag, json.dumps(res))
