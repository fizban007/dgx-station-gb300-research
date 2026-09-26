"""Validate the V4.1 A16 decode path (b12x_layers) against a dequantized reference."""
import types, torch
from b12x.preparation import PreparationSession
import vllm.models.deepseek_v4_1.b12x_layers as L
dev = torch.device("cuda", 0)
for n, k in ((1280, 5120), (32768, 1280), (5120, 2304)):
    layer = types.SimpleNamespace()
    layer.weight = (torch.randn(n, k, device=dev) * 2).to(torch.float8_e4m3fn)
    layer.weight_scale_inv = torch.randint(118, 128, ((n + 31) // 32, (k + 31) // 32), dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
    L._a16_prepare(layer)
    with PreparationSession(device=dev, autotune=False, compile_workers=0) as session:
        session.prepare(L._a16_requests(layer))
        scale = torch.exp2(layer.weight_scale_inv.view(torch.uint8).float() - 127).repeat_interleave(32, 0)[:n].repeat_interleave(32, 1)[:, :k]
        dense = layer.weight.float() * scale
        for rows in (1, 3, 8):
            x = torch.randn(rows, k, device=dev, dtype=torch.bfloat16)
            out = torch.empty(rows, n, device=dev, dtype=torch.bfloat16)
            assert L._a16_run(layer, x, out, rows)
            ref = x.float() @ dense.T
            err = float((out.float() - ref).norm() / ref.norm())
            print(n, k, rows, f"rel err {err:.2e}")
