"""torch.argmax over the DSpark draft vocab vs a two-stage argmax (CUDA-graph timed)."""
import json

import torch

V = 129280
CH = 1010  # 129280 = 1010 * 128


def two_stage(x):
    r = x.shape[0]
    v, i = x.view(r, CH, V // CH).max(dim=-1)  # [r, CH] per-chunk max + in-chunk index
    j = v.argmax(dim=-1, keepdim=True)          # winning chunk
    return j.squeeze(-1) * (V // CH) + i.gather(-1, j).squeeze(-1)


def t(fn, iters=200):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters / 10 * 1e3


for r in (1, 2, 4, 8, 12, 16, 24):
    x = torch.randn(r, V, device="cuda", dtype=torch.bfloat16)
    # ties: first index wins in both (torch.max/argmax return the first max on CUDA for these reductions)
    ok = torch.equal(x.argmax(-1), two_stage(x))
    print(json.dumps({"rows": r, "torch_argmax_us": round(t(lambda: x.argmax(-1)), 2),
                      "two_stage_us": round(t(lambda: two_stage(x)), 2), "equal": ok}), flush=True)
