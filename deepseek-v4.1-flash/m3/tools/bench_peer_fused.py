"""CUDA-graph replay time of the per-layer send glue: current PyTorch ops + _pack/_publish vs the fused kernel."""
import sys
import torch
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import test_peer_fused as t  # noqa: E402

dev = torch.device("cuda")
perm = torch.randperm(t.E)
rm = [0] * t.E
for r, e in enumerate(perm[:285].tolist()): rm[e] = r
for c, e in enumerate(perm[285:].tolist()): rm[e] = -(c + 1)
row_map = torch.tensor(rm, dtype=torch.int32, device=dev)
for tokens in (1, 6, 12, 24, 48, 64):
    ids = torch.stack([torch.randperm(t.E, device=dev)[:t.K] for _ in range(tokens)]).to(torch.int32)
    w = torch.rand(tokens, t.K, device=dev)
    xq = torch.randint(0, 256, (tokens, t.H), dtype=torch.uint8, device=dev)
    xs = torch.randint(0, 256, (tokens, t.H // 32), dtype=torch.uint8, device=dev)
    counts = torch.zeros(t.E, dtype=torch.int64, device=dev)
    res = {}
    for name, fn in (("torch+pack", lambda p: t.reference(p, ids, w, row_map, counts, xq, xs, 3, t.pt.SMALL_ROWS)),
                     ("fused", lambda p: t.peer_fused.route_send(p, ids, w, row_map, counts, xq, xs, 3, t.H, t.K, t.pt.SMALL_ROWS))):
        p = t.FakePeer(dev)
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3): fn(p)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(40): fn(p)   # one decode step's worth of layers
        g.replay(); torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(20):
            a.record(); g.replay(); b.record(); b.synchronize(); samples.append(a.elapsed_time(b) * 1000 / 40)
        samples.sort(); res[name] = samples[len(samples) // 2]
    print(f"T={tokens:3d}: torch+pack {res['torch+pack']:6.1f} us/layer, fused {res['fused']:5.1f} us/layer, "
          f"saves {40 * (res['torch+pack'] - res['fused']) / 1000:.2f} ms/step (40 layers)")
