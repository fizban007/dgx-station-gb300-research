import sys, torch
sys.path.insert(0, "/m"); sys.path.insert(0, "/m/hook")
import test_peer_fused2 as T, peer_fused2 as pf2, peer_fused as pf, peer_tier2 as pt
dev = torch.device("cuda"); E, K, H = T.E, T.K, T.H
perm = torch.randperm(E); rm = [0]*E
for r, e in enumerate(perm[:285].tolist()): rm[e] = r
for c, e in enumerate(perm[285:].tolist()): rm[e] = -(c+1)
row_map = torch.tensor(rm, dtype=torch.int32, device=dev)
def timed(fn):
    s = torch.cuda.Stream(); 
    with torch.cuda.stream(s): fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    g.replay(); torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); xs=[]
    for _ in range(20): a.record(); g.replay(); b.record(); b.synchronize(); xs.append(a.elapsed_time(b)*1000/40)
    return sorted(xs)[10]
for tokens in (1, 6, 48):
    ids = torch.stack([torch.randperm(E, device=dev)[:K] for _ in range(tokens)]).to(torch.int32)
    w = torch.rand(tokens, K, device=dev); x = torch.randn(tokens, H, device=dev).to(torch.bfloat16)
    p = T.Peer(dev); p.words[1] = 1 << 40; y = torch.zeros(tokens, H, device=dev, dtype=torch.bfloat16)
    xq, xs = T.quant(x); h = pf2.route_send2(p, ids, w, row_map, None, x, 0, H, K, pt.SMALL_ROWS)
    r = {
     "quant": timed(lambda: [T.quant(x) for _ in range(40)]),
     "send_old": timed(lambda: [pf.route_send(p, ids, w, row_map, None, xq, xs, 0, H, K, pt.SMALL_ROWS) for _ in range(40)]),
     "finish_old": timed(lambda: [pt.PeerTier2.finish(p, y, *h[1:]) for _ in range(40)]),
     "send2": timed(lambda: [pf2.route_send2(p, ids, w, row_map, None, x, 0, H, K, pt.SMALL_ROWS) for _ in range(40)]),
     "finish2": timed(lambda: [pf2.finish2(p, y, *h[1:]) for _ in range(40)]),
     "empty_triton": timed(lambda: [pt._globaltimer_probe(p.words) if hasattr(pt, "_globaltimer_probe") else torch.add(y, 0, out=y) for _ in range(40)]),
    }
    print(tokens, {k: round(v, 2) for k, v in r.items()})
