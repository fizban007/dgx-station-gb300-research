"""peer_fused2 (route_send2 + finish2) vs the live path (FlashInfer MXFP8 quantize + route_send + PeerTier2.finish).

Every output the peer or the GB300 reads must match bit for bit: hot ids, has/pos/rows, packed activations/scales/
ids/weights, pad masking, header, words, device sequence, route counts, and y after the scatter-add. Also checks the
in-kernel quantizer on adversarial blocks, and times one decode step's worth (40 layers) of each path under a CUDA
graph. Run inside the lane's image on the GB300:
    python test_peer_fused2.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hook"))
import peer_fused  # noqa: E402
import peer_fused2  # noqa: E402
import peer_tier2 as pt  # noqa: E402
from test_peer_fused import E, H, K, FakePeer  # noqa: E402


def quant(x):
    from flashinfer import mxfp8_quantize
    xq, xs = mxfp8_quantize(x.contiguous(), is_sf_swizzled_layout=False, alignment=32, backend="cute-dsl")
    return xq, xs.view(x.shape[0], -1)


class Peer(FakePeer):
    def __init__(self, dev):
        super().__init__(dev)
        self.pout = torch.randn(128 * H, device=dev).to(torch.bfloat16)
        self.ok = torch.zeros(1, dtype=torch.int32, device=dev)
        self.timeout = 1 << 20


def old_path(p, ids, w, row_map, counts, x, layer, y):
    xq, xs = quant(x)
    hot, has, pos, rows = peer_fused.route_send(p, ids, w, row_map, counts, xq, xs, layer, H, K, pt.SMALL_ROWS)
    p.words[1] = p.seq[0]  # the peer finished this sequence
    pt.PeerTier2.finish(p, y, has, pos, rows)
    return hot, has, pos, rows


def new_path(p, ids, w, row_map, counts, x, layer, y):
    hot, has, pos, rows = peer_fused2.route_send2(p, ids, w, row_map, counts, x, layer, H, K, pt.SMALL_ROWS,
                                                  stats=True)
    p.words[1] = p.seq[0]
    peer_fused2.finish2(p, y, has, pos, rows)
    return hot, has, pos, rows


def adversarial_rows(n, dev):
    x = torch.randn(n, H, device=dev) * torch.logspace(-6, 6, n, device=dev)[:, None]
    blk = x.view(n, H // 32, 32)
    blk[:, 0] = 0.0                                              # all-zero block
    blk[:, 1, 0] = 448.0 * 2.0 ** torch.randint(-20, 20, (n,), device=dev).float()  # amax exactly 448 * 2^j
    blk[:, 1, 1:] = blk[:, 1, :1] * 0.5
    blk[:, 2] = 1e-30                                            # tiny
    blk[:, 3, 5] = -3.0e30                                       # huge negative amax
    return x.to(torch.bfloat16)


def main():
    dev = torch.device("cuda")
    torch.manual_seed(0)
    # quantizer alone vs FlashInfer
    for n in (1, 7, 64):
        x = adversarial_rows(n, dev)
        p = Peer(dev)
        ids = torch.full((n, K), -1, dtype=torch.int32, device=dev)
        ids[:, 0] = 0
        rm = torch.full((E,), -1, dtype=torch.int32, device=dev)  # every expert cold: every row packed in order
        w = torch.rand(n, K, device=dev)
        peer_fused2.route_send2(p, ids, w, rm, None, x, 3, H, K, pt.SMALL_ROWS)
        xq, xs = quant(x)
        torch.cuda.synchronize()
        dq = (p.px[: n * H] != xq.view(torch.uint8).reshape(-1)).sum().item()
        ds = (p.pxs[: n * (H // 32)] != xs.view(torch.uint8).reshape(-1)).sum().item()
        print(f"quantizer n={n}: {dq} value bytes and {ds} scale bytes differ from FlashInfer")

    perm = torch.randperm(E)
    rm_list = [0] * E
    for r, e in enumerate(perm[:285].tolist()):
        rm_list[e] = r
    for c, e in enumerate(perm[285:].tolist()):
        rm_list[e] = -(c + 1)
    row_map = torch.tensor(rm_list, dtype=torch.int32, device=dev)
    failures = cases = 0
    for tokens in (1, 2, 3, 6, 8, 12, 16, 24, 32, 48, 64):
        for id_dtype in (torch.int32, torch.int64):
            for w_dtype in (torch.float32, torch.bfloat16):
                for cold_share in (0.0, 0.05, 0.3, 1.0):
                    cases += 1
                    ids = torch.stack([torch.randperm(E, device=dev)[:K] for _ in range(tokens)]).to(id_dtype)
                    cold_experts = perm[285:].to(dev)
                    pick = torch.rand(tokens, K, device=dev) < cold_share
                    ids = torch.where(pick, cold_experts[torch.randint(0, 99, (tokens, K), device=dev)].to(id_dtype), ids)
                    ids[torch.rand(tokens, K, device=dev) < 0.05] = -1
                    w = torch.rand(tokens, K, device=dev).to(w_dtype)
                    x = (torch.randn(tokens, H, device=dev) * 3).to(torch.bfloat16)
                    a, b = Peer(dev), Peer(dev)
                    b.pout.copy_(a.pout)
                    for p in (a, b):
                        p.seq.fill_(41)
                        p.words[3] = 1000
                        p.words[4] = 2000
                    ca = torch.full((E,), 5, dtype=torch.int64, device=dev)
                    cb = ca.clone()
                    ya = torch.randn(tokens, H, device=dev).to(torch.bfloat16)
                    yb = ya.clone()
                    ra = old_path(a, ids, w, row_map, ca, x, 17, ya)
                    rb = new_path(b, ids, w, row_map, cb, x, 17, yb)
                    torch.cuda.synchronize()
                    n = int(ra[3].item())
                    checks = {
                        "hot": torch.equal(ra[0], rb[0]), "has": torch.equal(ra[1], rb[1]),
                        "pos": torch.equal(ra[2], rb[2]), "rows": torch.equal(ra[3], rb[3]),
                        "px": torch.equal(a.px[: n * H], b.px[: n * H]),
                        "pxs": torch.equal(a.pxs[: n * (H // 32)], b.pxs[: n * (H // 32)]),
                        "pids": torch.equal(a.pids[: max(n, pt.SMALL_ROWS) * K], b.pids[: max(n, pt.SMALL_ROWS) * K]),
                        "pw": torch.equal(a.pw[: n * K], b.pw[: n * K]),
                        "header": torch.equal(a.header[:4], b.header[:4]),
                        "words": torch.equal(a.words[:7].clone().index_fill_(0, torch.tensor([5], device=dev), 0),
                                             b.words[:7].clone().index_fill_(0, torch.tensor([5], device=dev), 0)),
                        "seq": torch.equal(a.seq, b.seq), "counts": torch.equal(ca, cb), "y": torch.equal(ya, yb),
                    }
                    bad = [k for k, ok in checks.items() if not ok]
                    if bad:
                        failures += 1
                        print(f"FAIL T={tokens} ids={id_dtype} w={w_dtype} cold={cold_share} rows={n}: {bad}")
    print(f"{cases - failures}/{cases} cases bit-identical")

    # timing: 40 layers per graph, peer completion pre-set so the wait never spins
    for tokens in (1, 6, 12, 24, 48, 64):
        ids = torch.stack([torch.randperm(E, device=dev)[:K] for _ in range(tokens)]).to(torch.int32)
        w = torch.rand(tokens, K, device=dev)
        x = torch.randn(tokens, H, device=dev).to(torch.bfloat16)
        res = {}
        for name, fn in (("old", old_path), ("new", new_path)):
            p = Peer(dev)
            p.words[1] = 1 << 40
            y = torch.zeros(tokens, H, device=dev, dtype=torch.bfloat16)
            counts = torch.zeros(E, dtype=torch.int64, device=dev)

            def step(p=p, y=y, counts=counts, fn=fn):
                for layer in range(40):
                    xq_needed = fn is old_path
                    if xq_needed:
                        xq, xs = quant(x)
                        h = peer_fused.route_send(p, ids, w, row_map, counts, xq, xs, layer, H, K, pt.SMALL_ROWS)
                        pt.PeerTier2.finish(p, y, *h[1:])
                    else:
                        h = peer_fused2.route_send2(p, ids, w, row_map, counts, x, layer, H, K, pt.SMALL_ROWS)
                        peer_fused2.finish2(p, y, *h[1:])
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                step()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                step()
            g.replay()
            torch.cuda.synchronize()
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(20):
                st.record(); g.replay(); en.record(); en.synchronize()
                samples.append(st.elapsed_time(en) * 1000 / 40)
            samples.sort()
            res[name] = samples[len(samples) // 2]
        print(f"T={tokens:3d}: old (quant+send+wait+scatter) {res['old']:5.1f} us/layer, new {res['new']:5.1f} us/layer, "
              f"saves {40 * (res['old'] - res['new']) / 1000:.3f} ms/step")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
