"""Fused decode send (hook/peer_fused.py) vs the current PyTorch glue + PeerTier2._pack/_publish.

Both arms write into device-memory stand-ins for the shared buffer, from identical inputs, and every output the
peer or finish() reads must match bit for bit: hot ids, has/pos/rows, packed activations/scales/ids/weights, pad
masking, header, words, device sequence and route counts. Run on the GB300 with the lane's Triton:
    python test_peer_fused.py
"""
import sys

import torch

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "hook"))
import peer_fused  # noqa: E402
import peer_tier2 as pt  # noqa: E402

E, K, H = 384, pt.TOPK, pt.HIDDEN
ROWS = 128


class FakePeer:
    def __init__(self, dev):
        self.words = torch.zeros(64, dtype=torch.int64, device=dev)
        self.header = torch.zeros(pt.HEADER, dtype=torch.int64, device=dev)
        self.px = torch.full((ROWS * H,), 0x5A, dtype=torch.uint8, device=dev)
        self.pxs = torch.full((ROWS * (H // 32),), 0x5A, dtype=torch.uint8, device=dev)
        self.pids = torch.full((ROWS * K,), 1234, dtype=torch.int32, device=dev)
        self.pw = torch.full((ROWS * K,), 7.0, dtype=torch.float32, device=dev)
        self.seq = torch.zeros(1, dtype=torch.int64, device=dev)


def reference(peer, ids, w, row_map, counts, xq, xs, layer, pad):
    valid = ids >= 0
    counts.index_add_(0, ids.clamp_min(0).flatten().long(), valid.flatten().long())
    rm = row_map[ids.clamp_min(0).long()]
    hot = torch.where(valid & (rm >= 0), rm, -1).to(ids.dtype)
    cold = torch.where(valid & (rm < 0), -rm - 1, -1).to(torch.int32)
    tokens = ids.shape[0]
    has = (cold >= 0).any(dim=1)
    pos = torch.cumsum(has, 0, dtype=torch.int32) - 1
    rows = has.sum(dtype=torch.int32).reshape(1)
    has = has.to(torch.int8)
    pt._pack[(tokens,)](xq.view(torch.uint8), xs.view(torch.uint8), cold.contiguous(), w.float().contiguous(),
                        has, pos, peer.px, peer.pxs, peer.pids, peer.pw,
                        H=H, HS=H // 32, K=K, BLOCK=1024, BLOCK_S=256, BLOCK_K=8)
    pt._publish[(1,)](peer.words, peer.header, peer.seq, layer, rows, tokens, peer.pids, K=K, PAD=pad)
    return hot, has, pos, rows


def main():
    dev = torch.device("cuda")
    torch.manual_seed(0)
    perm = torch.randperm(E)
    rm_list = [0] * E
    for r, e in enumerate(perm[:285].tolist()):
        rm_list[e] = r
    for c, e in enumerate(perm[285:].tolist()):
        rm_list[e] = -(c + 1)
    row_map = torch.tensor(rm_list, dtype=torch.int32, device=dev)
    failures = 0
    cases = 0
    for tokens in (1, 2, 3, 6, 8, 12, 16, 24, 32, 48, 64):
        for id_dtype in (torch.int32, torch.int64):
            for w_dtype in (torch.float32, torch.bfloat16):
                for cold_share in (0.0, 0.05, 0.3, 1.0):
                    cases += 1
                    ids = torch.stack([torch.randperm(E, device=dev)[:K] for _ in range(tokens)]).to(id_dtype)
                    # steer the requested share of routes onto cold experts; mask a few routes (-1)
                    cold_experts = perm[285:].to(dev)
                    pick = torch.rand(tokens, K, device=dev) < cold_share
                    ids = torch.where(pick, cold_experts[torch.randint(0, 99, (tokens, K), device=dev)].to(id_dtype), ids)
                    ids[torch.rand(tokens, K, device=dev) < 0.05] = -1
                    w = torch.rand(tokens, K, device=dev).to(w_dtype)
                    xq = torch.randint(0, 256, (tokens, H), dtype=torch.uint8, device=dev)
                    xs = torch.randint(0, 256, (tokens, H // 32), dtype=torch.uint8, device=dev)
                    layer = 17
                    a, b = FakePeer(dev), FakePeer(dev)
                    for p in (a, b):   # previous publish left stale state
                        p.seq.fill_(41)
                        p.words[3] = 1000
                        p.words[4] = 2000
                    ca = torch.full((E,), 5, dtype=torch.int64, device=dev)
                    cb = ca.clone()
                    ra = reference(a, ids, w, row_map, ca, xq, xs, layer, pt.SMALL_ROWS)
                    rb = peer_fused.route_send(b, ids, w, row_map, cb, xq, xs, layer, H, K, pt.SMALL_ROWS)
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
                        "words": torch.equal(a.words[:5], b.words[:5]), "seq": torch.equal(a.seq, b.seq),
                        "counts": torch.equal(ca, cb),
                    }
                    bad = [k for k, ok in checks.items() if not ok]
                    if bad:
                        failures += 1
                        print(f"FAIL T={tokens} ids={id_dtype} w={w_dtype} cold={cold_share} rows={n}: {bad}")
    print(f"{cases - failures}/{cases} cases bit-identical")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
