"""Drive the running v2 sidecar through the real protocol and check every result against dense PyTorch.

Runs on the RTX PRO 6000 next to the sidecar (only while no server is using it). For random layers and
batch sizes it packs random activations and cold routes with PeerTier2.send, waits with finish into a zero
output, and compares with a dense reference of the same cold experts. Each batch is sent twice (determinism).
  MODE=eager (default) or MODE=graph (capture send+finish per size, like the GB300 does)
"""
import json
import os
import random
import sys

import torch

sys.path.insert(0, "/home/jasonc/research/megamoe/hook")
sys.path.insert(0, "/home/jasonc/research/megamoe")
import peer_tier2 as pt  # noqa: E402
from peer_prefill_bench import tensor, mxfp8, dense_ref, H, I, K, LIMIT  # noqa: E402,F401

ROWMAP = os.environ.get("ROWMAP_FILE", "/home/jasonc/research/megamoe/hook/rowmap-static-v1.json")
MODE = os.environ.get("MODE", "eager")
LAYERS = [int(v) for v in os.environ.get("LAYERS", "0 7 20 39").split()]
SIZES = [int(v) for v in os.environ.get("SIZES", "1 2 3 5 6 8 12 16 24 31 48 64 65 100 300").split()]


def main():
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    rowmap = json.load(open(ROWMAP))["layers"]
    peer = pt.PeerTier2(dev)
    rng = random.Random(5)
    g = torch.Generator(device=dev).manual_seed(5)
    bad, total, nondet = 0, 0, 0
    for layer in LAYERS:
        cold = rowmap[str(layer)]["cold"]
        W = {}
        parts = {k: [] for k in ("w13", "s13", "w2", "s2")}
        for e in cold:
            p = f"layers.{layer}.ffn.experts.{e}"
            parts["w13"].append(torch.cat([tensor(f"{p}.w1.weight"), tensor(f"{p}.w3.weight")]))
            parts["s13"].append(torch.cat([tensor(f"{p}.w1.scale"), tensor(f"{p}.w3.scale")]))
            parts["w2"].append(tensor(f"{p}.w2.weight"))
            parts["s2"].append(tensor(f"{p}.w2.scale"))
        W = {k: torch.stack(v).to(dev) for k, v in parts.items()}
        for T in SIZES:
            x = torch.randn(T, H, device=dev, generator=g) * 0.3
            xq, xs, xdq = mxfp8(x)
            ids = torch.full((T, K), -1, dtype=torch.int32, device=dev)
            wts = torch.rand(T, K, device=dev, generator=g) * 0.3
            for t in range(T):  # ~half the tokens get 1-2 cold routes
                if rng.random() < 0.5:
                    for k in rng.sample(range(K), rng.choice((1, 2))):
                        ids[t, k] = rng.randrange(len(cold))
            outs = []
            for rep in range(2):
                y = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
                if MODE == "graph":
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        has, pos, rows = peer.send(xq, xs.view(torch.uint8), ids, wts, layer)
                        peer.finish(y, has, pos, rows)
                    y.zero_()
                    graph.replay()
                else:
                    has, pos, rows = peer.send(xq, xs.view(torch.uint8), ids, wts, layer)
                    peer.finish(y, has, pos, rows)
                torch.cuda.synchronize()
                outs.append(y.float().clone())
            live = (ids >= 0).any(1)
            ref = dense_ref(W, xdq, ids, wts)
            if live.any():
                cos = float(torch.nn.functional.cosine_similarity(outs[0][live].flatten(), ref[live].flatten(), dim=0))
            else:
                cos = 1.0 if outs[0].abs().max() == 0 else 0.0
            same = bool(torch.equal(outs[0], outs[1]))
            total += 1
            bad += cos < 0.99
            nondet += not same
            flag = "" if cos >= 0.99 and same else "  <-- BAD"
            print(f"layer {layer:2d} T={T:4d} rows={int(live.sum()):4d} cos={cos:.5f} repeat_identical={same}{flag}", flush=True)
        del W
        torch.cuda.empty_cache()
    words = pt.torch.frombuffer(peer.buf, dtype=torch.int64, count=5).tolist()
    print(f"{MODE}: {total} batches, {bad} wrong, {nondet} non-repeatable; words {words}")


if __name__ == "__main__":
    main()
