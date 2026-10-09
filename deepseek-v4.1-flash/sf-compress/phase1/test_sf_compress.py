"""Phase 1 offline test: bit-exact round trip and decode timing of hook/sf_compress.py on real MegaMoE SF tensors.

Builds each layer's L1/L2 SF exactly as vLLM's DeepseekV4MegaMoEExperts.finalize_weights does (checkpoint UE8M0
bytes -> fp32 via <<23 -> deep_gemm.transform_sf_into_required_layout -> MegaMoE gate/up interleave + UTCCP
transpose), for the hot experts of a rowmap, then encodes, decodes (all experts and random subsets) and compares.

  python test_sf_compress.py --layers 0 37 38 39 --rowmap /w/rowmap-mix-h258.json
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/w")
import sf_compress as sfc  # noqa: E402

I, H = 2304, 5120


def reader(model):
    index = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    headers = {}

    def get(name):
        path = os.path.join(model, index[name])
        if path not in headers:
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                headers[path] = (8 + n, json.loads(f.read(n)))
        base, hdr = headers[path]
        meta = hdr[name]
        s, e = meta["data_offsets"]
        with open(path, "rb") as f:
            f.seek(base + s)
            return torch.from_numpy(np.frombuffer(f.read(e - s), dtype=np.uint8).reshape(meta["shape"]).copy())

    return get


def build_sf(get, layer, hot, dev):
    from vllm.third_party import deep_gemm as dg
    from vllm.third_party.deep_gemm import mega as dgm
    u8_13 = torch.stack([torch.cat([get(f"layers.{layer}.ffn.experts.{e}.w1.scale"),
                                    get(f"layers.{layer}.ffn.experts.{e}.w3.scale")]) for e in hot]).to(dev)
    u8_2 = torch.stack([get(f"layers.{layer}.ffn.experts.{e}.w2.scale") for e in hot]).to(dev)
    f = lambda u: (u.to(torch.int32) << 23).view(torch.float32).contiguous()  # noqa: E731
    E = len(hot)
    s13 = dg.transform_sf_into_required_layout(f(u8_13), 2 * I, H, (1, 32), E)
    s2 = dg.transform_sf_into_required_layout(f(u8_2), H, I, (1, 32), E)
    l1 = dgm._transpose_sf_for_utccp(dgm._interleave_weights(s13))
    l2 = dgm._transpose_sf_for_utccp(s2)
    return l1, l2


def bench(fn, iters=50):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/model")
    p.add_argument("--rowmap", default="/w/rowmap-mix-h258.json")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 20, 37, 38, 39])
    p.add_argument("--ch", type=int, nargs="+", default=[16])
    p.add_argument("--warps", type=int, nargs="+", default=[4])
    a = p.parse_args()
    dev = torch.device("cuda")
    get = reader(a.model)
    rowmap = json.load(open(a.rowmap))["layers"]
    g = torch.Generator(device="cpu").manual_seed(0)
    tot_raw = tot_comp = 0
    for layer in a.layers:
        hot = rowmap[str(layer)]["hot"]
        l1, l2 = build_sf(get, layer, hot, dev)
        for tag, sf in (("L1", l1), ("L2", l2)):
            E, N, Kp = sf.shape
            t0 = time.perf_counter()
            c = sfc.encode(sf)
            torch.cuda.synchronize()
            t_enc = time.perf_counter() - t0
            out = sfc.empty_like_sf(c.shape, c.stride, dev)
            all_ids = torch.arange(E, dtype=torch.int32, device=dev)
            exl = sfc.make_expert_list(E, dev)
            out.fill_(0x7F7F7F7F)
            sfc.decode(c, out, sfc.fill_expert_list(all_ids, exl))
            exact = torch.equal(out, sf)
            # subset: route ids like a decode step (6 tokens x top-6, -1 padding included), check only those decoded
            ids = torch.randint(0, E, (6, 6), generator=g).to(dev)
            ids[0, 0] = -1
            out.fill_(0x7F7F7F7F)
            sfc.decode(c, out, sfc.fill_expert_list(ids, exl))
            sel = torch.zeros(E, dtype=torch.bool, device=dev)
            sel[ids[ids >= 0].long()] = True
            sub_exact = (torch.equal(out[sel], sf[sel]) and bool((out[~sel] == 0x7F7F7F7F).all())
                         and int(exl.count) == int(sel.sum()))
            n_spill = int((c.slots[..., 0] >> 10 & 1).sum())
            raw = sf.numel() * 4
            tot_raw += raw
            tot_comp += c.nbytes()
            # timing (CUDA graph replay), list kernel included: all experts (ids = 0..E-1) and the 6x6 step
            timings = []
            for ch in a.ch:
                for nw in a.warps:
                    sfc.DECODE_CH, sfc.DECODE_WARPS = ch, nw
                    t_all = bench(lambda: sfc.decode(c, out, sfc.fill_expert_list(all_ids, exl)))
                    t_sub = bench(lambda: sfc.decode(c, out, sfc.fill_expert_list(ids, exl)))
                    timings.append(f"CH{ch}w{nw}: all {t_all:.1f} us ({raw / t_all / 1e3:.0f} GB/s), 36ids {t_sub:.1f} us")
            print(f"layer {layer:2d} {tag}: {tuple(sf.shape)} exact={exact} subset_exact={sub_exact} "
                  f"{raw / 2**20:6.1f} -> {c.nbytes() / 2**20:5.1f} MiB ({c.nbytes() / raw:.1%}), spill {n_spill} "
                  f"chunks | " + " | ".join(timings), flush=True)
            del c, out
        del l1, l2
        torch.cuda.empty_cache()
    print(f"total: {tot_raw / 2**30:.2f} GiB -> {tot_comp / 2**30:.3f} GiB ({tot_comp / tot_raw:.1%}); "
          f"peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
