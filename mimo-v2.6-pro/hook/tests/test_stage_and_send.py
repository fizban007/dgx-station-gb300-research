"""Standalone GPU tests for stage_grace.GraceStager and PeerTier2.send_fused/finish_fused (run inside the vLLM image).

  docker run --rm --gpus '"device=<GB300>"' --ipc host -v hook:/w -e PEER_SHM=/dev/shm/vllm_peer_mimo_test ... \
      python3 /w/test_stage_and_send.py
Never point PEER_SHM at the live /dev/shm/vllm_peer_mimo.
"""
import os
import sys
import time

import torch

assert os.environ.get("PEER_SHM", "").endswith("_test"), "set PEER_SHM to a *_test path"
sys.path.insert(0, "/w")
import peer_tier_mimo as pt  # noqa: E402
from stage_grace import GraceStager, _words  # noqa: E402

dev = torch.device("cuda", 0)
torch.cuda.set_device(dev)
g = torch.Generator(device=dev).manual_seed(0)
E, K, H = 384, 8, pt.HIDDEN
MiB = 2**20


def uva(t_cpu):
    from cuda.bindings import runtime as rt
    err, dptr = rt.cudaHostGetDevicePointer(t_cpu.data_ptr(), 0)
    assert err == rt.cudaError_t.cudaSuccess

    class _B:
        def __init__(self, p, n, dt, shape):
            self.__cuda_array_interface__ = {"shape": shape, "typestr": dt, "data": (p, False), "version": 3}
    return torch.as_tensor(_B(int(dptr), None, "<i4", tuple(t_cpu.shape)), device=dev)


# ---------------------------------------------------------------- stager
n_c = 192                                            # Grace experts in this layer
w13_cpu = torch.randint(-2**31, 2**31 - 1, (n_c, 3 * MiB // 4 * 4 // 4), dtype=torch.int32).pin_memory()  # 12 MiB/expert
w2_cpu = torch.randint(-2**31, 2**31 - 1, (n_c, 6 * MiB // 4), dtype=torch.int32).pin_memory()             # 6 MiB/expert
w13 = uva(w13_cpu)
w2 = uva(w2_cpu)
s13 = torch.randint(0, 255, (n_c, 4096, 192), dtype=torch.uint8, device=dev)
s2 = torch.randint(0, 255, (n_c, 6144, 64), dtype=torch.uint8, device=dev)
S = 128
st = GraceStager(w13, w2, s13, s2, slots=S, num_experts=E)
print(f"staging {st.nbytes / MiB:.0f} MiB for {S} slots")
grace_ids = torch.randperm(E, device=dev, generator=g)[:n_c]
cmap = torch.full((E,), -1, dtype=torch.int32, device=dev)
cmap[grace_ids] = torch.arange(n_c, dtype=torch.int32, device=dev)
src = tuple(_words(t) for t in (w13, w2, s13, s2))


def check_stage(topk_ids, emap):
    torch.cuda.synchronize()
    ids = topk_ids.flatten().tolist()
    want = sorted({e for e in ids if cmap[e] >= 0})
    got = {e: int(emap[e]) for e in range(E) if int(emap[e]) >= 0}
    assert sorted(got) == want, (sorted(got)[:8], want[:8])
    assert sorted(got.values()) == list(range(len(want))), "slots not dense"
    for e, s in got.items():
        c = int(cmap[e])
        for staged, full in ((st.w13, w13_cpu), (st.w2, w2_cpu)):
            assert torch.equal(staged[s].cpu(), full[c]), f"w mismatch e={e}"
        assert torch.equal(st.s13[s], s13[c]) and torch.equal(st.s2[s], s2[c]), f"scale mismatch e={e}"
    return len(want)


for T in (1, 4, 16):
    topk = torch.stack([torch.randperm(E, device=dev, generator=g)[:K] for _ in range(T)]).int()
    n = check_stage(topk, st.stage(topk, cmap, src))
    print(f"stage eager T={T}: {n} Grace experts staged, bytes exact")

# graph capture: capture once, change routing in place, replay
topk = torch.stack([torch.randperm(E, device=dev, generator=g)[:K] for _ in range(16)]).int()
st.stage(topk, cmap, src)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    st.stage(topk, cmap, src)
for trial in range(3):
    topk.copy_(torch.stack([torch.randperm(E, device=dev, generator=g)[:K] for _ in range(16)]).int())
    graph.replay()
    n = check_stage(topk, st.emap)
    print(f"stage graph replay {trial}: {n} experts staged, bytes exact")

# bandwidth vs Marlin-over-UVA (~258 GB/s at C1 in the lane's profile)
per_expert = sum(t[0].numel() * t.element_size() for t in (w13, w2, s13, s2))
for T in (1, 2, 4, 8, 16):
    topk = torch.stack([torch.randperm(E, device=dev, generator=g)[:K] for _ in range(T)]).int()
    st.stage(topk, cmap, src)
    torch.cuda.synchronize()
    n = int(st.nused)
    reps = 50
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        st.stage(topk, cmap, src)
    t1.record(); torch.cuda.synchronize()
    us = t0.elapsed_time(t1) * 1e3 / reps
    print(f"stage bandwidth T={T:2d}: {n:3d} experts, {n * per_expert / MiB:6.0f} MiB in {us:7.1f} us "
          f"= {n * per_expert / (us / 1e6) / 1e9:6.1f} GB/s")

# ---------------------------------------------------------------- fused peer send
tier = pt.PeerTier2(dev)
pm = torch.full((E,), -1, dtype=torch.int32, device=dev)
peer_ids = torch.randperm(E, device=dev, generator=g)[:70]
pm[peer_ids] = torch.arange(70, dtype=torch.int32, device=dev)


def quant_ref(x):
    xb = x.float().view(x.shape[0], -1, 32)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    return (xb / torch.exp2(e)).to(torch.float8_e4m3fn).view(x.shape[0], -1).view(torch.uint8), \
        (e.squeeze(-1) + 127).to(torch.uint8)


for T in (1, 7, 16, 300, 8192):
    x = (torch.randn(T, H, device=dev, generator=g) * 0.7).to(torch.bfloat16)
    topk = torch.stack([torch.randperm(E, device=dev, generator=g)[:K] for _ in range(T)]).int()
    wts = torch.rand(T, K, device=dev, generator=g)
    # reference: old path
    pids_old = torch.where(topk >= 0, pm[topk.clamp_min(0).long()], -1).int()
    xq, xs = quant_ref(x)
    has, pos_old, rows_old = tier.send(xq, xs, pids_old, wts, 5)
    torch.cuda.synchronize()
    r_old = int(rows_old)
    snap = {k: getattr(tier, k)[: r_old * n].clone() for k, n in (("pids", K), ("pw", K), ("px", H), ("pxs", H // 32))}
    pos_old = torch.where(has.bool(), pos_old, -1)
    # fused path
    pos_new, rows_new = tier.send_fused(x, topk, wts, pm, 5)
    torch.cuda.synchronize()
    assert int(rows_new) == r_old, (int(rows_new), r_old)
    assert torch.equal(pos_new[:T], pos_old.int()), "POS differs"
    assert torch.equal(tier.pids[: r_old * K], snap["pids"]), "ids differ"
    assert torch.equal(tier.pw[: r_old * K], snap["pw"]), "weights differ"
    # activations: same scales; E4M3 bytes may differ in the last bit of rounding, compare dequantized values
    assert torch.equal(tier.pxs[: r_old * (H // 32)], snap["pxs"]), "scales differ"
    deq = lambda q, s: (q.view(torch.float8_e4m3fn).float().view(r_old, -1, 32)
                        * torch.exp2(s.float() - 127).view(r_old, -1, 1)).view(r_old, H)
    a = deq(tier.px[: r_old * H].view(r_old, H), tier.pxs[: r_old * (H // 32)])
    b = deq(snap["px"].view(r_old, H), snap["pxs"])
    rel = float((a - b).norm() / b.norm().clamp_min(1e-9)) if r_old else 0.0
    assert rel < 1e-3, rel
    # finish: pretend the peer answered this sequence with known rows
    tier.pout[: r_old * H].copy_(torch.randn(r_old * H, device=dev, generator=g).to(torch.bfloat16))
    tier.words[1].copy_(tier.seq[0])
    y = torch.randn(T, H, device=dev, generator=g).to(torch.bfloat16)
    want = y.float().clone()
    rows_idx = (pos_new[:T] >= 0).nonzero().flatten()
    if r_old:
        want[rows_idx] += tier.pout[: r_old * H].view(r_old, H)[pos_new[rows_idx].long()].float()
    tier.finish_fused(y, (pos_new, rows_new))
    torch.cuda.synchronize()
    err = float((y.float() - want).abs().max())
    assert err < 0.05, err
    print(f"fused send T={T:5d}: rows {r_old}, ids/weights/pos/scales exact, activations rel {rel:.1e}, scatter-add ok")

# fused send under graph capture
T = 16
x = (torch.randn(T, H, device=dev, generator=g) * 0.7).to(torch.bfloat16)
topk = torch.stack([torch.randperm(E, device=dev, generator=g)[:K] for _ in range(T)]).int()
wts = torch.rand(T, K, device=dev, generator=g)
state = tier.send_fused(x, topk, wts, pm, 9)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    state = tier.send_fused(x, topk, wts, pm, 9)
for trial in range(3):
    topk.copy_(torch.stack([torch.randperm(E, device=dev, generator=g)[:K] for _ in range(T)]).int())
    graph.replay()
    torch.cuda.synchronize()
    ref = int(((pm[topk.long()] >= 0).any(1)).sum())
    assert int(state[1]) == ref, (int(state[1]), ref)
    print(f"fused send graph replay {trial}: rows {ref} ok")
print("ALL OK")
