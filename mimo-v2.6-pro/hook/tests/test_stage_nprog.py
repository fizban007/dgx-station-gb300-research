"""Stage bandwidth and overlap with an HBM-bound kernel vs copy program count (C1-like: 2 experts; C16-like: ~56)."""
import sys, torch
sys.path.insert(0, "/w")
from stage_grace import GraceStager, _words
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor as uva
dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
g = torch.Generator(device=dev).manual_seed(2)
E, K, n_c, MiB = 384, 8, 192, 2**20
w13_cpu = torch.randint(-2**31, 2**31 - 1, (n_c, 12 * MiB // 4), dtype=torch.int32).pin_memory()
w2_cpu = torch.randint(-2**31, 2**31 - 1, (n_c, 6 * MiB // 4), dtype=torch.int32).pin_memory()
w13, w2 = uva(w13_cpu), uva(w2_cpu)
s13 = torch.randint(0, 255, (n_c, 192, 4096), dtype=torch.uint8, device=dev)
s2 = torch.randint(0, 255, (n_c, 64, 6144), dtype=torch.uint8, device=dev)
cmap = torch.full((E,), -1, dtype=torch.int32, device=dev)
cmap[torch.randperm(E, device=dev, generator=g)[:n_c]] = torch.arange(n_c, dtype=torch.int32, device=dev)
src = tuple(_words(t) for t in (w13, w2, s13, s2))
hbm = torch.empty(512 * MiB, dtype=torch.int32, device=dev)   # an HBM-bound stand-in for the hot bank: stream 2 GiB
dst = torch.empty_like(hbm)
side = torch.cuda.Stream(dev); cap = torch.cuda.Stream(dev)
def timed(fn, reps=20):
    gr = torch.cuda.CUDAGraph()
    fn(); torch.cuda.synchronize()
    with torch.cuda.graph(gr, stream=cap):
        fn()
    gr.replay(); torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps): gr.replay()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) / reps * 1e3
for T, hbm_words in ((1, 48 * MiB // 4), (16, 512 * MiB // 4)):
    topk = torch.stack([torch.randperm(E, device=dev, generator=g)[:K] for _ in range(T)]).int()
    for nprog in (38, 76, 152, 304, 608):
        st = GraceStager(w13, w2, s13, s2, slots=128, num_experts=E, nprog=nprog)
        h = lambda: dst[:hbm_words].copy_(hbm[:hbm_words])
        def over():
            cur = torch.cuda.current_stream(); side.wait_stream(cur)
            with torch.cuda.stream(side):
                st.stage(topk, cmap, src)
            h(); cur.wait_stream(side)
        stage_us = timed(lambda: st.stage(topk, cmap, src)); h_us = timed(h); o_us = timed(over)
        n = int(st.nused); mb = n * 18.9
        print(f"T={T:2d} {n:2d} experts nprog={nprog:3d}: stage {stage_us:7.1f} us ({mb / stage_us * 1e6 / 1e9 * MiB / 1e6:5.0f} GB/s)"
              f"  hbm {h_us:6.1f}  overlapped {o_us:7.1f}  serial-sum {stage_us + h_us:7.1f}")
        del st
