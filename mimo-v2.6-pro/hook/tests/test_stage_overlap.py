"""Graph-capture test of the side-stream staging pattern hotsplit uses: fork, stage on the side stream, HBM work on the
capture stream, join, consume. Replays with new routing must stage the right bytes."""
import sys
import torch
sys.path.insert(0, "/w")
from stage_grace import GraceStager, _words

dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
g = torch.Generator(device=dev).manual_seed(1)
E, K, n_c, MiB = 384, 8, 192, 2**20
w13_cpu = torch.randint(-2**31, 2**31 - 1, (n_c, 12 * MiB // 4), dtype=torch.int32).pin_memory()
w2_cpu = torch.randint(-2**31, 2**31 - 1, (n_c, 6 * MiB // 4), dtype=torch.int32).pin_memory()
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor as uva
w13, w2 = uva(w13_cpu), uva(w2_cpu)
s13 = torch.randint(0, 255, (n_c, 192, 4096), dtype=torch.uint8, device=dev)
s2 = torch.randint(0, 255, (n_c, 64, 6144), dtype=torch.uint8, device=dev)
st = GraceStager(w13, w2, s13, s2, slots=128, num_experts=E)
cmap = torch.full((E,), -1, dtype=torch.int32, device=dev)
cmap[torch.randperm(E, device=dev, generator=g)[:n_c]] = torch.arange(n_c, dtype=torch.int32, device=dev)
src = tuple(_words(t) for t in (w13, w2, s13, s2))
a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
side = torch.cuda.Stream(dev)
topk = torch.stack([torch.randperm(E, device=dev, generator=g)[:K] for _ in range(16)]).int()
out = torch.zeros(1, device=dev)

def step():
    cur = torch.cuda.current_stream()
    side.wait_stream(cur)
    with torch.cuda.stream(side):
        st.stage(topk, cmap, src)
    c = a @ a                                            # HBM-heavy work on the capture stream meanwhile
    cur.wait_stream(side)
    out.copy_(st.w13[:4].float().sum() + c[0, 0].float())  # consume staged bytes after the join

step(); torch.cuda.synchronize()
s = torch.cuda.Stream(dev)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=s):
    step()
for trial in range(4):
    topk.copy_(torch.stack([torch.randperm(E, device=dev, generator=g)[:K] for _ in range(16)]).int())
    graph.replay(); torch.cuda.synchronize()
    em = st.emap.tolist()
    for e, slot in enumerate(em):
        if slot >= 0:
            c = int(cmap[e])
            assert torch.equal(st.w13[slot].cpu(), w13_cpu[c]) and torch.equal(st.w2[slot].cpu(), w2_cpu[c])
            assert torch.equal(st.s13[slot], s13[c]) and torch.equal(st.s2[slot], s2[c])
    print(f"overlap graph replay {trial}: {int(st.nused)} staged, bytes exact")
# timing: overlapped vs serial, inside graphs
def timed(fn, reps=20):
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, stream=s):
        fn()
    gr.replay(); torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps): gr.replay()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) / reps
def serial():
    st.stage(topk, cmap, src); c = a @ a; out.copy_(st.w13[:4].float().sum() + c[0, 0].float())
print(f"graph step: overlapped {timed(step):.3f} ms, serial {timed(serial):.3f} ms, stage alone "
      f"{timed(lambda: st.stage(topk, cmap, src)):.3f} ms, matmul alone {timed(lambda: a @ a):.3f} ms")
print("OVERLAP OK")
