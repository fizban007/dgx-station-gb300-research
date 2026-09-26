"""Per-layer GB300 -> RTX PRO 6000 -> GB300 round trip through pinned Grace memory.

role=main (GB300): per iteration, publish a 10 KB activation, run `--hot-us` of HBM
streaming (stand-in for the hot MoE), then wait for the peer's result.
role=peer (6000): wait for the activation, stream `--cold-mb` of VRAM (stand-in for
cold experts), publish a 10 KB result. Both sides enqueue every iteration up front.
"""
import argparse, mmap, os, time
import torch
from peer_ext import load

p = argparse.ArgumentParser()
p.add_argument("--role", choices=["main", "peer", "peer-bw"], required=True)
p.add_argument("--iters", type=int, default=400)
p.add_argument("--base", type=int, default=1)
p.add_argument("--hot-mb", type=float, default=0.0)
p.add_argument("--cold-mb", type=float, default=18.3)
p.add_argument("--act-bytes", type=int, default=10240)
p.add_argument("--no-peer", action="store_true")
a = p.parse_args()

arch = "10.3a" if a.role == "main" else "12.0a"
ext = load(arch)
dev = torch.device("cuda", 0)
torch.cuda.set_device(dev)
SIZE = 1 << 20
if a.role != "peer-bw":
    fd = os.open("/dev/shm/peer_buf", os.O_RDWR | os.O_CREAT, 0o600)
    os.ftruncate(fd, SIZE)
    buf = mmap.mmap(fd, SIZE)
    host = torch.frombuffer(buf, dtype=torch.uint8)
    hptr = host.data_ptr()
    ext.host_register(hptr, SIZE)
    dptr = ext.device_pointer(hptr)
    go, done, act_h, out_h = dptr, dptr + 64, dptr + 4096, dptr + 4096 + 65536
n = a.act_bytes
act = torch.randn(n // 2, device=dev, dtype=torch.bfloat16)
res = torch.empty_like(act)
sink = torch.zeros(1, device=dev)
sms = torch.cuda.get_device_properties(dev).multi_processor_count
hot = torch.empty(max(16, int(a.hot_mb * 1e6)) // 16 * 16, dtype=torch.uint8, device=dev)
cold = torch.empty(max(16, int(a.cold_mb * 1e6)) // 16 * 16, dtype=torch.uint8, device=dev)


def timed(fn, iters):
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn(iters)
    e.record(); e.synchronize()
    return s.elapsed_time(e) * 1000 / iters


if a.role == "peer-bw":
    for mb in (1.0, 18.3, 36.6, 73.2, 146.4):
        w = torch.empty(int(mb * 1e6) // 16 * 16, dtype=torch.uint8, device=dev)
        us = timed(lambda k: [ext.stream(w.data_ptr(), w.numel(), sink.data_ptr(), sms * 4) for _ in range(k)], 200)
        print(f"6000 stream {mb:6.1f} MB: {us:7.1f} us  {w.numel()/us/1e3:6.0f} GB/s", flush=True)
elif a.role == "peer":
    def run(k):
        for i in range(k):
            ext.recv(act_h, act.data_ptr(), n, go, a.base + i, 1)
            ext.stream(cold.data_ptr(), cold.numel(), sink.data_ptr(), sms * 4)
            ext.send(act.data_ptr(), out_h, n, done, a.base + i, 1)
    us = timed(run, a.iters)
    print(f"peer: {us:.1f} us/iter (includes waiting)", flush=True)
else:
    def run(k):
        for i in range(k):
            if not a.no_peer:
                ext.send(act.data_ptr(), act_h, n, go, a.base + i, 1)
            if a.hot_mb:
                ext.stream(hot.data_ptr(), hot.numel(), sink.data_ptr(), sms * 4)
            if not a.no_peer:
                ext.recv(out_h, res.data_ptr(), n, done, a.base + i, 1)
    us = timed(run, a.iters)
    print(f"main: {us:.1f} us/iter (hot {a.hot_mb} MB, peer {'off' if a.no_peer else 'on'})", flush=True)
