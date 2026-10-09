import time
import torch
x = torch.empty(258 * 4608 * 40, dtype=torch.int32, device="cuda")
y = torch.empty_like(x)
def b(f, n=50):
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e6
nb = x.numel() * 4
t = b(lambda: x.fill_(7)); print(f"fill 181MB: {t:.1f} us, {nb / t / 1e3:.0f} GB/s written")
t = b(lambda: y.copy_(x)); print(f"copy 181MB: {t:.1f} us, {nb / t / 1e3:.0f} GB/s written ({2 * nb / t / 1e3:.0f} GB/s moved)")
