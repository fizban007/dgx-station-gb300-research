"""Encode raw MXFP4 E8M0 block scales into b12x CSF scale planes (moe.CsfScalePlanes).

Per expert, rows are grouped in 16-row slabs; each slab is 16 row-base bytes followed by the slab's selector bits
(one per scale, packed little-endian per row: value = base + bit). Scales outside {base, base+1} go to a sorted
uint32 exception stream: flat position (row * columns + column, < 2^24) | value << 24. Lossless.
"""
import torch


def encode_planes(scales: torch.Tensor, batch: int = 8):
    """scales: uint8 [E, rows, columns] (CPU or CUDA). Returns (fixed, exceptions) tuples of CPU tensors per expert."""
    fixed, exceptions = (), ()
    for s in range(0, scales.shape[0], batch):
        f, x = _encode(scales[s:s + batch])
        fixed, exceptions = fixed + f, exceptions + x
    return fixed, exceptions


def _encode(scales: torch.Tensor):
    E, rows, cols = scales.shape
    assert rows % 16 == 0 and cols % 8 == 0 and rows * cols < 2**24, scales.shape
    x = scales.to(torch.int64)
    hist = torch.zeros(E, rows, 257, dtype=torch.int32, device=x.device)
    hist.scatter_add_(2, x, torch.ones_like(x, dtype=torch.int32))
    base = (hist[..., :255] + hist[..., 1:256]).argmax(dim=2)                 # [E, rows], <= 254
    d = x - base[..., None]
    bit = d == 1
    exc = (d != 0) & ~bit
    weights = (1 << torch.arange(8, device=x.device))
    sel = (bit.view(E, rows, cols // 8, 8).long() * weights).sum(-1).to(torch.uint8)   # [E, rows, cols/8]
    fixed = torch.cat((base.to(torch.uint8).view(E, rows // 16, 16), sel.view(E, rows // 16, 16 * cols // 8)), 2)
    fixed = fixed.reshape(E, -1).cpu()
    flat_exc, flat_x = exc.view(E, -1).cpu(), x.view(E, -1).cpu()
    exceptions = []
    for e in range(E):
        pos = flat_exc[e].nonzero().squeeze(1)                                 # ascending
        exceptions.append((pos | (flat_x[e, pos] << 24)).to(torch.uint32))
    return tuple(fixed[e].clone() for e in range(E)), tuple(exceptions)


def planes_nbytes(fixed, exceptions):
    return sum(t.numel() for t in fixed) + 4 * sum(t.numel() for t in exceptions)
