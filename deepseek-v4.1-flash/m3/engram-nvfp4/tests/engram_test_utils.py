# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the NVFP4 Engram overlay tests (run inside the af7f9488 image).

Rows are read straight from the safetensors shards with O_DIRECT, so sampling random rows
neither fills the host page cache nor reads a 64 KiB page per 144-byte row.
"""
import json
import math
import mmap
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

CKPT = "/data/checkpoints"
FP8_DIR = f"{CKPT}/DeepSeek-V4.1-Flash"
NVFP4_DIR = f"{CKPT}/DeepSeek-V4.1-Flash-engram-nvfp4/hf"
MERGED_DIR = f"{CKPT}/DeepSeek-V4.1-Flash-engram-nvfp4-merged"
SHARD = {1: "model-00047-of-00048.safetensors", 14: "model-00048-of-00048.safetensors"}
RESULTS = os.environ.get("RESULTS", "/work/tests/results")
ALIGN = 4096
_DTYPE_BYTES = {"U8": 1, "F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2, "F32": 4}


def global_scales() -> dict[int, float]:
    with open(f"{NVFP4_DIR}/config.json") as f:
        raw = json.load(f)["text_config"]["engram_quant_global_scale"]
    return {int(k): float(v) for k, v in raw.items()}


def st_header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return n, h


class ShardTensor:
    """One 2-D tensor of a safetensors shard, read by row with O_DIRECT."""

    def __init__(self, path: str, name: str) -> None:
        n, h = st_header(path)
        info = h[name]
        self.path, self.name = path, name
        self.dtype = info["dtype"]
        self.shape = info["shape"]
        self.start = 8 + n + info["data_offsets"][0]
        self.row_bytes = math.prod(self.shape[1:]) * _DTYPE_BYTES[self.dtype]
        assert info["data_offsets"][1] - info["data_offsets"][0] == self.shape[0] * self.row_bytes

    def _span(self, off: int, length: int) -> tuple[int, int]:
        a0 = off - off % ALIGN
        a1 = -(-(off + length) // ALIGN) * ALIGN
        return a0, a1

    def read_rows(self, rows: np.ndarray, threads: int = 32) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        out = np.empty((rows.size, self.row_bytes), dtype=np.uint8)

        def work(lo: int, hi: int) -> None:
            fd = os.open(self.path, os.O_RDONLY | os.O_DIRECT)
            buf = mmap.mmap(-1, 2 * ALIGN + self.row_bytes)
            view = memoryview(buf)
            try:
                for i in range(lo, hi):
                    off = self.start + int(rows[i]) * self.row_bytes
                    a0, a1 = self._span(off, self.row_bytes)
                    got = os.preadv(fd, [view[: a1 - a0]], a0)
                    assert got >= off - a0 + self.row_bytes
                    out[i] = np.frombuffer(buf, np.uint8, self.row_bytes, off - a0)
            finally:
                view.release()
                buf.close()
                os.close(fd)

        step = -(-rows.size // threads)
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(lambda lo: work(lo, min(lo + step, rows.size)), range(0, rows.size, step)))
        return out

    def read_range(self, row0: int, nrows: int, chunk: int = 256 << 20) -> np.ndarray:
        out = np.empty((nrows, self.row_bytes), dtype=np.uint8)
        flat = out.reshape(-1)
        total = nrows * self.row_bytes
        fd = os.open(self.path, os.O_RDONLY | os.O_DIRECT)
        buf = mmap.mmap(-1, chunk + 2 * ALIGN)
        view = memoryview(buf)
        try:
            done = 0
            while done < total:
                n = min(chunk, total - done)
                off = self.start + row0 * self.row_bytes + done
                a0, a1 = self._span(off, n)
                got = os.preadv(fd, [view[: a1 - a0]], a0)
                assert got >= off - a0 + n
                flat[done : done + n] = np.frombuffer(buf, np.uint8, n, off - a0)
                done += n
        finally:
            view.release()
            buf.close()
            os.close(fd)
        return out


def nvfp4_tensors(layer: int, root: str = NVFP4_DIR) -> tuple[ShardTensor, ShardTensor]:
    path = f"{root}/{SHARD[layer]}"
    return (
        ShardTensor(path, f"layers.{layer}.engram.embed.weight"),
        ShardTensor(path, f"layers.{layer}.engram.embed.scale"),
    )


def fp8_tensors(layer: int) -> tuple[ShardTensor, ShardTensor]:
    path = f"{FP8_DIR}/{SHARD[layer]}"
    return (
        ShardTensor(path, f"layers.{layer}.engram.embed.weight"),
        ShardTensor(path, f"layers.{layer}.engram.embed.scale"),
    )


# ---- reference dequantization (CPU, fp32 math, one rounding to bf16 like the kernels) ----
E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def ref_nvfp4_fp32(packed: np.ndarray, scales: np.ndarray, g: float) -> torch.Tensor:
    """(e2m1 * e4m3) * global in fp32; packed [n, 128] u8 (low nibble = even), scales [n, 16] u8."""
    p = torch.from_numpy(packed)
    codes = torch.stack((p & 0x0F, p >> 4), dim=-1).reshape(p.shape[0], -1).long()
    values = E2M1[codes]
    s = torch.from_numpy(scales).view(torch.float8_e4m3fn).float()
    s = s.repeat_interleave(values.shape[1] // s.shape[1], dim=1)
    return (values * s) * torch.tensor(g, dtype=torch.float32)


def ref_nvfp4(packed: np.ndarray, scales: np.ndarray, g: float) -> torch.Tensor:
    return ref_nvfp4_fp32(packed, scales, g).to(torch.bfloat16)


def ref_fp8_fp32(values: np.ndarray, e8m0: np.ndarray) -> torch.Tensor:
    """e4m3 * 2^(e-127) in fp32 (the kernel builds the scale as the exponent field)."""
    v = torch.from_numpy(values).view(torch.float8_e4m3fn).float()
    s = (torch.from_numpy(e8m0).to(torch.int32) << 23).view(torch.float32)
    return v * s.repeat_interleave(v.shape[1] // s.shape[1], dim=1)


def ref_fp8(values: np.ndarray, e8m0: np.ndarray) -> torch.Tensor:
    return ref_fp8_fp32(values, e8m0).to(torch.bfloat16)


# ---- vLLM plumbing ----
_INIT = False
_CONTEXTS: list = []


def init_vllm(max_tokens: int = 8192, cpu_offload: bool = True, use_thp: bool = False):
    """World-size-1 TP group plus a current VllmConfig carrying an EngramConfig."""
    global _INIT
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.engram import EngramConfig

    cfg = VllmConfig()
    cfg.scheduler_config.max_num_batched_tokens = max_tokens
    cfg.engram_config = EngramConfig(
        cpu_offload=cpu_offload, use_thp=use_thp, dp_shared_memory=False
    )
    # Keep the context alive: a dropped generator context resets the config on GC.
    ctx = set_current_vllm_config(cfg)
    ctx.__enter__()
    _CONTEXTS.append(ctx)
    if not _INIT:
        from vllm.distributed import init_distributed_environment, initialize_model_parallel

        port = 29500 + os.getpid() % 1000
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(1, 1)
        _INIT = True
    return cfg


def head_split(rows: int, heads: int = 24) -> tuple[int, ...]:
    base, extra = divmod(rows, heads)
    return tuple(base + (1 if h < extra else 0) for h in range(heads))


def make_embedding(rows: int, quant: str, *, cpu_offload: bool = True, use_thp: bool = False,
                   global_scale: float | None = None, heads: int = 24):
    """nvidia ParallelEngramEmbedding over `rows` rows split into `heads` hash heads."""
    from vllm.models.deepseek_v41.nvidia.engram import ParallelEngramEmbedding

    kwargs = {}
    if quant == "nvfp4":
        kwargs = {"block_size": 16, "quant": "nvfp4", "global_scale": global_scale}
    sizes = head_split(rows, heads)
    with torch.device("cuda"):
        emb = ParallelEngramEmbedding(
            rows, 256, sizes, cpu_offload=cpu_offload, use_thp=use_thp, **kwargs
        )
    emb._test_head_sizes = sizes
    return emb


def load_rows(emb, values: np.ndarray, scales: np.ndarray, quant: str) -> None:
    """Feed rows through the parameters' own weight loaders, with checkpoint dtypes."""
    v = torch.from_numpy(values)
    s = torch.from_numpy(scales)
    if quant == "nvfp4":
        s = s.view(torch.float8_e4m3fn)  # checkpoint: U8 values, F8_E4M3 scales
    else:
        v = v.view(torch.float8_e4m3fn)  # checkpoint: F8_E4M3 values, F8_E8M0 scales
        s = s.view(torch.float8_e8m0fnu)
    emb.weight.weight_loader(emb.weight, v)
    emb.weight_scale_inv.weight_loader(emb.weight_scale_inv, s)


def make_ids(tokens: int, emb, gen: torch.Generator, layers: int = 2) -> torch.Tensor:
    """[tokens, heads] int32 hash ids as the model passes them: a strided view of a
    [tokens, layers, heads] buffer, each head's ids inside its own bucket range."""
    sizes = torch.tensor(emb_head_sizes(emb), dtype=torch.int64)
    starts = torch.cumsum(sizes, 0) - sizes
    r = torch.rand((tokens, sizes.numel()), generator=gen, dtype=torch.float64)
    ids = (starts + (r * sizes).long().clamp_max(sizes - 1)).to(torch.int32)
    buf = torch.full((tokens, layers, sizes.numel()), -1, dtype=torch.int32)
    buf[:, 0] = ids
    return buf.cuda()[:, 0]


def emb_head_sizes(emb) -> tuple[int, ...]:
    return emb._test_head_sizes


def rss_gib() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 2**20
    return float("nan")


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def write_result(name: str, payload: dict) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=str)
        f.write("\n")
    return path


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.s = time.time() - self.t0
