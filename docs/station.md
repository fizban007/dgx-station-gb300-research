# The station

All results in this repository come from one machine, measured 2026-09-23 to 2026-09-25.

| part | details |
|---|---|
| system | NVIDIA DGX Station GB300 (aarch64, Ubuntu 24.04, kernel `7.0.0-1019-nvidia-64k`) |
| GPU 0 | NVIDIA GB300: SM 10.3, 152 SMs, 256,703 MiB reported by `nvidia-smi` (about 250 GiB usable) |
| CPU and memory | Grace, 72 Neoverse-V2 cores, about 494 GiB LPDDR5X visible to Linux. The GB300 reads it coherently over NVLink-C2C. |
| GPU 1 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition: SM 12.0, 97,887 MiB, on PCIe. It has no NVLink to the GB300, and P2P is not enabled. |
| driver and CUDA | NVIDIA 595.91.07; CUDA 13.2 on the host. Containers bring their own CUDA. |

The GB300 serves models through vLLM or SGLang in Docker, selected by UUID. The RTX PRO 6000 runs the expert
sidecar as a plain host process in a Python venv ([sidecar design](sidecar-peer-tier.md)).

## Memory mode (changed 2026-09-25)

In the driver's default NUMA mode, the GB300's HBM is onlined as NUMA node 1, and Linux may place page cache
there. `nvidia-smi` and `cudaMemGetInfo` then count that cache as GPU memory in use, and host memory tools
report HBM as system RAM. We switched to driver-managed coherent memory (CDMM), which NVIDIA makes the
default on Station GB300 from driver R610:

| file | setting |
|---|---|
| [`host/nvidia-coherent-memory.conf`](host/nvidia-coherent-memory.conf) → `/etc/modprobe.d/` | `options nvidia NVreg_CoherentGPUMemoryMode=driver` (then refresh the initramfs and reboot) |
| [`host/60-gb300-hbm-no-pagecache.conf`](host/60-gb300-hbm-no-pagecache.conf) → `/etc/sysctl.d/` | `vm.zone_reclaim_mode = 1`: reclaim Grace cache locally rather than spilling into HBM. Moot under CDMM, but harmless. |
| [`host/60-boot-timeout.cfg`](host/60-boot-timeout.cfg) → `/etc/default/grub.d/` | `GRUB_RECORDFAIL_TIMEOUT=5`: `/boot` on md RAID otherwise always shows a 30 s boot menu |

Check the result with [`host/verify-cdmm.sh`](host/verify-cdmm.sh). NUMA node 1 should be 0 MB, and `free` should
show about 506 GB. Under CDMM the CPU cannot access GPU memory directly, `malloc`'d memory cannot migrate into
HBM, and MIG is unavailable. GPU access to Grace memory over C2C still works, and every lane here relies on
it. The switch happened on the morning of 2026-09-25: the DeepSeek-V4.1 runs and the early Qwen3.8 runs used
NUMA mode, and the MiMo runs used CDMM. The Qwen3.8 Rust-frontend A/B was re-run under CDMM (see
[`../qwen3.8-flash-next/`](../qwen3.8-flash-next/)).

## Host gotchas

- Docker's `--cpuset-mems` breaks CUDA initialisation on this host (error 802). Bind memory with host
  `numactl --membind=0` as the container entrypoint, plus `--cap-add SYS_NICE`. The launchers mount
  `/usr/bin/numactl` for this.
- Files in `/dev/shm` that another user created must be opened without `O_CREAT`. `fs.protected_regular`
  refuses `O_CREAT` on another user's file in a sticky directory, even for root.
- Models whose experts live in Grace need pinned host memory. Lanes that pin 300+ GiB drop the page cache
  before starting (`echo 3 > /proc/sys/vm/drop_caches`).
- For the station's measured Grace read bandwidth from the GB300, see the MiMo-Pro
  [DETAILS](../mimo-v2.6-pro/DETAILS.md#where-c1-decode-time-goes-v2-notes).
