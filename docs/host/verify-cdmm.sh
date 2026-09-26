#!/usr/bin/env bash
# Post-reboot check for driver-managed coherent memory (NVreg_CoherentGPUMemoryMode=driver) and the GRUB timeout.
echo "--- driver param (expect \"driver\")"; grep -E "CoherentGPUMemoryMode|EnableUserNUMAManagement" /proc/driver/nvidia/params
echo "--- NUMA nodes (expect node 1 size 0 MB)"; numactl -H | grep -E "node [01] (size|free)"
echo "--- free (expect ~506 GB total)"; free -g | head -2
echo "--- nvidia-smi"; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo "--- zone_reclaim_mode (1)"; cat /proc/sys/vm/zone_reclaim_mode
echo "--- grub timeout (expect 5)"; sudo grep -n -A1 'grub_platform = efi' /boot/grub/grub.cfg | head -3
echo "--- node0 Shmem (was 66 GB before reboot)"; numastat -m 2>/dev/null | awk '/^Shmem /{print $2 " MB"}'
