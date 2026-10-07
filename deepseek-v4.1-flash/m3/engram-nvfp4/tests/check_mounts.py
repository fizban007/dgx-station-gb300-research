# SPDX-License-Identifier: Apache-2.0
"""Inside a container mounted like the lane: does /model read as the merged NVFP4 view, and do
the overlays (vllm#58132 + NVFP4 Engram) import together?"""
import glob
import json
import os
import struct
import sys

from vllm.model_executor.model_loader.weight_utils import filter_duplicate_safetensors_files
from vllm.transformers_utils.config import get_config

out = {"mode": os.environ.get("MOUNT_MODE")}
cfg = get_config("/model", trust_remote_code=True)
from vllm.models.deepseek_v41.common.engram import EngramLayout  # noqa: E402

layout = EngramLayout(cfg)
out["layout"] = [layout.quant, layout.quant_block_size, layout.global_scales]
files = sorted(glob.glob("/model/*.safetensors"))
kept = filter_duplicate_safetensors_files(files, "/model", "model.safetensors.index.json")
names, engram = {}, {}
for f in kept:
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    h.pop("__metadata__", None)
    for k, v in h.items():
        assert k not in names, f"duplicate tensor {k}"
        names[k] = os.path.basename(f)
        if ".engram.embed." in k:
            engram[k] = v["dtype"]
with open("/model/model.safetensors.index.json") as fh:
    wm = json.load(fh)["weight_map"]
out["shards"] = len(kept)
out["tensors"] = len(names)
out["tensors_equal_index"] = set(names) == set(wm)
out["engram_dtypes"] = engram
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("/model", trust_remote_code=True)
out["tokenizer_vocab"] = len(tok)
import vllm.models.deepseek_v41.nvidia.engram as nve  # noqa: E402
import vllm.models.deepseek_v41.nvidia.model as nvm  # noqa: E402

out["overlay_engram_loaded"] = hasattr(nve.ParallelEngramEmbedding, "_allocate_nvfp4_host_weights")
out["overlay_58132_loaded"] = os.path.exists(
    os.path.join(os.path.dirname(nvm.__file__), "..", "decoder_replay_layers.py"))
out["pass"] = (layout.quant == "nvfp4" and len(kept) == 48 and out["tensors_equal_index"]
               and set(engram.values()) == {"U8", "F8_E4M3"} and out["overlay_engram_loaded"])
print(json.dumps(out))
sys.exit(0 if out["pass"] else 1)
