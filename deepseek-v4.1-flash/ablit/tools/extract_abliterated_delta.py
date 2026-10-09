#!/usr/bin/env python3
"""Extract the abliterated DeepSeek-V4.1-Flash weight delta by HTTP range requests.

The abliterated checkpoint at distributedcognition/DeepSeek-V4.1-Flash-abliterated (gated) mirrors the
upstream 48-shard layout and differs only in `attn.wo_b` and `ffn.shared_experts.w2` for layers 4-36:
132 tensors, ~1.65 GiB of 475 GiB. This tool pulls exactly those byte ranges, checks them against the local
official checkpoint, spot-checks tensors that must be unchanged, and writes a delta (~1.7 GiB) that
`../ablit_switch.py` applies in place. See ../PLAN-abliterated-switch.md.

Range requests are verified (206 plus an exact Content-Range) so a CDN that ignores Range can never turn this
into a 475 GiB download, and every spot-check tensor is size capped (an Engram table here is 93 GiB).

Host-side, standard library only. Usage:

  export HF_TOKEN=hf_...
  python3 tools/extract_abliterated_delta.py                 # writes ../data/ by default

  # same mechanics against the ungated official repo, to self-test without a token (expect all-identical):
  python3 tools/extract_abliterated_delta.py --repo deepseek-ai/DeepSeek-V4.1-Flash --out-dir /tmp/ablit-selftest

  python3 tools/extract_abliterated_delta.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request

TENSOR_SUFFIXES = (
    "attn.wo_b.weight",
    "attn.wo_b.scale",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w2.scale",
)
SCHEMA = "dsv41-abliterated-delta/1"
HF = "https://huggingface.co"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "data"))
# Safety net: the biggest tensor this tool ever needs is the 40 MiB wo_b weight. Anything larger means a size
# cap was applied to the wrong field (an Engram table here is 93 GiB).
HARD_FETCH_CAP = 256 * 2**20


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- local checkpoint


class LocalCheckpoint:
    """Read tensor bytes out of a local safetensors checkpoint (headers cached, no full-file reads)."""

    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.index = json.load(f)["weight_map"]
        self._headers: dict[str, tuple[int, dict]] = {}

    def header(self, shard: str) -> tuple[int, dict]:
        if shard not in self._headers:
            with open(os.path.join(self.root, shard), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._headers[shard] = (8 + n, json.loads(f.read(n)))
        return self._headers[shard]

    def names(self) -> list[str]:
        return list(self.index)

    def meta(self, name: str) -> tuple[str, int, int, dict]:
        shard = self.index[name]
        base, hdr = self.header(shard)
        m = hdr[name]
        return shard, base + m["data_offsets"][0], m["data_offsets"][1] - m["data_offsets"][0], m

    def read(self, name: str) -> bytes:
        shard, start, nbytes, _ = self.meta(name)
        with open(os.path.join(self.root, shard), "rb") as f:
            f.seek(start)
            data = f.read(nbytes)
        if len(data) != nbytes:
            raise IOError(f"short read for {name}: {len(data)} != {nbytes}")
        return data


# --------------------------------------------------------------------------- remote shards


class RemoteShard:
    """One safetensors shard in the remote repo, read through HTTP range requests."""

    def __init__(self, repo: str, shard: str, token: str | None, retries: int = 4):
        self.repo = repo
        self.shard = shard
        self.url = f"{HF}/{repo}/resolve/main/{shard}"
        self.token = token
        self.retries = retries
        first = self._get(0, 7)
        n = struct.unpack("<Q", first)[0]
        self.header_bytes = n
        self.base = 8 + n
        self.header = json.loads(self._get(8, 8 + n - 1).decode())
        self.total = self._total

    def _get(self, start: int, end: int) -> bytes:
        headers = {"Range": f"bytes={start}-{end}", "User-Agent": "dsv41-ablit-delta/1"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as r:
                    status = r.status
                    crange = r.headers.get("Content-Range") or ""
                    data = r.read()
                    if status != 206:
                        raise RuntimeError(
                            f"{self.shard}: server answered {status} for a range request (Content-Length "
                            f"{r.headers.get('Content-Length')}); it is ignoring HTTP ranges"
                        )
                    want_prefix = f"bytes {start}-{end}/"
                    if not crange.startswith(want_prefix):
                        raise RuntimeError(f"{self.shard}: Content-Range {crange!r} does not match {start}-{end}")
                    self._total = int(crange.split("/")[1])
                    if len(data) != end - start + 1:
                        raise RuntimeError(f"{self.shard}: short range read {len(data)} != {end - start + 1}")
                    return data
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise SystemExit(
                        f"{self.shard}: HTTP {e.code}. The repo is gated: pass a token that has accepted the "
                        f"terms (--token / --token-file / HF_TOKEN)."
                    ) from e
                last = e
            except Exception as e:  # transient network or range problems
                last = e
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"{self.shard}: range fetch failed: {last!r}")

    def read(self, offset: int, nbytes: int) -> bytes:
        if nbytes > HARD_FETCH_CAP:
            raise RuntimeError(
                f"{self.shard}: refusing a {nbytes / 2**30:.1f} GiB range; the target tensors are at most "
                f"40 MiB, so this is a bug (size caps are applied before fetching)"
            )
        return self._get(self.base + offset, self.base + offset + nbytes - 1)


# --------------------------------------------------------------------------- plan


def target_names(layers: range) -> list[str]:
    return [f"layers.{i}.{s}" for i in layers for s in TENSOR_SUFFIXES]


def spot_check_names(local: LocalCheckpoint, layers: list[int], per_category: int,
                     max_mib: float, budget_mib: float) -> tuple[list[str], list[tuple[str, str]]]:
    """Tensors the abliteration claims are untouched, plus neighbours of the changed ones.

    Sizes matter: an Engram table here is 93 GiB and embed/head are 1.26 GiB each, so every candidate is size
    capped and the total is budgeted. Skips are reported rather than silently dropped.
    """
    names = local.names()
    picks: list[str] = []
    skipped: list[tuple[str, str]] = []
    budget = budget_mib * 2**20
    used = 0

    def take(pred, label, n=per_category):
        nonlocal used
        hits = sorted((local.meta(x)[2], x) for x in names if pred(x))
        if not hits:
            skipped.append((label, "no tensor matched"))
            return
        added = 0
        for nbytes, x in hits:
            if added >= n:
                break
            if nbytes > max_mib * 2**20:
                skipped.append((x, f"{nbytes / 2**20:.0f} MiB exceeds the {max_mib:.0f} MiB spot cap"))
                continue
            if used + nbytes > budget:
                skipped.append((label, f"spot budget of {budget_mib:.0f} MiB is exhausted"))
                break
            if x in picks:
                continue
            picks.append(x)
            used += nbytes
            added += 1
        if added == 0 and not any(s[0] == label for s in skipped):
            skipped.append((label, "nothing within the size cap"))

    take(lambda n: ".ffn.experts." in n and n.endswith(".w2.weight"), "routed expert weight")
    take(lambda n: ".ffn.experts." in n and n.endswith(".w1.scale"), "routed expert scale")
    take(lambda n: n.endswith(".ffn.gate.bias"), "router bias")
    take(lambda n: ".ffn.shared_experts.w1.weight" in n, "shared gate_up (not ablated)")
    take(lambda n: ".attn.wkv.weight" in n, "dense attention (not ablated)")
    take(lambda n: n.startswith("vision."), "vision")
    take(lambda n: n.startswith("norm."), "final norm")
    take(lambda n: "engram" in n.lower(), "engram")
    take(lambda n: n.startswith(("head.", "embed.")), "head/embed")
    # Routed experts of an ablated layer and of a layer outside the range, covering both sides of it.
    take(lambda n: f"layers.{layers[0]}.ffn.experts.0.w2." in n, f"layer {layers[0]} experts")
    take(lambda n: f"layers.{layers[-1] + 1}.ffn.experts.0.w2." in n, "experts outside the range")
    return picks, skipped


# --------------------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=DEFAULT_OUT, help="where the delta lands (created if missing)")
    p.add_argument("--repo", default="distributedcognition/DeepSeek-V4.1-Flash-abliterated")
    p.add_argument("--official-root", default=os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))
    p.add_argument("--official-repo", default="deepseek-ai/DeepSeek-V4.1-Flash")
    p.add_argument("--layers", default="4-36", help="ablated layer range, e.g. 4-36")
    p.add_argument("--token", default=None)
    p.add_argument("--token-file", default=os.path.expanduser("~/.cache/huggingface/token"))
    p.add_argument("--spot", type=int, default=1, help="tensors per spot-check category")
    p.add_argument("--spot-max-mib", type=float, default=64.0,
                   help="skip spot-check tensors larger than this (an Engram table here is 93 GiB)")
    p.add_argument("--spot-budget-mib", type=float, default=512.0, help="total spot-check download budget")
    p.add_argument("--limit", type=int, default=0, help="only the first N target tensors (for testing)")
    p.add_argument("--verify-only", action="store_true", help="re-check an existing delta against the local files")
    p.add_argument("--no-spot", action="store_true")
    a = p.parse_args()

    lo, hi = (int(x) for x in a.layers.split("-"))
    layers = list(range(lo, hi + 1))
    os.makedirs(a.out_dir, exist_ok=True)
    bin_path = os.path.join(a.out_dir, "abliterated-delta.bin")
    json_path = os.path.join(a.out_dir, "abliterated-delta.json")
    report_path = os.path.join(a.out_dir, "delta-report.json")

    local = LocalCheckpoint(a.official_root)
    names = target_names(layers)
    if a.limit:
        names = names[: a.limit]
    missing = [n for n in names if n not in local.index]
    if missing:
        raise SystemExit(f"{len(missing)} target tensors are absent from the local checkpoint, e.g. {missing[:3]}")

    if a.verify_only:
        if not os.path.exists(json_path):
            raise SystemExit(f"{json_path} does not exist; run without --verify-only first")
        with open(json_path) as f:
            delta = json.load(f)
        bad = changed_ok = identical_ok = 0
        with open(bin_path, "rb") as fb:
            for t in delta["tensors"]:
                fb.seek(t["bin_offset"])
                data = fb.read(t["nbytes"])
                if len(data) != t["nbytes"] or hashlib.sha256(data).hexdigest() != t["sha256"]:
                    log(f"MISMATCH blob {t['name']}: wrong size or sha256")
                    bad += 1
                    continue
                same = local.read(t["name"]) == data
                # The delta is expected to differ from the official checkpoint exactly where it says it does,
                # and to match it everywhere else (both are true for an identity self-test delta).
                if same == bool(t["changed"]):
                    log(f"MISMATCH {t['name']}: the index says changed={t['changed']} but the bytes are "
                        f"{'identical to' if same else 'different from'} the official checkpoint")
                    bad += 1
                elif t["changed"]:
                    changed_ok += 1
                else:
                    identical_ok += 1
        log(f"verify-only: {len(delta['tensors'])} tensors, {changed_ok} differ from official as recorded, "
            f"{identical_ok} identical as recorded, {bad} mismatches")
        return 1 if bad else 0

    token = a.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token is None and os.path.exists(a.token_file):
        token = open(a.token_file).read().strip()
    if token is None:
        log("note: no token supplied; this only works for ungated repos")

    # Group the targets by remote shard: the layout mirrors upstream, but fetch headers rather than assume it.
    shards = sorted({local.index[n] for n in names})
    log(f"{len(names)} target tensors across {len(shards)} shards; fetching remote headers")
    remote: dict[str, RemoteShard] = {}
    for shard in shards:
        remote[shard] = RemoteShard(a.repo, shard, token)
        base, hdr = local.header(shard)
        r_tensors = set(remote[shard].header)
        l_tensors = set(hdr)
        if r_tensors != l_tensors:
            raise SystemExit(
                f"{shard}: tensor sets differ (local {len(l_tensors)}, remote {len(r_tensors)}); "
                f"the remote checkpoint does not mirror upstream"
            )

    tensors = []
    changed = 0
    identical = 0
    offsets_differ = 0
    seen_names = []
    t0 = time.time()
    with open(bin_path, "wb", buffering=1024 * 1024) as fb:
        for i, name in enumerate(names, 1):
            shard, _, nbytes, m = local.meta(name)
            r = remote[shard]
            rm = r.header[name]
            if rm["shape"] != m["shape"] or rm["dtype"] != m["dtype"]:
                raise SystemExit(f"{name}: remote {rm['dtype']}{rm['shape']} != local {m['dtype']}{m['shape']}")
            if rm["data_offsets"] != m["data_offsets"]:
                offsets_differ += 1
            data = r.read(rm["data_offsets"][0], rm["data_offsets"][1] - rm["data_offsets"][0])
            loc = local.read(name)
            same = data == loc
            if same:
                identical += 1
            else:
                changed += 1
            tensors.append(
                {
                    "name": name,
                    "dtype": m["dtype"],
                    "shape": m["shape"],
                    "nbytes": nbytes,
                    "bin_offset": fb.tell(),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "changed": not same,
                    "remote_shard": shard,
                    "remote_offset": rm["data_offsets"][0],
                }
            )
            seen_names.append(name)
            fb.write(data)
            if i % 8 == 0 or i == len(names):
                log(f"  {i}/{len(names)} tensors, {fb.tell() / 2**20:.1f} MiB, changed {changed}")

    spot, spot_skipped = ([], []) if a.no_spot else spot_check_names(
        local, layers, a.spot, a.spot_max_mib, a.spot_budget_mib)
    spot_results = []
    for name in spot:
        shard, _, nbytes, m = local.meta(name)
        r = remote.setdefault(shard, RemoteShard(a.repo, shard, token))
        rm = r.header.get(name)
        if rm is None:
            spot_results.append({"name": name, "identical": None, "error": "absent in remote shard"})
            continue
        data = r.read(rm["data_offsets"][0], rm["data_offsets"][1] - rm["data_offsets"][0])
        spot_results.append(
            {
                "name": name,
                "shard": shard,
                "identical": data == local.read(name),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )

    repo_sha = None
    try:
        api = json.load(urllib.request.urlopen(f"{HF}/api/models/{a.repo}", timeout=30))
        repo_sha = api.get("sha")
    except Exception as e:
        log(f"note: could not read the repo revision: {e!r}")

    delta = {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo": a.repo,
        "repo_sha": repo_sha,
        "official_root": a.official_root,
        "official_repo": a.official_repo,
        "layers": [lo, hi],
        "tensor_suffixes": list(TENSOR_SUFFIXES),
        "totals": {
            "tensors": len(tensors),
            "bytes": sum(t["nbytes"] for t in tensors),
            "changed": changed,
            "identical": identical,
            "offset_mismatches_vs_local": offsets_differ,
        },
        "tensors": tensors,
    }
    report = {
        "schema": SCHEMA + "+report",
        "created": delta["created"],
        "repo": a.repo,
        "repo_sha": repo_sha,
        "changed": changed,
        "identical": identical,
        "spot_checks": spot_results,
        "spot_skipped": [{"name": n, "why": w} for n, w in spot_skipped],
        "all_spot_checks_identical": all(s.get("identical") for s in spot_results) if spot_results else None,
    }
    with open(json_path, "w") as f:
        json.dump(delta, f, indent=1)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=1)

    log("")
    log(f"delta: {len(tensors)} tensors, {delta['totals']['bytes'] / 2**20:.1f} MiB -> {bin_path}")
    log(f"changed {changed}, identical {identical}, shard-header offset mismatches {offsets_differ}")
    for s in spot_results:
        log(f"spot check {'OK ' if s.get('identical') else 'FAIL'} {s['name']}")
    for name, why in spot_skipped:
        log(f"spot check skipped {name}: {why}")
    log(f"elapsed {time.time() - t0:.0f}s")
    if identical:
        log(f"WARNING: {identical} target tensors are byte-identical to upstream; check the layer range")
    if changed == 0:
        log("ERROR: no tensor differs; wrong repo or wrong target list")
        return 1
    if spot_results and not report["all_spot_checks_identical"]:
        log("ERROR: a tensor that should be untouched differs; stop and re-derive the target list")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
