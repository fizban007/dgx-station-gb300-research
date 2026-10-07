#!/usr/bin/env python3
"""Compare fp8_regress_stock.json and fp8_regress_overlay.json (host-side)."""
import json
import os
import sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
a = json.load(open(os.path.join(R, "fp8_regress_stock.json")))
b = json.load(open(os.path.join(R, "fp8_regress_overlay.json")))
diff_cases = [k for k in a["cases"] if a["cases"][k] != b["cases"].get(k)]
pa, pb = dict(a["real_size_params"]), dict(b["real_size_params"])
quant = (pa.pop("layout_quant"), pb.pop("layout_quant"))
diff_params = [k for k in pa if pa[k] != pb.get(k)]
out = {
    "cases_compared": len(a["cases"]),
    "cases_identical": len(a["cases"]) - len(diff_cases),
    "differing_cases": diff_cases,
    "ref_bit_identical_stock": all(a["ref_bit_identical"].values()),
    "ref_bit_identical_overlay": all(b["ref_bit_identical"].values()),
    "real_size_fp8_params_identical": not diff_params and set(pa) == set(pb),
    "differing_param_entries": diff_params,
    "layout_quant": {"stock": quant[0], "overlay": quant[1]},
    "fp8_timing_fg_us": {"stock": a["timing_fg_us"], "overlay": b["timing_fg_us"]},
}
out["pass"] = (not diff_cases and out["ref_bit_identical_stock"] and out["ref_bit_identical_overlay"]
               and out["real_size_fp8_params_identical"])
with open(os.path.join(R, "fp8_regress_compare.json"), "w") as f:
    json.dump(out, f, indent=1)
    f.write("\n")
print(json.dumps(out, indent=1))
sys.exit(0 if out["pass"] else 1)
