"""Run a Python script with the global `random` module seeded (BENCH_SEED, default 1234), so llm_decode_bench's
padding text and run_id repeat exactly across servers: identical prompts make spec-decode A/Bs comparable."""
import os, random, runpy, sys
random.seed(int(os.environ.get("BENCH_SEED", "1234")))
script = sys.argv[1]; sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
