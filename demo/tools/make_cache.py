"""Fill demo/cache/ with real Groq answers so the demo works without a key.

Needs GROQ_API_KEY in the environment (never pass it on the command line).
Usage: python demo/tools/make_cache.py [--lengths 8 24] [--sleep 15]
Each file stores the exact prompt hash, the model that answered and the time.
Only real answers are written; nothing is written when a call fails.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from method import FALLBACK_MODEL, PRIMARY_MODEL, TEMPERATURE, parse_response, temporal_prompt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lengths", type=int, nargs="+", default=[8, 24])
ap.add_argument("--sleep", type=float, default=15)
args = ap.parse_args()
key = os.environ.get("GROQ_API_KEY")
if not key:
    sys.exit("GROQ_API_KEY is not set; no cache written.")
from groq import Groq  # noqa: E402
client = Groq(api_key=key).with_options(max_retries=2, timeout=120.0)
pool = pd.read_csv(HERE / "demo_data" / "heldout_pool.csv")
for sid in sorted(pool.stretch_id.unique()):
    for n in args.lengths:
        seq = pool[pool.stretch_id.eq(sid)].sort_values("position").head(n).copy()
        seq["sentence_number"] = range(1, n + 1)
        prompt = temporal_prompt(seq)
        for model in (PRIMARY_MODEL, FALLBACK_MODEL):
            try:
                text = client.chat.completions.create(
                    model=model, temperature=TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}]).choices[0].message.content
                if not text or not text.strip():
                    raise ValueError("empty")
                break
            except Exception as err:  # noqa: BLE001
                print(f"stretch {sid} n={n} {model} failed: {type(err).__name__}")
                text = None
        if text is None:
            continue
        got = parse_response(text, n).llm_load_level.notna().sum()
        (HERE / "cache" / f"stretch{sid}_n{n:02d}.json").write_text(json.dumps({
            "stretch_id": int(sid), "n": n, "model": model, "temperature": TEMPERATURE,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "labels_parsed": int(got), "response": text}, indent=1))
        print(f"stretch {sid} n={n}: {model}, {got}/{n} labels")
        time.sleep(args.sleep)
