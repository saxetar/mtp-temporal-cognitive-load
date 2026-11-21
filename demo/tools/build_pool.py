"""Build the held-out demo pool (run once, offline; needs the full NB05 output).

Input : speech_to_narrative_llm_input.csv from the full-data NB05 run (step s05).
        It already holds the per-corpus 0.25/0.75 levels used in Experiment 2.
        speech_to_narrative_temporal_evaluation_50_corrected.csv (Experiment 2 output),
        used only to find the conversations that must be excluded.
Output: demo_data/heldout_pool.csv  (6 stretches x 24 consecutive sentences)
        demo_data/pool_manifest.json

Selection (seed 42):
  1. Switchboard units only; drop every conversation that appears in Experiment 2
     (both sides, so no speaker or dialogue overlaps with the 396 sentences).
  2. Order each conversation side by the original sentence number (as NB06 c39).
  3. A candidate stretch = 24 sentences with consecutive original sentence numbers
     (no unit removed by the duration/token filter inside it), whose first four
     sentences have at least 10 tokens (the MIN_TOTAL_TOKENS rule of Exp. 2).
  4. Sample 6 conversation sides from 6 different conversations with
     random_state=42, then pick one candidate start per side with the same seed.
Usage: python demo/tools/build_pool.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED, N_STRETCHES, STRETCH_LEN, MIN_PREFIX, MIN_TOKENS = 42, 6, 24, 4, 10
FEATURES = ["duration_sec", "tokens", "speech_rate", "articulation_rate", "pause_ratio",
            "n_pauses", "mean_pause", "disfl_rate", "fp_rate", "rp_rate", "rv_rate",
            "pw_rate", "cluster_count"]
LEVELS = ["speech_rate_level", "pause_ratio_level", "disfl_rate_level", "cluster_level",
          "dominant_disfluency"]

src = Path(sys.argv[1])
here = Path(__file__).resolve().parents[1] / "demo_data"
full = pd.read_csv(src / "speech_to_narrative_llm_input.csv")
exp2 = pd.read_csv(src / "speech_to_narrative_temporal_evaluation_50_corrected.csv")
excluded_convs = sorted(exp2["conversation_id"].unique())
excluded_sides = sorted(exp2["conversation_side"].unique())

sw = full[full["dataset"].eq("Switchboard")].copy()
parts = sw["unit_id"].str.extract(r"^(?P<conversation_id>.+)_(?P<speaker_side>[AB])"
                                  r"_sentence_(?P<original_sentence_number>\d+)$")
sw[["conversation_id", "speaker_side", "original_sentence_number"]] = parts
sw["original_sentence_number"] = sw["original_sentence_number"].astype(int)
sw["conversation_side"] = sw["conversation_id"] + "_" + sw["speaker_side"]
sw = sw[~sw["conversation_id"].isin(excluded_convs)]
sw = sw.sort_values(["conversation_id", "speaker_side", "original_sentence_number"])

cands = {}
for side, g in sw.groupby("conversation_side", sort=True):
    g = g.reset_index(drop=True)
    num, tok = g["original_sentence_number"].to_numpy(), g["tokens"].to_numpy()
    starts = [i for i in range(len(g) - STRETCH_LEN + 1)
              if num[i + STRETCH_LEN - 1] - num[i] == STRETCH_LEN - 1
              and tok[i:i + MIN_PREFIX].sum() >= MIN_TOKENS]
    if starts:
        cands[side] = (g, starts)

rng = np.random.default_rng(SEED)
sides = pd.Series(sorted(cands)).sample(frac=1, random_state=SEED).tolist()
chosen, used_convs = [], set()
for side in sides:
    conv = side.rsplit("_", 1)[0]
    if conv in used_convs:
        continue
    chosen.append(side)
    used_convs.add(conv)
    if len(chosen) == N_STRETCHES:
        break

rows = []
for k, side in enumerate(chosen):
    g, starts = cands[side]
    s = int(rng.choice(starts))
    st = g.iloc[s:s + STRETCH_LEN].copy()
    st.insert(0, "stretch_id", k + 1)
    st.insert(1, "position", np.arange(1, STRETCH_LEN + 1))
    rows.append(st)
pool = pd.concat(rows, ignore_index=True)
cols = ["stretch_id", "position", "unit_id", "conversation_id", "speaker_side",
        "conversation_side", "original_sentence_number", "sentence_text"] + FEATURES + LEVELS
pool = pool[cols]
assert not set(pool["conversation_id"]) & set(excluded_convs)
assert not set(pool["unit_id"]) & set(exp2["unit_id"])
here.mkdir(exist_ok=True)
pool.to_csv(here / "heldout_pool.csv", index=False)

nb05 = json.loads((src / "nb05_summary.json").read_text())
manifest = {
    "seed": SEED, "stretches": N_STRETCHES, "sentences_per_stretch": STRETCH_LEN,
    "total_sentences": len(pool),
    "candidate_sides_after_exclusion": len(cands),
    "excluded_experiment2_conversations": excluded_convs,
    "excluded_experiment2_sides": excluded_sides,
    "level_thresholds_switchboard": nb05["thresholds"]["Switchboard"],
    "stretch_table": [
        {"stretch_id": int(sid), "conversation_side": g.conversation_side.iloc[0],
         "original_sentences": f"{int(g.original_sentence_number.min())}-{int(g.original_sentence_number.max())}",
         "tokens": int(g.tokens.sum()), "one_word_units": int(g.tokens.eq(1).sum())}
        for sid, g in pool.groupby("stretch_id")],
}
(here / "pool_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest["stretch_table"], indent=1), len(pool), len(cands))
