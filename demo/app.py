"""Streamlit demo for the MTP thesis
"LLM-based Temporal Reasoning for Cognitive Load and Disfluency Analysis in
Spontaneous Speech" (Tarun Saxena, M25AI2135, IIT Jodhpur).

It repeats Experiment 2 on held-out Switchboard sentences that were not among
the 396 sentences of the thesis: rule-based load labels, the same Groq prompt,
the comparison (agreement, confusion matrix, macro-F1), a trajectory plot and
the model's explanation for every sentence where the two labels differ.

Run locally:  streamlit run demo/app.py
The Groq key is read from st.secrets["GROQ_API_KEY"] or the GROQ_API_KEY
environment variable. It is never written to disk or shown.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from method import (FALLBACK_MODEL, LABELS, PRIMARY_MODEL, TEMPERATURE,  # noqa: E402
                    metrics, overall_trajectory, parse_response, rule_based_load,
                    temporal_prompt)

POOL = HERE / "demo_data" / "heldout_pool.csv"
MANIFEST = HERE / "demo_data" / "pool_manifest.json"
CACHE = HERE / "cache"
LEVEL_Y = {"low": 0, "medium": 1, "high": 2}

st.set_page_config(page_title="MTP demo: temporal cognitive-load labelling",
                   page_icon="🗣️", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def load_pool() -> tuple[pd.DataFrame, dict]:
    return pd.read_csv(POOL), json.loads(MANIFEST.read_text())


def build_sequence(pool: pd.DataFrame, stretch_id: int, n: int) -> pd.DataFrame:
    seq = pool[pool.stretch_id.eq(stretch_id)].sort_values("position").head(n).copy()
    seq["sentence_number"] = range(1, len(seq) + 1)
    seq["rule_based_load_level"] = seq.apply(rule_based_load, axis=1)
    return seq.reset_index(drop=True)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ LLM
def read_key() -> str | None:
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:  # noqa: BLE001 - no secrets.toml
        key = None
    return key or os.environ.get("GROQ_API_KEY") or None


def call_groq(prompt: str, model: str, key: str) -> str:
    from groq import Groq
    client = Groq(api_key=key).with_options(max_retries=1, timeout=90.0)
    text = client.chat.completions.create(
        model=model, temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    if not text or not text.strip():
        raise ValueError("empty response")
    return text


def cache_path(stretch_id: int, n: int) -> Path:
    return CACHE / f"stretch{stretch_id}_n{n:02d}.json"


def read_cache(stretch_id: int, n: int, prompt: str) -> dict | None:
    p = cache_path(stretch_id, n)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return data if data.get("prompt_sha256") == sha(prompt) else None


def get_answer(stretch_id: int, n: int, prompt: str, use_live: bool) -> dict:
    """120b -> 20b -> cached answer -> rule labels only."""
    log = []
    key = read_key() if use_live else None
    if use_live and not key:
        log.append("No GROQ_API_KEY found (st.secrets or environment): live call skipped.")
    if key:
        for model in (PRIMARY_MODEL, FALLBACK_MODEL):
            t0 = time.time()
            try:
                text = call_groq(prompt, model, key)
                log.append(f"{model}: success in {time.time() - t0:.1f} s.")
                return {"source": "live", "model": model, "response": text, "log": log}
            except Exception as err:  # noqa: BLE001
                log.append(f"{model}: failed ({type(err).__name__}).")
    cached = read_cache(stretch_id, n, prompt)
    if cached:
        log.append(f"Using the cached answer ({cached['model']}, saved {cached['created_utc']}).")
        return {"source": "cache", "model": cached["model"], "response": cached["response"],
                "log": log}
    log.append("No cached answer for this stretch and length: showing rule labels only.")
    return {"source": "rules_only", "model": None, "response": None, "log": log}


# ------------------------------------------------------------------ plots
def trajectory_plot(df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.1, 1]})
    x = df["sentence_number"]
    ax1.plot(x, df["rule_based_load_level"].map(LEVEL_Y), "o-", color="#1f4e79",
             label="Rule-based label", lw=2)
    if "llm_load_level" in df and df["llm_load_level"].notna().any():
        ax1.plot(x, df["llm_load_level"].map(LEVEL_Y), "s--", color="#c55a11",
                 label="LLM label", lw=2, alpha=.85)
        mism = df[df["llm_load_level"].notna() & df["llm_load_level"].ne(df["rule_based_load_level"])]
        ax1.scatter(mism["sentence_number"], [2.35] * len(mism), marker="v", color="crimson",
                    label="Mismatch", zorder=5)
    ax1.set_yticks([0, 1, 2], LABELS)
    ax1.set_ylim(-0.4, 2.6)
    ax1.set_ylabel("Cognitive load")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, frameon=False)
    ax1.grid(alpha=.3)
    ax2.plot(x, df["speech_rate"], "o-", label="Speech rate (words/s)", color="#2e7d32")
    ax2b = ax2.twinx()
    ax2b.plot(x, df["pause_ratio"], "^-", label="Pause ratio", color="#6a1b9a")
    ax2b.plot(x, df["disfl_rate"], "d-", label="Disfluency rate", color="#ef6c00")
    ax2.set_ylabel("Speech rate")
    ax2b.set_ylabel("Ratio / rate")
    ax2.set_xlabel("Sentence in the stretch")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, frameon=False)
    ax2.grid(alpha=.3)
    ax2.set_xticks(list(x))
    fig.tight_layout()
    return fig


def rule_reason(row) -> str:
    """Plain description of which branch of rule_based_load() applied (display only)."""
    d, p, s = row["disfl_rate_level"], row["pause_ratio_level"], row["speech_rate_level"]
    if d == "high" and (p == "high" or s == "low"):
        return f"high: disfluency high and ({'pause high' if p == 'high' else 'speech rate low'})"
    if d == "low" and p == "low" and s == "high":
        return "low: disfluency low, pause low and speech rate high"
    return f"medium: no extreme pattern (speech {s}, pause {p}, disfluency {d})"


# ------------------------------------------------------------------ UI
pool, manifest = load_pool()
st.title("Temporal cognitive-load labelling on held-out Switchboard speech")
st.caption("Demo for the M.Tech thesis of Tarun Saxena (M25AI2135), IIT Jodhpur. "
           "Supervisor: Prof. Sidharth Ranjan.")
st.markdown(
    "This page repeats **Experiment 2** of the thesis on sentences that were **not** used in it. "
    f"The pool has {manifest['stretches']} stretches of {manifest['sentences_per_stretch']} "
    "consecutive sentences, each from a different Switchboard conversation side; the four "
    "conversations of Experiment 2 (" + ", ".join(manifest["excluded_experiment2_conversations"])
    + ") were left out completely. The rule labels, the prompt and the comparison are the same "
    "as in the thesis. The features only *suggest* load; they do not measure it directly.")

with st.sidebar:
    st.header("Settings")
    table = pd.DataFrame(manifest["stretch_table"])
    opts = {int(r.stretch_id): f"Stretch {r.stretch_id}: {r.conversation_side} "
            f"(sentences {r.original_sentences})" for r in table.itertuples()}
    stretch_id = st.selectbox("Stretch", list(opts), format_func=opts.get)
    n = st.slider("Number of sentences", 4, 24, 8)
    have_key = read_key() is not None
    use_live = st.toggle("Call the Groq model", value=have_key,
                         help="Uses openai/gpt-oss-120b, then openai/gpt-oss-20b. Needs GROQ_API_KEY.")
    st.caption("API key: " + ("found" if have_key else "not set (cache or rules only)"))
    run = st.button("Run analysis", type="primary", use_container_width=True)
    st.divider()
    st.caption(f"Fallback order: {PRIMARY_MODEL} → {FALLBACK_MODEL} → cached answer → rule labels only. "
               f"Temperature {TEMPERATURE}, as in Experiment 2.")

seq = build_sequence(pool, stretch_id, n)
prompt = temporal_prompt(seq)
state_key = f"{stretch_id}-{n}-{sha(prompt)[:12]}-{use_live}"
if run:
    with st.spinner("Asking the model ..."):
        st.session_state[state_key] = get_answer(stretch_id, n, prompt, use_live)
answer = st.session_state.get(state_key)

st.subheader(f"1. Rule-based labels ({n} sentences, {seq.conversation_side.iloc[0]})")
st.caption("Rule (NB06): high if disfluency is high and (pause ratio is high or speech rate is low); "
           "low if disfluency is low, pause ratio is low and speech rate is high; otherwise medium. "
           "Levels use the Switchboard 25th/75th percentile cuts of the full corpus.")
show = seq[["sentence_number", "original_sentence_number", "sentence_text", "tokens",
            "speech_rate", "pause_ratio", "disfl_rate", "speech_rate_level",
            "pause_ratio_level", "disfl_rate_level", "rule_based_load_level"]]
st.dataframe(show.rename(columns={"sentence_number": "#", "original_sentence_number": "orig #",
                                  "rule_based_load_level": "rule label"}),
             hide_index=True, use_container_width=True,
             column_config={"speech_rate": st.column_config.NumberColumn(format="%.3f"),
                            "pause_ratio": st.column_config.NumberColumn(format="%.3f"),
                            "disfl_rate": st.column_config.NumberColumn(format="%.3f")})

st.subheader("2. LLM call")
with st.expander("Prompt sent to the model (same template as Experiment 2)"):
    st.code(prompt, language="text")
if answer is None:
    st.info("Choose a stretch and a number of sentences, then press **Run analysis**.")
    st.pyplot(trajectory_plot(seq))
    st.stop()

for line in answer["log"]:
    st.write("• " + line)
src_label = {"live": "live call", "cache": "cached answer", "rules_only": "rule labels only"}
st.markdown(f"**Source:** {src_label[answer['source']]}"
            + (f" · **model:** `{answer['model']}`" if answer["model"] else ""))

if answer["response"] is None:
    st.warning("No model answer is available, so only the rule labels and the feature "
               "trajectory are shown. Add GROQ_API_KEY to run the model.")
    st.pyplot(trajectory_plot(seq))
    st.stop()

parsed = parse_response(answer["response"], n)
df = seq.merge(parsed, on="sentence_number", how="left")
with st.expander("Full model answer"):
    st.text(answer["response"])
missing = int(df.llm_load_level.isna().sum())
if missing:
    st.warning(f"The model gave no readable label for {missing} sentence(s); they are left out "
               "of the comparison.")

st.subheader("3. Comparison")
ok = df[df.llm_load_level.notna()]
if ok.empty:
    st.error("No labels could be read from the answer.")
    st.stop()
m = metrics(ok.rule_based_load_level, ok.llm_load_level)
base = ok.rule_based_load_level.value_counts().max() / len(ok) * 100
c1, c2, c3, c4 = st.columns(4)
c1.metric("Agreement", f"{m['agreement_percent']:.2f}%", f"{m['matching_labels']}/{m['sentences']}",
          delta_color="off")
c2.metric("Macro-F1", f"{m['macro_f1']:.4f}")
c3.metric("Cohen's kappa", "n/a" if pd.isna(m["cohen_kappa"]) else f"{m['cohen_kappa']:.4f}")
c4.metric("Majority-label baseline", f"{base:.2f}%")
st.caption("For reference, the thesis result on 396 sentences: agreement 214/396 = 54.04%, "
           "macro-F1 0.5423, kappa 0.3250 (always-medium baseline 66.92%). "
           "A short stretch is a very small sample, so its values vary a lot.")
cc1, cc2 = st.columns([1, 1.3])
with cc1:
    cm = pd.DataFrame(m["confusion"], index=[f"Rule: {l}" for l in LABELS],
                      columns=[f"LLM: {l}" for l in LABELS])
    st.markdown("**Confusion matrix** (rows = rule, columns = LLM)")
    st.dataframe(cm, use_container_width=True)
with cc2:
    st.markdown("**Per-class scores** (rule labels as reference)")
    st.dataframe(m["per_class"].style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                              "f1": "{:.3f}"}), use_container_width=True)

st.subheader("4. Trajectory")
st.pyplot(trajectory_plot(df))
traj = overall_trajectory(answer["response"])
if traj:
    st.markdown("**Model's overall trajectory:** " + traj)

st.subheader("5. Reasons for each mismatch")
mm = ok[ok.llm_load_level.ne(ok.rule_based_load_level)]
if mm.empty:
    st.success("The model and the rule agree on every sentence in this stretch.")
else:
    st.caption(f"{len(mm)} of {len(ok)} sentences differ. The rule column shows which branch of the "
               "rule applied; the explanation is the model's own text.")
    out = pd.DataFrame({
        "#": mm.sentence_number, "transcript": mm.sentence_text, "tokens": mm.tokens,
        "rule": mm.apply(rule_reason, axis=1), "LLM": mm.llm_load_level,
        "LLM explanation": mm.llm_explanation})
    st.dataframe(out, hide_index=True, use_container_width=True)

st.download_button("Download this comparison (CSV)",
                   df.to_csv(index=False).encode(),
                   file_name=f"demo_stretch{stretch_id}_n{n}.csv", mime="text/csv")
st.caption("Research demo only; no diagnostic claims are made.")
