"""Thesis method used by the demo, copied from the replication pipeline so the
demo runs on its own (Streamlit Cloud installs only the root requirements.txt).

Source: mtp/s06_llm_temporal.py of the replication package
  rule_based_load  = NB06 c31 / c51 (unchanged)
  temporal_prompt  = NB06 c41 (prompt text unchanged, byte for byte)
  LABEL_RE         = NB06 c51 label pattern (unchanged)
  metrics          = NB06 c53 / c55 (same scikit-learn calls)
The model settings are those of Experiment 2 (temperature 0.2).
tests/test_demo.py checks that temporal_prompt() here gives exactly the saved
Experiment 2 prompts.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

LABELS = ["low", "medium", "high"]
LABEL_RE = r"Sentence\s+(\d+)\s*:\s*Cognitive\s+load\s*:\s*(low|medium|high)\b"
PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0.2


def rule_based_load(row) -> str:
    """NB06 c31 / c51."""
    d, p, s = row["disfl_rate_level"], row["pause_ratio_level"], row["speech_rate_level"]
    if d == "high" and (p == "high" or s == "low"):
        return "high"
    if d == "low" and p == "low" and s == "high":
        return "low"
    return "medium"


def temporal_prompt(seq: pd.DataFrame) -> str:
    """NB06 c41 (text kept byte-for-byte)."""
    seq = seq.sort_values("sentence_number")
    parts = ["Analyse the temporal fluency and cognitive-processing patterns "
             "in the following ordered speech sequence.", "",
             "For each sentence, use only the provided features.",
             "Assign one cognitive-load level: low, medium, or high.",
             "Explain the relationship between speech rate, pausing, and disfluencies.",
             "Do not make diagnostic claims.", "Use cautious language."]
    for _, row in seq.iterrows():
        parts.extend(["", f"Sentence {int(row['sentence_number'])}:"])
        if pd.notna(row["sentence_text"]):
            parts.append(f"Transcript: {row['sentence_text']}")
        parts.extend([
            f"Duration: {row['duration_sec']:.3f} seconds",
            f"Token count: {row['tokens']}",
            f"Speech rate: {row['speech_rate']:.3f} words per second",
            f"Pause ratio: {row['pause_ratio']:.3f}",
            f"Number of pauses: {row['n_pauses']}",
            f"Disfluency rate: {row['disfl_rate']:.3f}",
            f"Cluster count: {row['cluster_count']}",
            f"Speech-rate level: {row['speech_rate_level']}",
            f"Pause-ratio level: {row['pause_ratio_level']}",
            f"Disfluency-rate level: {row['disfl_rate_level']}",
            f"Cluster level: {row['cluster_level']}",
            f"Dominant disfluency: {row['dominant_disfluency']}"])
    parts.extend(["", "Return the result in this exact format:", "",
                  "Sentence 1:", "Cognitive load: low, medium, or high",
                  "Explanation: one or two sentences", "",
                  "Sentence 2:", "Cognitive load: low, medium, or high",
                  "Explanation: one or two sentences", "",
                  "Continue this format for every sentence in the sequence.", "",
                  "Overall trajectory:",
                  "Describe how cognitive load changes across the sequence.",
                  "Explain which features support the changes.",
                  "State that the features do not directly measure cognitive load."])
    return "\n".join(parts)


def clean_response(text: str) -> str:
    """Same normalisation as NB06 c51 before the label regex."""
    return re.sub(r"[*_`#]", "", unicodedata.normalize("NFKC", str(text)))


def parse_response(text: str, n: int) -> pd.DataFrame:
    """Labels via the NB06 c51 regex; explanation = text after 'Explanation:' up to
    the next sentence block (used only to show reasons for mismatches)."""
    t = clean_response(text)
    labels = {int(k): v.lower() for k, v in re.findall(LABEL_RE, t, flags=re.IGNORECASE)}
    expl = {}
    blocks = re.split(r"(?=Sentence\s+\d+\s*:\s*Cognitive\s+load)", t, flags=re.IGNORECASE)
    for b in blocks:
        m = re.match(r"Sentence\s+(\d+)", b, flags=re.IGNORECASE)
        if not m:
            continue
        e = re.search(r"Explanation\s*:\s*(.+?)(?:\n\s*\n|Overall trajectory|$)", b,
                      flags=re.IGNORECASE | re.DOTALL)
        expl[int(m.group(1))] = " ".join(e.group(1).split()) if e else ""
    return pd.DataFrame({"sentence_number": range(1, n + 1),
                         "llm_load_level": [labels.get(i) for i in range(1, n + 1)],
                         "llm_explanation": [expl.get(i, "") for i in range(1, n + 1)]})


def overall_trajectory(text: str) -> str:
    t = clean_response(text)
    m = re.search(r"Overall trajectory\s*:?\s*(.*)$", t, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def metrics(y: pd.Series, p: pd.Series) -> dict:
    """NB06 c53 / c55 on the sentences where the model returned a label."""
    prec, rec, f1, sup = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    match = int((y.values == p.values).sum())
    return {"sentences": len(y), "matching_labels": match,
            "agreement_percent": 100 * match / len(y) if len(y) else float("nan"),
            "macro_f1": f1_score(y, p, labels=LABELS, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, p, labels=LABELS, average="weighted", zero_division=0),
            "cohen_kappa": cohen_kappa_score(y, p, labels=LABELS) if len(set(y) | set(p)) > 1 else float("nan"),
            "confusion": confusion_matrix(y, p, labels=LABELS),
            "per_class": pd.DataFrame({"precision": prec, "recall": rec, "f1": f1,
                                       "support": sup}, index=LABELS)}
