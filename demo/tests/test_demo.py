"""Checks: (1) the demo prompt equals the saved Experiment 2 prompts byte for byte,
(2) the rule labels equal the saved Experiment 2 rule labels, (3) the pool does not
overlap Experiment 2, (4) the app runs headless (AppTest) in rules-only mode.
Set MTP_EXP2_DIR to the folder holding the NB06 outputs to run checks 1-2."""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from method import rule_based_load, temporal_prompt  # noqa: E402

EXP2 = Path(os.environ.get("MTP_EXP2_DIR", "/nonexistent"))
need_exp2 = pytest.mark.skipif(not EXP2.is_dir(), reason="MTP_EXP2_DIR not set")


@need_exp2
def test_prompt_and_rules_match_experiment2():
    seqs = pd.read_csv(EXP2 / "speech_to_narrative_temporal_reference_50_corrected.csv")
    prompts = pd.read_csv(EXP2 / "speech_to_narrative_temporal_prompts_50_corrected.csv")
    for sid, g in seqs.groupby("sequence_id"):
        assert temporal_prompt(g) == prompts.loc[prompts.sequence_id.eq(sid), "temporal_prompt"].iloc[0]
    assert seqs.apply(rule_based_load, axis=1).eq(seqs.rule_based_load_level).all()


@need_exp2
def test_pool_disjoint_from_experiment2():
    ev = pd.read_csv(EXP2 / "speech_to_narrative_temporal_evaluation_50_corrected.csv")
    pool = pd.read_csv(HERE / "demo_data" / "heldout_pool.csv")
    assert not set(pool.conversation_id) & set(ev.conversation_id)
    assert not set(pool.unit_id) & set(ev.unit_id)


def test_pool_shape():
    pool = pd.read_csv(HERE / "demo_data" / "heldout_pool.csv")
    assert pool.stretch_id.nunique() == 6 and len(pool) == 144
    for _, g in pool.groupby("stretch_id"):
        assert g.conversation_side.nunique() == 1
        assert g.original_sentence_number.diff().dropna().eq(1).all()


def test_app_rules_only():
    from streamlit.testing.v1 import AppTest
    os.environ.pop("GROQ_API_KEY", None)
    at = AppTest.from_file(str(HERE / "app.py"), default_timeout=60).run()
    assert not at.exception
    at.slider[0].set_value(12).run()
    at.button[0].click().run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "rule labels only" in text


@need_exp2
def test_parser_reads_saved_experiment2_answers():
    """parse_response() gives the same 396 labels as NB06 c51 on the real saved answers."""
    from method import parse_response
    resp = pd.read_csv(EXP2 / "speech_to_narrative_temporal_trajectories_50_corrected.csv")
    ev = pd.read_csv(EXP2 / "speech_to_narrative_temporal_evaluation_50_corrected.csv")
    for r in resp.itertuples():
        got = parse_response(r.llm_trajectory_response, int(r.number_of_sentences))
        want = ev[ev.sequence_id.eq(r.sequence_id)].sort_values("sentence_number")
        assert got.llm_load_level.tolist() == want.llm_load_level.tolist()
        assert got.llm_explanation.str.len().gt(0).all()
