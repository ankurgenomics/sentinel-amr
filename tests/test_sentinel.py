"""Pytest suite for SENTINEL.

These tests run fully offline against the committed dataset and the trained
model artifacts. They verify the contract of each module and a full agent
pass, including the novelty-escalation safety path.

Run:  pytest -q
"""

from __future__ import annotations

import os

import numpy as np
import pytest

DATA = "data/sentinel_dataset.csv"
MODEL = "models/xgb_model.json"

needs_data = pytest.mark.skipif(not os.path.exists(DATA), reason="dataset not present")
needs_model = pytest.mark.skipif(not os.path.exists(MODEL), reason="model not trained")


@needs_data
def test_dataset_loads_balanced():
    from sentinel.features import load_dataset

    X, y, names, ids = load_dataset(DATA)
    assert X.shape[0] == len(y) == len(ids)
    assert X.shape[1] == len(names)
    assert set(np.unique(y)).issubset({0, 1})
    # presence/absence features only
    assert set(np.unique(X)).issubset({0.0, 1.0})


@needs_data
def test_feature_token_helpers():
    from sentinel.features import feature_gene, feature_property

    tok = "Antibiotic Resistance::CTX-M-15"
    assert feature_property(tok) == "Antibiotic Resistance"
    assert feature_gene(tok) == "CTX-M-15"


@needs_data
def test_group_split_is_disjoint():
    from sentinel.features import lineage_groups, load_dataset
    from sentinel.model import _split

    X, y, names, ids = load_dataset(DATA)
    groups = lineage_groups(ids)
    tr, te = _split(X, y, groups)
    assert len(set(groups[tr]) & set(groups[te])) == 0  # no group leakage


@needs_data
def test_vectorize_roundtrip():
    from sentinel.features import load_vocab, vectorize

    names = load_vocab("models/feature_names.json")
    present = {names[0], names[5]}
    vec = vectorize(present, names)
    assert vec.sum() == 2
    assert vec[0] == 1.0 and vec[5] == 1.0


@needs_model
def test_novelty_flags_unknown():
    import sentinel.novelty as nv

    names = _load_names()
    # an empty / unknown sample must be flagged novel
    x = np.zeros(len(names), dtype=np.float32)
    out = nv.score(x)
    assert out["is_novel"] is True
    assert 0.0 <= out["novelty_distance"] <= 1.0


@needs_model
def test_agent_end_to_end_known_sample():
    import pandas as pd

    from sentinel.agent import triage
    from sentinel.features import load_vocab

    names = load_vocab("models/feature_names.json")
    df = pd.read_csv(DATA)
    row = df.iloc[0]
    tokens = [f for f in names if row.get(f, 0) == 1]
    state = triage(tokens, sample_id="test_known")
    assert state["decision"] in {"Resistant", "Susceptible"}
    assert 0.0 <= state["resistance_probability"] <= 1.0
    assert state["confidence"] in {"high", "medium", "low"}
    assert "top_drivers" in state["explanation"]
    assert os.path.exists(state["report_paths"]["markdown"])
    assert os.path.exists(state["report_paths"]["json"])


@needs_model
def test_agent_escalates_novel_sample():
    from sentinel.agent import triage

    state = triage(["Virulence Factor::__totally_unknown__"], sample_id="test_novel")
    assert state["novelty"]["is_novel"] is True
    assert any("NOVEL" in f for f in state["flags"])


@needs_model
def test_critic_is_single_pass_deterministic():
    # Same input -> identical decision (bounded, no stochastic cycle).
    import pandas as pd

    from sentinel.agent import triage
    from sentinel.features import load_vocab

    names = load_vocab("models/feature_names.json")
    df = pd.read_csv(DATA)
    tokens = [f for f in names if df.iloc[3].get(f, 0) == 1]
    a = triage(tokens, sample_id="det_a")["resistance_probability"]
    b = triage(tokens, sample_id="det_b")["resistance_probability"]
    assert a == b


def _load_names():
    from sentinel.features import load_vocab

    return load_vocab("models/feature_names.json")
