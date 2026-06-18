"""SENTINEL triage agent -- a bounded LangGraph StateGraph.

Flow (no cycles -> no infinite-loop risk):

  ingest -> features -> classify -> novelty -> explain -> critic -> report

Each node is a pure function over a shared TypedDict state. The critic node runs
ONCE: it inspects probability, confidence, and the novelty flag, then decides
whether to escalate to human review. It never rewrites the report in a loop --
a deliberate design choice for a deterministic, auditable surveillance tool.

This mirrors the author's prior agentic-genomics / outbreak-agent pattern
(graph of specialised nodes + a critic/validation gate), applied to AMR triage.
"""

from __future__ import annotations

import json
import os
from typing import TypedDict

import numpy as np
import xgboost as xgb

from . import explain as explain_mod
from . import novelty as novelty_mod
from . import report as report_mod
from .features import load_vocab, vectorize

MODEL_DIR = "models"


class TriageState(TypedDict, total=False):
    sample_id: str
    organism: str
    antibiotic: str
    gene_tokens: list[str]      # input: 'PROPERTY::gene' tokens for the sample
    x: np.ndarray               # vectorised features
    feature_names: list[str]
    resistance_probability: float
    decision: str
    confidence: str
    novelty: dict
    explanation: dict
    flags: list[str]
    report_paths: dict


def _node_ingest(state: TriageState) -> TriageState:
    state.setdefault("sample_id", "sample")
    state.setdefault("organism", "K. pneumoniae (assumed)")
    state.setdefault("antibiotic", "ciprofloxacin")
    state.setdefault("flags", [])
    if not state.get("gene_tokens"):
        state["flags"].append("No genes recovered from sample -- low information input")
    return state


def _node_features(state: TriageState) -> TriageState:
    feature_names = load_vocab(f"{MODEL_DIR}/feature_names.json")
    state["feature_names"] = feature_names
    state["x"] = vectorize(set(state.get("gene_tokens", [])), feature_names)
    return state


def _node_classify(state: TriageState) -> TriageState:
    booster = xgb.Booster()
    booster.load_model(f"{MODEL_DIR}/xgb_model.json")
    proba = float(booster.predict(xgb.DMatrix(state["x"].reshape(1, -1)))[0])
    state["resistance_probability"] = round(proba, 4)
    state["decision"] = "Resistant" if proba >= 0.5 else "Susceptible"
    # confidence band from distance to the 0.5 boundary
    margin = abs(proba - 0.5)
    state["confidence"] = "high" if margin > 0.35 else "medium" if margin > 0.15 else "low"
    return state


def _node_novelty(state: TriageState) -> TriageState:
    state["novelty"] = novelty_mod.score(state["x"])
    return state


def _node_explain(state: TriageState) -> TriageState:
    state["explanation"] = explain_mod.explain_sample(state["x"], state["feature_names"])
    return state


def _node_critic(state: TriageState) -> TriageState:
    """Single-pass validation gate -> sets escalation flags."""
    flags = state.get("flags", [])
    if state["novelty"]["is_novel"]:
        flags.append("NOVEL organism (no close database match) -- escalate; "
                     "do not trust automated resistance call")
    if state["confidence"] == "low":
        flags.append("Low-confidence prediction (probability near decision boundary) "
                     "-- recommend phenotypic AST confirmation")
    if state["decision"] == "Resistant" and state["confidence"] in ("high", "medium"):
        flags.append("Predicted RESISTANT -- flag for stewardship / outbreak review")
    state["flags"] = flags
    return state


def _node_report(state: TriageState) -> TriageState:
    metrics_path = "metrics/metrics.json"
    roc = json.load(open(metrics_path))["roc_auc"] if os.path.exists(metrics_path) else None
    result = {
        "sample_id": state["sample_id"],
        "organism": state["organism"],
        "antibiotic": state["antibiotic"],
        "resistance_probability": state["resistance_probability"],
        "decision": state["decision"],
        "confidence": state["confidence"],
        "novelty": state["novelty"],
        "explanation": state["explanation"],
        "flags": state["flags"],
        "model": "XGBoost (gene presence/absence)",
        "model_roc_auc": roc,
    }
    md, js = report_mod.render(result)
    state["report_paths"] = {"markdown": md, "json": js}
    return state


def build_graph():
    """Compile the bounded triage StateGraph. Falls back to a manual chain if
    langgraph is unavailable (keeps the demo robust)."""
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception:  # noqa: BLE001
        return None

    g = StateGraph(TriageState)
    g.add_node("ingest", _node_ingest)
    g.add_node("features", _node_features)
    g.add_node("classify", _node_classify)
    g.add_node("novelty", _node_novelty)
    g.add_node("explain", _node_explain)
    g.add_node("critic", _node_critic)
    g.add_node("report", _node_report)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "features")
    g.add_edge("features", "classify")
    g.add_edge("classify", "novelty")
    g.add_edge("novelty", "explain")
    g.add_edge("explain", "critic")
    g.add_edge("critic", "report")
    g.add_edge("report", END)
    return g.compile()


_CHAIN = [_node_ingest, _node_features, _node_classify,
          _node_novelty, _node_explain, _node_critic, _node_report]


def triage(gene_tokens: list[str], sample_id: str = "sample",
           organism: str | None = None, antibiotic: str = "ciprofloxacin") -> TriageState:
    """Run the triage pipeline on a sample's gene tokens. Uses LangGraph if
    available, else an identical manual chain."""
    state: TriageState = {"gene_tokens": gene_tokens, "sample_id": sample_id,
                          "antibiotic": antibiotic}
    if organism:
        state["organism"] = organism
    graph = build_graph()
    if graph is not None:
        return graph.invoke(state)
    for node in _CHAIN:
        state = node(state)
    return state
