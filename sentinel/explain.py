"""SHAP explainability for the SENTINEL AMR model.

Why this matters for SFA: a government surveillance tool must be auditable.
A clinician or risk officer needs to see *which genes* drove a "Resistant"
call, not just a probability. We use SHAP TreeExplainer, which is exact and
fast for tree ensembles (seconds, not hours -- it does NOT suffer the
kernel-SHAP blow-up that makes high-dimensional SHAP slow).

Two views:
- global_importance(): mean |SHAP| per gene across the cohort -> what the model
  relies on overall (sanity check: should surface known AMR genes).
- explain_sample(): top positive/negative gene contributions for one genome ->
  the per-sample audit trail shown in the risk report.
"""

from __future__ import annotations

import json
import os

import numpy as np
import xgboost as xgb

from .features import feature_gene, feature_property, load_dataset, load_vocab

MODEL_DIR = "models"
METRIC_DIR = "metrics"


def _load_booster() -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(f"{MODEL_DIR}/xgb_model.json")
    return booster


def _shap_values(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    """Exact tree SHAP via the booster's pred_contribs. Returns (n, n_feat)
    array of contributions (the final column, the bias, is dropped)."""
    dm = xgb.DMatrix(X)
    contribs = booster.predict(dm, pred_contribs=True)
    return contribs[:, :-1]  # drop bias term


def global_importance(csv_path: str = "data/sentinel_dataset.csv", top_n: int = 20):
    """Compute + plot global mean|SHAP| importance. Returns list of dicts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(METRIC_DIR, exist_ok=True)
    X, _y, feature_names, _ = load_dataset(csv_path)
    booster = _load_booster()
    sv = _shap_values(booster, X)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]

    items = [{
        "gene": feature_gene(feature_names[i]),
        "property": feature_property(feature_names[i]),
        "mean_abs_shap": float(mean_abs[i]),
    } for i in order]
    json.dump(items, open(f"{METRIC_DIR}/global_importance.json", "w"), indent=2)

    labels = [f"{it['gene'][:38]} ({it['property'][:4]})" for it in items][::-1]
    vals = [it["mean_abs_shap"] for it in items][::-1]
    plt.figure(figsize=(7, 6))
    plt.barh(labels, vals)
    plt.xlabel("mean |SHAP|"); plt.title("Top AMR drivers (global)")
    plt.tight_layout(); plt.savefig(f"{METRIC_DIR}/shap_global.png", dpi=130); plt.close()
    return items


def explain_sample(x: np.ndarray, feature_names: list[str], top_n: int = 8) -> dict:
    """Top gene contributions for a single sample feature vector x."""
    booster = _load_booster()
    sv = _shap_values(booster, x.reshape(1, -1))[0]
    present = np.where(x > 0)[0]
    # rank present genes by absolute contribution
    ranked = sorted(present, key=lambda i: abs(sv[i]), reverse=True)[:top_n]
    drivers = [{
        "gene": feature_gene(feature_names[i]),
        "property": feature_property(feature_names[i]),
        "shap": round(float(sv[i]), 4),
        "direction": "toward Resistant" if sv[i] > 0 else "toward Susceptible",
    } for i in ranked]
    return {"top_drivers": drivers}


if __name__ == "__main__":
    items = global_importance()
    print("[explain] top global drivers:")
    for it in items[:10]:
        print(f"  {it['mean_abs_shap']:.3f}  {it['property'][:20]:20s}  {it['gene'][:50]}")
