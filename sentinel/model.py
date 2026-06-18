"""Train and evaluate the SENTINEL AMR classifier (XGBoost).

Honest evaluation principles (important for a government science panel):
- Group-aware train/test split: related isolates never straddle the split, so
  reported metrics reflect generalisation to new clones, not memorisation.
- Report ROC-AUC, PR-AUC (average precision), accuracy, and a confusion matrix
  on the held-out test set -- as-is, even if modest.
- Calibrate-aware: we also save predicted probabilities for inspection.

Artifacts written to models/ and metrics/:
- models/xgb_model.json        trained booster
- models/feature_names.json    feature vocabulary (for live scoring)
- models/train_centroids.npz   per-class feature centroids (used by novelty.py)
- metrics/metrics.json         numeric results
- metrics/*.png                ROC, PR, confusion matrix plots
"""

from __future__ import annotations

import json
import os

import numpy as np

from .features import load_dataset, lineage_groups, save_vocab

MODEL_DIR = "models"
METRIC_DIR = "metrics"


def _split(X, y, groups, test_frac=0.25, seed=42):
    """Group-aware split via sklearn GroupShuffleSplit."""
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return train_idx, test_idx


def train(csv_path: str = "data/sentinel_dataset.csv", seed: int = 42) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xgboost as xgb
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)

    X, y, feature_names, genome_ids = load_dataset(csv_path)
    groups = lineage_groups(genome_ids)
    train_idx, test_idx = _split(X, y, groups, seed=seed)
    Xtr, Xte, ytr, yte = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    print(f"[train] train={len(ytr)} test={len(yte)} features={len(feature_names)} "
          f"groups_train={len(set(groups[train_idx]))} groups_test={len(set(groups[test_idx]))}")

    clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_lambda=2.0,
        min_child_weight=2,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(yte, proba))
    pr_auc = float(average_precision_score(yte, proba))
    acc = float((pred == yte).mean())
    cm = confusion_matrix(yte, pred).tolist()

    # --- plots ---
    fpr, tpr, _ = roc_curve(yte, proba)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC -- ciprofloxacin resistance"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{METRIC_DIR}/roc.png", dpi=130); plt.close()

    prec, rec, _ = precision_recall_curve(yte, proba)
    plt.figure(figsize=(4, 4))
    plt.plot(rec, prec, label=f"AP={pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{METRIC_DIR}/pr.png", dpi=130); plt.close()

    plt.figure(figsize=(3.6, 3.2))
    cm_arr = np.array(cm)
    plt.imshow(cm_arr, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm_arr[i, j], ha="center", va="center")
    plt.xticks([0, 1], ["S", "R"]); plt.yticks([0, 1], ["S", "R"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix")
    plt.tight_layout(); plt.savefig(f"{METRIC_DIR}/confusion.png", dpi=130); plt.close()

    # --- persist model + vocab + class centroids (for novelty detection) ---
    clf.get_booster().save_model(f"{MODEL_DIR}/xgb_model.json")
    save_vocab(feature_names, f"{MODEL_DIR}/feature_names.json")
    cen_pos = Xtr[ytr == 1].mean(axis=0)
    cen_neg = Xtr[ytr == 0].mean(axis=0)
    train_mean = Xtr.mean(axis=0)
    np.savez(f"{MODEL_DIR}/train_centroids.npz",
             pos=cen_pos, neg=cen_neg, train_mean=train_mean,
             train_matrix=Xtr.astype(np.float32))

    metrics = {
        "task": "K. pneumoniae ciprofloxacin resistance (Resistant=1)",
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "n_features": int(len(feature_names)),
        "roc_auc": round(roc_auc, 4), "pr_auc": round(pr_auc, 4),
        "accuracy": round(acc, 4), "confusion_matrix": cm,
        "split": "group-aware (GroupShuffleSplit on assembly-cluster groups)",
        "model": "XGBoost (400 trees, depth 4, lr 0.05)",
        "data_source": "BV-BRC genome_amr + sp_gene (CARD/VFDB)",
    }
    json.dump(metrics, open(f"{METRIC_DIR}/metrics.json", "w"), indent=2)
    print(f"[train] ROC-AUC={roc_auc:.3f}  PR-AUC={pr_auc:.3f}  acc={acc:.3f}")
    print(f"[train] artifacts in {MODEL_DIR}/ and {METRIC_DIR}/")
    return metrics


if __name__ == "__main__":
    train()
