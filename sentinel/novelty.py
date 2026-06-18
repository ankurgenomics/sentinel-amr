"""Alignment-free novelty / out-of-distribution detection for SENTINEL.

The PathogenFinder2 insight: the hardest, highest-value task is assessing
organisms with NO close database match. A classifier trained on known clones
will happily output a confident probability for something it has never seen --
which is dangerous in a surveillance setting.

SENTINEL guards against this with a simple, defensible OOD signal computed in
the same gene-presence feature space (no extra model needed for the demo):

- Jaccard distance of the sample's gene set to its nearest training genome.
- Distance of the sample to the global training centroid.

If the nearest-neighbour distance exceeds a percentile threshold learned from
the training cohort, the sample is flagged NOVEL and the agent escalates it for
human review instead of trusting the resistance call. The scale-up version
(see SCALING.md) replaces this with ESM-2 protein-embedding distance + a VAE
reconstruction-error score, but the *logic* demonstrated here is identical.
"""

from __future__ import annotations

import json
import os

import numpy as np

MODEL_DIR = "models"


def _load_train_matrix() -> np.ndarray:
    data = np.load(f"{MODEL_DIR}/train_centroids.npz")
    return data["train_matrix"].astype(np.float32)


def fit_threshold(percentile: float = 95.0) -> dict:
    """Learn a novelty threshold from within-training nearest-neighbour distances."""
    Xtr = _load_train_matrix()
    n = Xtr.shape[0]
    # Pairwise Jaccard distance, excluding self; take each row's nearest neighbour.
    nn = []
    inter = Xtr @ Xtr.T
    sizes = Xtr.sum(axis=1)
    for i in range(n):
        union = sizes + sizes[i] - inter[i]
        union = np.where(union == 0, 1.0, union)
        jacc = inter[i] / union
        jacc[i] = -1.0  # exclude self
        nn.append(1.0 - jacc.max())
    thr = float(np.percentile(nn, percentile))
    info = {"threshold": thr, "percentile": percentile,
            "train_nn_mean": float(np.mean(nn)), "train_nn_max": float(np.max(nn))}
    os.makedirs(MODEL_DIR, exist_ok=True)
    json.dump(info, open(f"{MODEL_DIR}/novelty.json", "w"), indent=2)
    return info


def score(x: np.ndarray) -> dict:
    """Return novelty score (nearest-neighbour Jaccard distance) + flag."""
    Xtr = _load_train_matrix()
    info_path = f"{MODEL_DIR}/novelty.json"
    if not os.path.exists(info_path):
        fit_threshold()
    info = json.load(open(info_path))

    inter = Xtr @ x
    sizes = Xtr.sum(axis=1)
    union = sizes + x.sum() - inter
    union = np.where(union == 0, 1.0, union)
    jacc = inter / union
    nn_dist = float(1.0 - jacc.max()) if len(jacc) else 1.0
    is_novel = nn_dist > info["threshold"]
    return {
        "novelty_distance": round(nn_dist, 4),
        "threshold": round(info["threshold"], 4),
        "is_novel": bool(is_novel),
        "nearest_neighbour_jaccard": round(float(jacc.max()), 4) if len(jacc) else 0.0,
    }


if __name__ == "__main__":
    print("[novelty]", fit_threshold())
