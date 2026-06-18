"""Feature handling for SENTINEL.

Features are presence/absence of curated specialty genes (AMR genes from CARD,
virulence factors from VFDB, drug targets, metal-resistance genes) per genome.
This is an interpretable, alignment-free representation: each feature is a named
gene, so model explanations point to real biology rather than opaque k-mers.

We deliberately chose this over raw k-mer spectra because:
- It is tiny and fast (no FASTQ, no assembly, no out-of-RAM k-mer counting).
- SHAP values name actual resistance/virulence genes -- audit-friendly.
- It mirrors how a surveillance analyst reasons ("which AMR genes are present?").

`load_dataset` reads the committed CSV. `vectorize` maps an arbitrary set of
gene tokens onto a trained feature vocabulary so the agent can score new samples.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

META_COLS = ("genome_id", "label")


def load_dataset(csv_path: str = "data/sentinel_dataset.csv"):
    """Return (X, y, feature_names, genome_ids) from the committed dataset."""
    df = pd.read_csv(csv_path)
    feature_names = [c for c in df.columns if c not in META_COLS]
    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    genome_ids = df["genome_id"].astype(str).tolist()
    return X, y, feature_names, genome_ids


def lineage_groups(genome_ids: list[str]) -> np.ndarray:
    """Derive a grouping key for group-aware splitting.

    BV-BRC genome_ids look like '573.18697' (taxon.assembly). Without MLST we
    approximate "related isolates" by the integer part of the assembly id // 1000,
    which clusters genomes deposited together (often the same study/clone). This
    is a conservative proxy so the test set isn't trivially similar to training.
    """
    groups = []
    for gid in genome_ids:
        try:
            assembly = int(gid.split(".")[1])
            groups.append(assembly // 1000)
        except (IndexError, ValueError):
            groups.append(hash(gid) % 1000)
    return np.asarray(groups)


def feature_property(feature_name: str) -> str:
    """'Antibiotic Resistance::sul1' -> 'Antibiotic Resistance'."""
    return feature_name.split("::", 1)[0] if "::" in feature_name else "Unknown"


def feature_gene(feature_name: str) -> str:
    """'Antibiotic Resistance::sul1' -> 'sul1'."""
    return feature_name.split("::", 1)[1] if "::" in feature_name else feature_name


def vectorize(present_tokens: set[str], feature_names: list[str]) -> np.ndarray:
    """Map a set of 'PROPERTY::gene' tokens onto the trained vocabulary."""
    idx = {name: i for i, name in enumerate(feature_names)}
    vec = np.zeros(len(feature_names), dtype=np.float32)
    for tok in present_tokens:
        if tok in idx:
            vec[idx[tok]] = 1.0
    return vec


def save_vocab(feature_names: list[str], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(feature_names, open(path, "w"))


def load_vocab(path: str) -> list[str]:
    return json.load(open(path))
