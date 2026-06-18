"""SENTINEL end-to-end demo.

Runs the full triage agent on real held-out genomes and prints + writes
auditable risk reports. Works fully offline using the committed dataset; if
--genome-id is given and the network is available, it triages a live BV-BRC
genome instead.

Examples:
    # Offline: triage 3 held-out genomes from the committed dataset
    python -m sentinel.demo

    # Also fabricate a "novel" organism to show the OOD escalation path
    python -m sentinel.demo --show-novel

    # Live: pull one genome's gene calls from BV-BRC and triage it
    python -m sentinel.demo --genome-id 573.18697
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .agent import triage
from .features import load_vocab


def _tokens_from_row(row: pd.Series, feature_names: list[str]) -> list[str]:
    return [f for f in feature_names if row.get(f, 0) == 1]


def _print_summary(state: dict) -> None:
    print(f"\n=== {state['sample_id']} ===")
    if state.get("_truth"):
        correct = "OK" if state["_truth"] == state["decision"] else "MISS"
        print(f"  ground truth  : {state['_truth']}  [{correct}]")
    print(f"  decision      : {state['decision']}  (p={state['resistance_probability']:.3f}, "
          f"confidence={state['confidence']})")
    print(f"  novelty       : dist={state['novelty']['novelty_distance']} "
          f"novel={state['novelty']['is_novel']}")
    print("  top drivers   :")
    for d in state["explanation"]["top_drivers"][:5]:
        print(f"     {d['shap']:+.3f}  {d['property'][:12]:12s}  {d['gene'][:46]}")
    print("  flags         :")
    for f in state["flags"]:
        print(f"     - {f}")
    print(f"  report        : {state['report_paths']['markdown']}")


def run_offline(show_novel: bool) -> None:
    df = pd.read_csv("data/sentinel_dataset.csv")
    feature_names = load_vocab("models/feature_names.json")

    # Use the SAME group-aware held-out test set the model was evaluated on,
    # so demo cases are genuine generalisation examples (not training data).
    from .features import lineage_groups
    from .model import _split
    X = df[feature_names].to_numpy(dtype="float32")
    y = df["label"].to_numpy()
    groups = lineage_groups(df["genome_id"].astype(str).tolist())
    _, test_idx = _split(X, y, groups)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Show two genuine Resistant and two genuine Susceptible held-out genomes.
    sample_rows = pd.concat([
        test_df[test_df.label == 1].head(2),
        test_df[test_df.label == 0].head(2),
    ])
    for _, row in sample_rows.iterrows():
        tokens = _tokens_from_row(row, feature_names)
        truth = "Resistant" if row["label"] == 1 else "Susceptible"
        state = triage(tokens, sample_id=str(row["genome_id"]))
        state["_truth"] = truth
        _print_summary(state)

    if show_novel:
        # Fabricate an out-of-distribution sample: almost no known genes.
        novel_tokens = ["Virulence Factor::__synthetic_unknown_toxin__"]
        state = triage(novel_tokens, sample_id="SYNTHETIC_NOVEL_001",
                       organism="Unknown environmental isolate")
        _print_summary(state)


def run_live(genome_id: str) -> None:
    from .fetch_data import fetch_one_genome_tokens

    print(f"[demo] fetching live gene calls for {genome_id} from BV-BRC ...")
    tokens = sorted(fetch_one_genome_tokens(genome_id))
    print(f"[demo] recovered {len(tokens)} gene tokens")
    state = triage(tokens, sample_id=genome_id)
    _print_summary(state)


def main() -> None:
    ap = argparse.ArgumentParser(description="SENTINEL triage demo")
    ap.add_argument("--genome-id", default=None, help="Live BV-BRC genome id to triage")
    ap.add_argument("--show-novel", action="store_true",
                    help="Also triage a synthetic novel organism to show escalation")
    args = ap.parse_args()
    if args.genome_id:
        run_live(args.genome_id)
    else:
        run_offline(args.show_novel)


if __name__ == "__main__":
    main()
