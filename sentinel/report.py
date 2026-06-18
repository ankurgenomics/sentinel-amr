"""Auditable risk-report rendering for SENTINEL.

Produces a human-readable markdown report (for a risk officer) and a machine
JSON record (for downstream LIMS / surveillance dashboards). Designed so a
non-bioinformatician can act on it -- the same motivation as rMAP's HTML report.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone


def render(result: dict, out_dir: str = "output") -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    sid = result.get("sample_id", "sample")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    decision = result["decision"]
    prob = result["resistance_probability"]
    novelty = result["novelty"]
    drivers = result["explanation"]["top_drivers"]
    flags = result.get("flags", [])

    lines = [
        f"# SENTINEL Risk Report -- {sid}",
        "",
        f"- Generated (UTC): {ts}",
        f"- Organism context: {result.get('organism', 'K. pneumoniae (assumed)')}",
        f"- Antibiotic: {result.get('antibiotic', 'ciprofloxacin')}",
        "",
        "## Decision",
        f"- **Call:** {decision}",
        f"- **Resistance probability:** {prob:.3f}",
        f"- **Confidence band:** {result['confidence']}",
        "",
        "## Novelty / out-of-distribution check",
        f"- Nearest-neighbour Jaccard distance: {novelty['novelty_distance']} "
        f"(threshold {novelty['threshold']})",
        f"- **Novel organism flag:** {'YES -- escalate for human review' if novelty['is_novel'] else 'no'}",
        "",
        "## Why (top gene drivers, SHAP)",
        "| Gene | Category | SHAP | Pushes |",
        "|------|----------|------|--------|",
    ]
    for d in drivers:
        lines.append(f"| {d['gene'][:48]} | {d['property']} | {d['shap']} | {d['direction']} |")

    lines += ["", "## Action flags"]
    if flags:
        lines += [f"- {f}" for f in flags]
    else:
        lines.append("- none")
    lines += [
        "",
        "## Audit trail",
        f"- Model: {result.get('model', 'XGBoost')} | "
        f"Holdout ROC-AUC: {result.get('model_roc_auc', 'see metrics/metrics.json')}",
        "- Data source: BV-BRC (PATRIC) genome_amr + sp_gene (CARD/VFDB)",
        "- This is a decision-support output. All escalations require analyst confirmation.",
        "",
    ]
    md = "\n".join(lines)
    md_path = os.path.join(out_dir, f"{sid}_report.md")
    json_path = os.path.join(out_dir, f"{sid}_report.json")
    open(md_path, "w").write(md)
    json.dump(result, open(json_path, "w"), indent=2, default=str)
    return md_path, json_path
