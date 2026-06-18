"""Fetch REAL, curated public data from BV-BRC (PATRIC) for AMR triage.

Two endpoints, both returning pre-curated data (no FASTQ, no assembly):

1. genome_amr  -> antibiotic susceptibility labels (Resistant/Susceptible),
   derived from laboratory AST and curated by PATRIC/BV-BRC.
2. sp_gene     -> per-genome "specialty genes": antibiotic-resistance genes
   (CARD), virulence factors (VFDB), drug targets, metal-resistance genes.

We turn (2) into a gene presence/absence feature matrix and (1) into labels.
Every gene name is real (e.g. sul1, MCR-1, blaKPC), so SHAP explanations name
actual biology -- exactly what a science review panel expects.

Usage:
    python -m sentinel.fetch_data --taxon 573 --antibiotic ciprofloxacin \
        --max-per-class 150 --out data/

Design choices that beat the "bioinformatics data is messy" time sink:
- We never download genome sequences; curated gene calls are tiny and fast.
- We cache every API page to data/raw/ so re-runs are instant and offline.
- A committed subset (data/sentinel_dataset.csv) lets the demo run with no net.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from typing import Iterable

import requests

BVBRC = "https://www.bv-brc.org/api"
HEADERS = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}

# Properties from sp_gene we treat as features. "Drug Target" and "Transporter"
# are informative context; AMR + Virulence are the biologically critical ones.
FEATURE_PROPERTIES = {
    "Antibiotic Resistance",
    "Virulence Factor",
    "Virulance factor",  # BV-BRC has this misspelling in some rows; keep both.
    "Metal Resistance",
    "Drug Target",
}


def _get(path: str, rql: str, max_retries: int = 4, timeout: int = 60) -> list[dict]:
    """GET a BV-BRC API page with retries. Returns parsed JSON list.

    `rql` is a raw BV-BRC RQL query string (e.g. "and(...)&select(...)&limit(...)").
    It MUST be appended raw -- letting requests URL-encode the parens/commas
    breaks the Solr backend (HTTP 400). We only encode unsafe chars manually.
    """
    # Encode spaces (e.g. in "Antibiotic Resistance") but keep RQL syntax chars.
    safe_rql = rql.replace(" ", "%20")
    url = f"{BVBRC}/{path}/?{safe_rql}"
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers={"Accept": "application/json"}, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:  # noqa: BLE001
            last_err = str(e)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"BV-BRC request failed after {max_retries} tries: {last_err}")


def fetch_labeled_ids(taxon_id: int, antibiotic: str, phenotype: str,
                      limit: int, cache_dir: str) -> list[str]:
    """Return genome_ids with a given AST phenotype for an antibiotic."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"ids_{taxon_id}_{antibiotic}_{phenotype}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    q = (f"and(eq(antibiotic,{antibiotic}),eq(taxon_id,{taxon_id}),"
         f"eq(resistant_phenotype,{phenotype}))&select(genome_id)&limit({limit})")
    rows = _get("genome_amr", q)
    ids = sorted({r["genome_id"] for r in rows})
    json.dump(ids, open(cache, "w"))
    return ids


def fetch_genes_for_ids(genome_ids: Iterable[str], cache_dir: str,
                        batch: int = 12) -> dict[str, set[str]]:
    """Map each genome_id -> set of 'PROPERTY::gene' feature tokens.

    Batched via in(genome_id,(...)). Genes without a name fall back to product.
    """
    os.makedirs(cache_dir, exist_ok=True)
    ids = list(genome_ids)
    result: dict[str, set[str]] = {gid: set() for gid in ids}
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        cache = os.path.join(cache_dir, f"spgene_{chunk[0]}_{len(chunk)}.json")
        if os.path.exists(cache):
            rows = json.load(open(cache))
        else:
            id_list = ",".join(chunk)
            q = (f"in(genome_id,({id_list}))"
                 f"&select(genome_id,gene,product,property)&limit(25000)")
            rows = _get("sp_gene", q)
            json.dump(rows, open(cache, "w"))
            time.sleep(0.3)  # be polite to the public API
        for r in rows:
            prop = r.get("property", "")
            if prop not in FEATURE_PROPERTIES:
                continue
            gid = r.get("genome_id")
            if gid not in result:
                continue
            name = (r.get("gene") or r.get("product") or "").strip()
            if not name:
                continue
            # Normalise the virulence misspelling.
            prop_norm = "Virulence Factor" if prop.lower().startswith("virul") else prop
            result[gid].add(f"{prop_norm}::{name}")
    return result


def build_dataset(taxon_id: int, antibiotic: str, max_per_class: int,
                  out_dir: str) -> str:
    """Build a labeled gene-presence dataset and write a tidy CSV.

    Output CSV columns: genome_id, label (1=Resistant,0=Susceptible),
    then one 0/1 column per gene feature observed across the cohort.
    """
    import pandas as pd

    raw = os.path.join(out_dir, "raw")
    os.makedirs(out_dir, exist_ok=True)

    res_ids = fetch_labeled_ids(taxon_id, antibiotic, "Resistant", max_per_class, raw)[:max_per_class]
    sus_ids = fetch_labeled_ids(taxon_id, antibiotic, "Susceptible", max_per_class, raw)[:max_per_class]
    print(f"[fetch] {len(res_ids)} Resistant, {len(sus_ids)} Susceptible genome_ids")

    labels = {gid: 1 for gid in res_ids}
    labels.update({gid: 0 for gid in sus_ids})
    all_ids = list(labels)

    genes = fetch_genes_for_ids(all_ids, raw)
    # Drop genomes with zero recovered features (no usable signal).
    all_ids = [g for g in all_ids if genes.get(g)]
    feature_vocab = sorted({tok for g in all_ids for tok in genes[g]})
    print(f"[fetch] {len(all_ids)} genomes with features, {len(feature_vocab)} gene features")

    rows = []
    for gid in all_ids:
        row = {"genome_id": gid, "label": labels[gid]}
        present = genes[gid]
        for tok in feature_vocab:
            row[tok] = 1 if tok in present else 0
        rows.append(row)

    df = pd.DataFrame(rows)
    meta = {
        "source": "BV-BRC (PATRIC) genome_amr + sp_gene APIs",
        "taxon_id": taxon_id,
        "antibiotic": antibiotic,
        "n_genomes": len(df),
        "n_features": len(feature_vocab),
        "label_counts": dict(Counter(df["label"])),
        "positive_label": "Resistant",
    }
    csv_path = os.path.join(out_dir, "sentinel_dataset.csv")
    df.to_csv(csv_path, index=False)
    json.dump(meta, open(os.path.join(out_dir, "dataset_meta.json"), "w"), indent=2)
    print(f"[fetch] wrote {csv_path}  ({meta['label_counts']})")
    return csv_path


def fetch_one_genome_tokens(genome_id: str, cache_dir: str = "data/raw") -> set[str]:
    """Fetch the 'PROPERTY::gene' feature tokens for a single genome (live triage).

    Reuses the same sp_gene endpoint + caching as the training fetcher, so a demo
    sample is processed exactly like the training data. Returns an empty set on
    failure (the agent then flags low-information input).
    """
    try:
        genes = fetch_genes_for_ids([genome_id], cache_dir, batch=1)
        return genes.get(genome_id, set())
    except Exception:  # noqa: BLE001
        return set()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch real BV-BRC AMR data for SENTINEL")
    ap.add_argument("--taxon", type=int, default=573, help="NCBI taxon id (573=K. pneumoniae)")
    ap.add_argument("--antibiotic", default="ciprofloxacin")
    ap.add_argument("--max-per-class", type=int, default=150)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    build_dataset(args.taxon, args.antibiotic, args.max_per_class, args.out)


if __name__ == "__main__":
    main()
