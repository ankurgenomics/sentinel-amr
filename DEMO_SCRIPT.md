# SENTINEL - 5-minute interview demo script

A tight, honest walkthrough you can run live or narrate from screenshots.

## Before the room (1 min setup, do this beforehand)

```bash
cd Interview_slides
source .venv-sentinel/bin/activate     # or use .venv-sentinel/bin/python directly
```

Have `metrics/roc.png`, `metrics/shap_global.png`, and an `output/*_report.md` open.

## The 5-minute flow

### 0:00 - Frame the problem (30s)
"SFA's mandate is AI/ML for pathogen characterisation and risk assessment from genomic
data. The hard, real problem isn't building one more classifier - it's triaging
surveillance data fast, explaining every call to an auditor, and staying safe when a
sample doesn't match anything in the database. I built a small working system that does
exactly that."

### 0:30 - Show the honest result (60s)
"On 298 real *Klebsiella pneumoniae* genomes from BV-BRC, labelled by lab AST, predicting
ciprofloxacin resistance: ROC-AUC 0.89 on a **group-aware** holdout, so related isolates
never leak across the split. That's a genuine generalisation number - I'd rather show you
0.89 honest than 0.99 leaked."

Show `metrics/roc.png` and `metrics/confusion.png`.

### 1:30 - Show it learned real biology (60s)
Show `metrics/shap_global.png`.
"I didn't tell it any biology. SHAP shows it relies on CTX-M-15, the acrAB efflux
regulator, and DNA topoisomerase IV - the actual fluoroquinolone target. It rediscovered
the resistance mechanisms. Every prediction comes with this gene-level audit trail."

### 2:30 - Run the agent live (90s)
```bash
python -m sentinel.demo --show-novel
```
Narrate as it prints:
- "Each genome flows through a bounded LangGraph agent: classify -> novelty check ->
  SHAP -> a critic node -> report."
- Point at a correct Resistant call (high confidence, flagged for stewardship).
- Point at the **low-confidence** case: "the critic flags it for phenotypic AST
  confirmation instead of trusting it."
- Point at **SYNTHETIC_NOVEL_001**: "an organism with no database match - novelty
  distance 1.0, flagged NOVEL, escalated to a human. The model's confident-looking
  probability is explicitly *not* trusted. That's the PathogenFinder2 insight in action."

### 4:00 - Show the report + scale story (60s)
Open `output/SYNTHETIC_NOVEL_001_report.md`: "auditable markdown for a risk officer, JSON
for a LIMS." Then: "Same architecture scales to ESM-2 embeddings, a pan-genome GNN, VAE
novelty, and streaming GenomeTrakr/GISAID on AWS - that's in SCALING.md."

### 5:00 - Land it
"Small, honest, runnable today - and a clear path to a national platform. The point is a
*deployable, auditable, novelty-aware* surveillance system, which is what an agency can
actually trust in production."

## Likely questions + answers

- **"Why gene-presence, not k-mers / deep learning end-to-end?"**
  Interpretability and speed for the prototype - SHAP names real genes an auditor trusts.
  The scale-up swaps in ESM-2/DNABERT embeddings and a GNN (SCALING.md). I chose the
  representation that makes the *demo honest and auditable*, not the flashiest.

- **"0.81 accuracy isn't production-grade."**
  Correct, and that's the point of the design: confidence gating, novelty detection, and
  human-in-the-loop mean the system never auto-acts on weak or novel cases. Production
  accuracy comes from more data + embeddings + lineage-split tuning.

- **"How do you know it's not leaking / overfitting?"**
  Group-aware split (GroupShuffleSplit on assembly clusters); I can also hold out an entire
  sequence type (e.g. ST258) to prove it survives new clones - the standard AMR-ML failure
  mode.

- **"What about a brand-new pathogen?"**
  That's the novelty node. Alignment-free OOD distance here; VAE reconstruction error on
  protein-LM embeddings at scale. It escalates rather than guessing.

- **"Could the critic loop forever?"**
  No - it's a single-pass gate by design. Deterministic and auditable; no cyclic rewrite.

- **"Is the data real?"**
  Yes - BV-BRC `genome_amr` AST labels + `sp_gene` CARD/VFDB gene calls. `fetch_data.py`
  documents the exact API queries; the dataset is committed for offline reproducibility.
