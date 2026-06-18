"""SENTINEL: explainable, agentic pathogen-surveillance triage.

A small, honest, locally-runnable prototype built for the SFA
(Singapore Food Agency) Bioinformatics AI/ML interview.

Pipeline: real BV-BRC labels + curated AMR/virulence gene features
-> XGBoost classifier -> SHAP explainability -> alignment-free novelty
flag -> bounded LangGraph triage agent -> auditable risk report.
"""

__version__ = "0.1.0"
