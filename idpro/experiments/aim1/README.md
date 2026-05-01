# Aim 1 — Establish AI advantage on the dark proteome

This directory contains all Phase-I experiments that demonstrate IDPro's
advantage over text-based baselines on dark, orphan, and temporally-held-out
proteins, plus the calibration / trust-boundary work.

## Layout

- `a_benchmarks/` — **Subaim 1A.** Accuracy, temporal, orphan, scaling.
  Includes the frozen-backbone probe pipeline (EC-L1 macro-AUC, dark-genome
  eval) and the natural-language head-to-head against InterLabelGo / CLEAN.
  Outputs Fig 2A (probe spider plot), Fig 2B (NL head-to-head), Fig 3A (scaling).

- `b_trust/` — **Subaim 1B.** Calibration + audit + red-team.
  Conformal-prediction selective curves (Fig 3B), per-shift conformal robustness
  (Fig 3C, temporal/Pfam/synthetic), the rationale-audit placeholder (200-protein
  M2 deliverable), and the Sandia red-team placeholder.

- `reports/` — Figure / report builders. Spider plots, contrast reports,
  Fig 1 / Fig 2 composers.

## Paper-figure map

| Figure | Builder |
|---|---|
| Fig 1B (architecture) | `reports/compose_figure1.py` |
| Fig 2A (probe spider) | `reports/make_spider_plot_v2.py` ← `a_benchmarks/evaluate_ec_classifier.py` |
| Fig 2B (NL head-to-head) | `a_benchmarks/idpro_vs_interlabelgo.py` |
| Fig 3A (scaling) | `a_benchmarks/train_probe_variants.py` |
| Fig 3B (selective curve) | `b_trust/conformal_selective_curve.py` |
| Fig 3C (per-shift conformal) | `b_trust/run_e1_conformal_splits.py` |
