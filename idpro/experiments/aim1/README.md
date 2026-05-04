# Aim 1 — Establish AI advantage on the dark proteome

This directory contains all Phase-I experiments that demonstrate IDPro's
advantage over text-based baselines on dark, orphan, and temporally-held-out
proteins, plus the calibration / trust-boundary work.

## Layout

```
aim1/
  probe_benchmarks/             Subaim 1A. Accuracy, scaling, orphan, dark-genome.
    data_prep/                  Build reference + benchmark + dark splits + label vocabs
      prepare_reference_benchmark.py   → datasets/probe_data/probe_splits/{reference,benchmark}.jsonl
      prepare_dark.py                  → datasets/probe_data/probe_splits/dark.jsonl
    extract_embeddings.py       (rag-index | views) — write the 3-view IDPro caches
                                and the ESM C mean-pool RAG index.
    run_probe.py                (variants | cv5fold | dark | zeroshot637 | fair) —
                                end-to-end probe training + evaluation runner.
    run_baselines.py            (mmseqs | deepfri | deepgometa | clean | interlabelgo) —
                                one-stop baseline EC-prediction runner.
    utils/                      Shared probe heads, view stacking, label loaders,
                                strict-keyword AUC scoring rule.

  conformal/                    Subaim 1B. Calibration + audit + red-team.
    selective_curve.py          AUC-vs-coverage on the 80/20 split (oracle vs conformal).
    classifier_conformal.py     Split conformal on the 8-way EC classifier;
                                evaluation on dark + benchmark.
    shift_splits.py             E1 robustness — temporal / Pfam-family / synthetic
                                shifts, weighted vs unweighted conformal.
    fetch_uniprot_metadata.py   Build datasets/probe_data/uniprot_metadata_cache.jsonl
                                (used by shift_splits.py).

  reports/                      Figure / report builders.
    make_spider_plot_v2.py      Strict-keyword EC spider on benchmark + dark.
    build_contrast_report.py    Dark-protein contrast examples.
    analyze_f1_meaning.py       What does token-F1 mean? — calibration anchor.
    compose_figure1.py          Fig 1 composite (a + b + c + d).
    compose_figures.py          Fig 1 (spider) + Fig 2 (composite) renderers.
```

## Data flow

```
datasets/probe_data/
  benchmark/, dark_genome/                 Source manifests (committed)
  probe_splits/{reference,benchmark,dark}.jsonl   ← data_prep/
  extracted_embeddings/{*_embeddings.pt, rag_index.npz}   ← extract_embeddings.py
  baseline_predictions/{method}_{split}_predictions.json  ← run_baselines.py
  uniprot_metadata_cache.jsonl             ← conformal/fetch_uniprot_metadata.py

$IDPRO_RUNS_ROOT/aim1/
  probe_benchmarks/{variants,cv5fold,dark,zeroshot637,fair}.json    ← run_probe.py
  conformal/{selective_curve,classifier_conformal,shift_splits}.json ← conformal/
```

All paths above resolve through `idpro.paths` (`PROBE_SPLITS_DIR`,
`EXTRACTED_EMBEDDINGS_DIR`, `BASELINE_PREDS_DIR`, `UNIPROT_METADATA_CACHE`,
`PROBE_RESULTS_DIR`, `CONFORMAL_RESULTS_DIR`, `FIGURES_DIR`).

## Paper-figure map

| Figure | Builder |
|---|---|
| Fig 1B (architecture) | `reports/compose_figure1.py` |
| Fig 2A (probe spider) | `reports/make_spider_plot_v2.py` ← `probe_benchmarks/run_probe.py cv5fold` |
| Fig 2B (NL head-to-head) | `probe_benchmarks/run_probe.py zeroshot637` |
| Fig 3A (scaling) | `probe_benchmarks/run_probe.py variants` |
| Fig 3B (selective curve) | `conformal/selective_curve.py` |
| Fig 3C (per-shift conformal) | `conformal/shift_splits.py` |

## Typical end-to-end run

```bash
source env.sh

# 1. Splits + label vocab
python idpro/experiments/aim1/probe_benchmarks/data_prep/prepare_reference_benchmark.py
python idpro/experiments/aim1/probe_benchmarks/data_prep/prepare_dark.py

# 2. Embedding caches
python idpro/experiments/aim1/probe_benchmarks/extract_embeddings.py rag-index --include benchmark
for split in reference benchmark dark; do
  python idpro/experiments/aim1/probe_benchmarks/extract_embeddings.py views \
      --which $split --ckpt $IDPRO_RUNS_ROOT/checkpoints/stage4_step80000
done

# 3. Probe sweeps + dark eval
python idpro/experiments/aim1/probe_benchmarks/run_probe.py variants
python idpro/experiments/aim1/probe_benchmarks/run_probe.py cv5fold
python idpro/experiments/aim1/probe_benchmarks/run_probe.py dark
python idpro/experiments/aim1/probe_benchmarks/run_probe.py zeroshot637

# 4. Baselines
for m in mmseqs deepfri deepgometa interlabelgo; do
  python idpro/experiments/aim1/probe_benchmarks/run_baselines.py $m \
      --splits benchmark dark
done

# 5. Conformal
python idpro/experiments/aim1/conformal/selective_curve.py
python idpro/experiments/aim1/conformal/classifier_conformal.py
python idpro/experiments/aim1/conformal/fetch_uniprot_metadata.py
python idpro/experiments/aim1/conformal/shift_splits.py

# 6. Figures
python idpro/experiments/aim1/reports/make_spider_plot_v2.py
python idpro/experiments/aim1/reports/compose_figures.py
```
