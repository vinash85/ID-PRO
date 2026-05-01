# IDPro

Code for **IDPro** — a multimodal protein-function model that pairs a frozen
protein encoder (ESM C / ESM2 / ESM3) with a Qwen 3.5 27B language backbone via
a residue-preserving adaptor + projector bridge, then layers retrieval (RAG),
evidence-span supervision, and split-conformal prediction on top.

```
Protein seq (T residues)
  → Frozen encoder (ESM C / ESM2 / ESM3, dim D)
  → Residue Adaptor (1D Conv, kernel 7)
  → Per-residue MLP Projector (D → llm_dim)
  → Modality embed + protein-side position
  → [protein tokens ‖ RAG context ‖ question]   (text positions reset)
  → Qwen 3.5 27B (LoRA)
+ Evidence span heads (pre-LLM and at an intermediate LLM layer)
+ Split-conformal predictor for calibrated coverage
```

The encoder is dimension-agnostic. Swap ESM C ↔ ESM2 ↔ ESM3 by changing
`EncoderConfig.PRESETS` in `idpro/config.py`; the adaptor and projector pick up
the new dim automatically.

## Repository layout

```
idpro_struct/
├── README.md
├── env.sh.template               # cp → env.sh, edit, then `source env.sh`
├── .gitignore
│
├── idpro/                        # Single Python package
│   ├── paths.py                  # Env-driven path resolution; import this
│   ├── config.py                 # IDProConfig, EncoderConfig, LLMConfig, …
│   │
│   ├── model/
│   │   ├── model.py              # IDProModel: composes the two halves below
│   │   ├── p2t/                  # Protein-to-text base
│   │   │   ├── encoder.py        # Frozen ESM C / ESM2 / ESM3 wrapper
│   │   │   ├── adaptor.py        # Residue-preserving 1D Conv
│   │   │   ├── projector.py      # Per-residue MLP (encoder_dim → llm_dim)
│   │   │   └── position.py       # Protein-side sinusoidal pos; text reset
│   │   └── idpro/                # Reliability layers
│   │       ├── rag.py            # ESM-embedding retrieval (no alignment)
│   │       ├── evidence.py       # Pre-LLM + post-LLM evidence span heads
│   │       └── conformal.py      # SplitConformalPredictor + token_f1
│   │
│   ├── data/
│   │   ├── dataset.py            # QA dataset (multi-stage)
│   │   └── batch.py              # Collator
│   │
│   ├── training/
│   │   ├── train.py              # Canonical Stage-1 / Stage-4 trainer
│   │   ├── configs/ds_zero2.json # DeepSpeed ZeRO-2 config
│   │   └── launchers/setup_qwen.sh  # First-time Qwen weight download
│   │
│   ├── experiments/              # Eval scripts, organised by paper aim
│   │   └── aim1/
│   │       ├── a_benchmarks/     # Accuracy / temporal / orphan / scaling
│   │       │   ├── extract_probe_embeddings.py
│   │       │   ├── evaluate_ec_classifier.py     → EC-L1 macro-AUC
│   │       │   ├── evaluate_probe_on_dark.py     → dark-genome eval
│   │       │   ├── idpro_vs_interlabelgo.py      → head-to-head
│   │       │   ├── train_probe_variants.py       → scaling curve
│   │       │   └── …
│   │       ├── b_trust/          # Calibration + audit
│   │       │   ├── conformal_selective_curve.py
│   │       │   ├── conformal_on_classifier.py
│   │       │   └── run_e1_conformal_splits.py
│   │       └── reports/          # Figure / report builders
│   │
│   ├── metrics/                  # Pure metric primitives (placeholder)
│   └── utils/                    # build_structure_manifest, filter_qa_by_structure
│
└── preliminary_data/
    ├── benchmark/                # Benchmark eval-set manifests + downloader
    ├── dark_genome/              # Dark-genome eval set + downloaders
    ├── rag/                      # RAG feasibility script
    └── training_data/
        └── scripts/              # UniProt / InterPro / M-CSA / Prosite /
                                  # AlphaFold downloaders + record builder
                                  # + QA generator
```

## First-time setup

```bash
cp env.sh.template env.sh
$EDITOR env.sh                  # set IDPRO_DATA_ROOT, IDPRO_RUNS_ROOT, IDPRO_QWEN_PATH
source env.sh                   # re-source whenever you open a new shell
```

`env.sh` is gitignored — it stays local. Every script under `idpro/` resolves
data and output paths through `idpro/paths.py`, which reads those environment
variables. There is no hardcoded path in the package.

## Training

```bash
# Stage 1: warm-start adaptor + projector + LoRA
python idpro/training/train.py --stage 1

# Stage 4: full multimodal fine-tune
python idpro/training/train.py --stage 4 --resume

# Multi-GPU via DeepSpeed
deepspeed --num_gpus=N idpro/training/train.py --stage 4 \
    --deepspeed idpro/training/configs/ds_zero2.json
```

CLI flags override the defaults in `idpro/paths.py` (e.g. `--qa-dir`,
`--ckpt-dir`, `--results-dir`).

## Evaluation

```bash
# Extract probe embeddings (uses Stage-4 checkpoint + frozen encoder)
python idpro/experiments/aim1/a_benchmarks/extract_probe_embeddings.py \
    --ckpt $CKPT --which reference --encoder esmc-600m

# Downstream probes
python idpro/experiments/aim1/a_benchmarks/evaluate_ec_classifier.py
python idpro/experiments/aim1/a_benchmarks/evaluate_probe_on_dark.py
python idpro/experiments/aim1/a_benchmarks/idpro_vs_interlabelgo.py

# Calibration (split-conformal)
python idpro/experiments/aim1/b_trust/conformal_selective_curve.py
python idpro/experiments/aim1/b_trust/conformal_on_classifier.py
python idpro/experiments/aim1/b_trust/run_e1_conformal_splits.py
```

## Encoder swap

Pick an encoder by name; the rest of the model adapts automatically.

| Preset       | Backend | Dim   | Structure track |
|--------------|---------|-------|-----------------|
| `esmc-600m`  | ESM C   | 1152  | —               |
| `esmc-300m`  | ESM C   | 960   | —               |
| `esm2-650m`  | ESM2    | 1280  | —               |
| `esm2-3b`    | ESM2    | 2560  | —               |
| `esm3-1.4b`  | ESM3    | 1536  | optional (per-sample PDB) |

```python
from idpro.config import IDProConfig

cfg = IDProConfig()
cfg.encoder.name = "esm3-1.4b"
cfg.encoder.structure_track = True            # only meaningful for esm3
cfg.encoder.structure_manifest_path = "manifest.jsonl"
cfg.resolve()
```

For ESM3 with structure on, the manifest is JSONL with
`{"accession": ..., "pdb_path": ...}` per line; the encoder loads the PDB at
encode-time and routes it through ESM3's structure track.
