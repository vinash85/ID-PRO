## Directory Structure

```
idpro_struct/
├── README.md
├── env.sh.template      # Copy → env.sh, edit, then `source env.sh`
├── .gitignore
│
├── idpro/                   # Single Python package (paths.py, config.py, …)
│   ├── paths.py             # Env-driven path resolution; import this
│   ├── config.py            # IDProConfig, EncoderConfig, LLMConfig, RAGConfig
│   ├── rag.py
│   ├── model/               # encoder, adaptor, projector, model, evidence, position
│   ├── data/                # dataset + collator
│   ├── training/            # train.py, configs/, launchers/
│   ├── eval/
│   │   ├── probes/          # frozen-backbone probe pipeline
│   │   ├── nl/              # natural-language eval (text-output baselines)
│   │   └── reports/         # report + figure generation
│   ├── metrics/             # conformal + selective-prediction
│   └── utils/
│
└── preliminary_data/
    ├── benchmark/           # Benchmark eval-set manifests + downloader
    ├── dark_genome/         # Dark-genome eval set + downloaders
    ├── rag/                 # RAG feasibility script
    └── training_data/
        └── scripts/         # UniProt / InterPro / M-CSA / Prosite / AlphaFold
                             # downloaders + record builder + QA generator
```

## First-time setup

```bash
cp env.sh.template env.sh
$EDITOR env.sh                # set IDPRO_DATA_ROOT, IDPRO_RUNS_ROOT, IDPRO_QWEN_PATH
source env.sh
```

Every script under `idpro/` resolves paths through `idpro/paths.py`, which reads
those environment variables. `env.sh` is gitignored.

Checkpoints, downloaded corpora, generated QA pairs, and pre-computed embeddings
are not tracked in git — they live on HuggingFace.
