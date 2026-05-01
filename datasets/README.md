# datasets/

Data inputs for IDPro, split into three categories. Only the small probe
data is committed; the large training corpora and AlphaFold PDBs are
gitignored and have to be regenerated locally via the download scripts in
each directory.

## Layout

```
datasets/
├── probe_data/          # ── COMMITTED ──
│   ├── benchmark/        # microbiome + bioenergy enzyme manifests
│   ├── dark_genome/      # 415-protein dark-genome eval set
│   └── rag/              # RAG feasibility experiment
│
├── training_data/       # ── SCRIPTS COMMITTED, qa_stages*/ gitignored ──
│   ├── scripts/          # download_*.py + build_records.py + generate_qa.py
│   ├── finetune_*.py     # legacy LoRA finetune utilities
│   ├── eval_finetuned.py
│   ├── regenerate_qa_sequence_informed.py
│   ├── run_finetune.sh
│   └── qa_stages_*/      # generated multi-stage QA pairs (gitignored, ~GB)
│
└── alphafold/           # ── DOWNLOADERS COMMITTED, *.pdb + accessions gitignored ──
    ├── download_alphafold.py
    ├── download_structure.py
    ├── accessions.txt    # gitignored (1.8 MB)
    ├── pdbs/             # gitignored (multi-100k AF-*.pdb files)
    └── logs/             # gitignored
```

## Regenerating the large data

From a fresh clone (with `env.sh` sourced):

```bash
# 1. Source corpora (UniProt + InterPro + M-CSA + Prosite)
cd datasets/training_data/scripts
bash download_all.sh                # ~6 h, ~16 GB
python build_records.py             # parsed records + feature_index.pkl
python generate_qa.py               # multi-stage QA pairs into ../qa_stages/

# 2. AlphaFold PDBs (only needed for ESM3 with --structure-track)
cd ../../alphafold
python download_alphafold.py        # populates pdbs/ from accessions.txt
```

The `IDPRO_DATA_ROOT` env var lets you point the training/eval scripts at a
different on-disk location if you keep these large inputs outside the repo.
