# datasets/

Data inputs for IDPro, split into three categories. Only the small probe
data is committed; the large training corpora and AlphaFold PDBs are
gitignored and have to be regenerated locally via the download scripts in
each directory.

This directory is **data + downloaders only**. All evaluation,
fine-tuning, and analysis scripts live under `idpro/experiments/aim1/`.
Legacy P2T-era scripts (finetune / eval / probe runners) have been moved
to the gitignored `archive/` folder at the repo root.

## Layout

```
datasets/
├── probe_data/          # ── COMMITTED (data + downloaders) ──
│   ├── benchmark/        # microbiome + bioenergy enzyme manifests + downloader
│   └── dark_genome/      # 415-protein dark-genome eval set + downloader
│
├── training_data/       # ── SCRIPTS COMMITTED, qa_stages*/ gitignored ──
│   ├── scripts/          # download_*.py + build_records.py + generate_qa.py
│   └── qa_stages_*/      # generated multi-stage QA pairs (gitignored, ~GB)
│
└── alphafold/           # ── DOWNLOADERS COMMITTED, *.pdb + accessions gitignored ──
    ├── download_alphafold.py
    ├── download_structure.py
    ├── accessions.txt    # gitignored (1.8 MB)
    ├── pdbs/             # gitignored (multi-100k AF-*.pdb files)
    └── logs/             # gitignored
```

## Path resolution

Every script reads `IDPRO_DATA_ROOT` from the environment. If set, outputs
land at `$IDPRO_DATA_ROOT/{training_data,probe_data,alphafold}/...`. If
unset, each script falls back to the in-repo `datasets/<subdir>` location
it lives under, so a bare `python download_*.py` works from a fresh clone.

## Regenerating the large data

From a fresh clone (with `env.sh` sourced — or unsourced, to use the
in-repo defaults):

```bash
# 1. Source corpora (UniProt + InterPro + M-CSA + Prosite)
bash datasets/training_data/scripts/download_all.sh   # ~6 h, ~16 GB
python datasets/training_data/scripts/build_records.py  # parsed records + feature_index.pkl
python datasets/training_data/scripts/generate_qa.py    # multi-stage QA pairs into training_data/qa_stages/

# 2. AlphaFold PDBs (only needed for ESM3 with --structure-track)
python datasets/alphafold/download_alphafold.py        # populates alphafold/pdbs/ from accessions.txt
```
