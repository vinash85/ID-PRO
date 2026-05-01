"""
Convert InterLabelGO+ outputs (TSV per workdir) into the JSON format that
make_spider_plot_v2.py expects: one dict accession→"text", where the text
contains GO IDs + names so the existing strict_class_scores or
deepgometa_strict_scores helpers can score it.

InterLabelGO+ outputs both raw GO IDs AND their natural-language names. We
include both so the matcher can use either keyword path.

Output: benchmark/results/interlabelgo_{set}_predictions.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from idpro.paths import DATA_ROOT as REPO  # noqa: E402

SRC_BASE = REPO / "benchmark" / "InterLabelGO" / "InterLabelGO+"
OUT_DIR = REPO / "benchmark" / "results"

# Only keep the top-K terms per protein (avoids the matcher seeing 500+ low-conf
# terms which dilutes the strict-keyword evaluation).
TOP_K_PER_PROTEIN = 30
# And ignore terms below a minimum confidence
MIN_SCORE = 0.05


def parse_one(tsv_path: Path) -> dict:
    out: dict[str, list[tuple]] = defaultdict(list)
    with tsv_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                score = float(row["score"])
            except (KeyError, ValueError):
                continue
            if score < MIN_SCORE:
                continue
            entry = row["EntryID"]
            out[entry].append((score, row["term"], row.get("go_term_name", "").strip(),
                               row.get("aspect", "")))

    # Build the prediction string per protein
    final = {}
    for entry, terms in out.items():
        terms.sort(reverse=True)  # by score desc
        terms = terms[:TOP_K_PER_PROTEIN]
        # Strip prefixes like "sp|P01308|INS_HUMAN" → "P01308" if present
        pid = entry
        if "|" in pid:
            pid = pid.split("|")[1]
        # Compose text: "GO:xxxxxxx (name) [aspect, score]; ..."
        parts = []
        for s, go, name, asp in terms:
            parts.append(f"{go} {name} ({asp}, score={s:.3f})")
        final[pid] = "; ".join(parts)
    return final


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for set_name in ["benchmark", "dark", "reference"]:
        src = SRC_BASE / f"work_{set_name}" / "InterLabelGO+.tsv"
        if not src.exists():
            print(f"  skip {set_name}: {src} missing")
            continue
        preds = parse_one(src)
        out_path = OUT_DIR / f"interlabelgo_{set_name}_predictions.json"
        out_path.write_text(json.dumps(preds, indent=2))
        print(f"{set_name}: {len(preds)} proteins → {out_path}")


if __name__ == "__main__":
    main()
