"""
Convert CLEAN's per-protein EC predictions into the JSON format that
make_spider_plot_v2.py expects (one dict: accession → "text" string with
EC numbers + scores). The strict-keyword scorer in the existing pipeline
will then convert these to per-class strict scores.

CLEAN output format (CSV, no header):
   protein_id,EC:x.x.x.x/score,EC:x.x.x.x/score,...

Output: benchmark/results/clean_{set}_predictions.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from idpro.paths import DATA_ROOT as REPO  # noqa: E402

SRC_DIR = REPO / "benchmark" / "CLEAN" / "app" / "results" / "inputs"
OUT_DIR = REPO / "benchmark" / "results"

SET_NAMES = {"benchmark": "clean", "dark": "clean", "reference": "clean"}


def parse_one(csv_path: Path) -> dict:
    out = {}
    with csv_path.open() as f:
        for row in csv.reader(f):
            if not row:
                continue
            pid = row[0]
            # Each subsequent field is "EC:x.x.x.x/score"
            preds = []
            for tok in row[1:]:
                if "/" not in tok:
                    continue
                ec, score = tok.split("/", 1)
                ec = ec.replace("EC:", "").strip()
                try:
                    s = float(score)
                except ValueError:
                    s = 0.0
                preds.append((ec, s))
            # Build a "text" string the existing strict_class_scores can parse:
            # we include the EC numbers AND a natural-language tag for each so
            # the keyword matcher (which looks for words like "hydrolase") will
            # still work. CLEAN outputs JUST EC numbers, no enzyme names — so
            # we map EC L1 to canonical enzyme-class name on the fly.
            ec_to_name = {
                "1": "oxidoreductase", "2": "transferase", "3": "hydrolase",
                "4": "lyase", "5": "isomerase", "6": "ligase", "7": "translocase",
            }
            parts = []
            for ec, s in preds:
                ec1 = ec.split(".")[0]
                name = ec_to_name.get(ec1, "")
                parts.append(f"EC {ec} ({name}, score={s:.3f})")
            out[pid] = "; ".join(parts)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for set_name in ["benchmark", "dark", "reference"]:
        src = SRC_DIR / f"{set_name}_maxsep.csv"
        if not src.exists():
            print(f"  skip {set_name}: {src} missing")
            continue
        preds = parse_one(src)
        out_path = OUT_DIR / f"clean_{set_name}_predictions.json"
        out_path.write_text(json.dumps(preds, indent=2))
        print(f"{set_name}: {len(preds)} proteins → {out_path}")


if __name__ == "__main__":
    main()
