"""Fetch UniProt metadata (date_created, date_modified, Pfam IDs) for every
accession in the 3,637-protein labeled pool (reference + benchmark).

Output: idpro/data/probe/uniprot_metadata_cache.jsonl
  One JSON object per line with keys:
    accession, date_created, date_modified, pfam_ids (list of strings),
    taxonomy_id, organism_lineage (list)

Usage:
  python idpro/scripts/fetch_uniprot_metadata_for_e1.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from idpro.paths import PROBE_DIR as DATA_DIR  # noqa: E402

CACHE_PATH = DATA_DIR / "uniprot_metadata_cache.jsonl"

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{acc}.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("fetch_uniprot")


def load_accessions() -> list[str]:
    """Union of reference + benchmark accessions."""
    accs: set[str] = set()
    for name in ("reference.jsonl", "benchmark.jsonl"):
        p = DATA_DIR / name
        with p.open() as fh:
            for line in fh:
                d = json.loads(line)
                accs.add(d["accession"])
    return sorted(accs)


def load_cached() -> set[str]:
    if not CACHE_PATH.exists():
        return set()
    done: set[str] = set()
    with CACHE_PATH.open() as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["accession"])
            except Exception:
                continue
    return done


def extract_metadata(acc: str, d: dict) -> dict:
    # date_created / date_modified lives in entryAudit
    ea = d.get("entryAudit", {}) or {}
    # pfam IDs under uniProtKBCrossReferences where database == "Pfam"
    pfam_ids = []
    for xr in d.get("uniProtKBCrossReferences", []) or []:
        if xr.get("database") == "Pfam":
            pid = xr.get("id")
            if pid:
                pfam_ids.append(pid)
    organism = d.get("organism", {}) or {}
    tax_id = organism.get("taxonId")
    lineage = organism.get("lineage", []) or []
    return {
        "accession": acc,
        "date_created": ea.get("firstPublicDate"),
        "date_modified": ea.get("lastAnnotationUpdateDate"),
        "pfam_ids": pfam_ids,
        "taxonomy_id": tax_id,
        "organism_lineage": lineage[:10],  # cap for file size
    }


async def fetch_one(client: httpx.AsyncClient, acc: str,
                    sem: asyncio.Semaphore) -> dict | None:
    async with sem:
        for attempt in range(3):
            try:
                r = await client.get(UNIPROT_URL.format(acc=acc), timeout=30.0)
                if r.status_code == 200:
                    return extract_metadata(acc, r.json())
                if r.status_code == 404:
                    return {"accession": acc, "date_created": None, "date_modified": None,
                            "pfam_ids": [], "taxonomy_id": None, "organism_lineage": [],
                            "_note": "404 not found"}
                if r.status_code in (429, 502, 503, 504):
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {"accession": acc, "_error": f"HTTP {r.status_code}"}
            except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout) as e:
                await asyncio.sleep(2 ** attempt)
                if attempt == 2:
                    return {"accession": acc, "_error": str(e)[:200]}
    return {"accession": acc, "_error": "unreachable after retries"}


async def fetch_all(missing: list[str], concurrency: int = 20) -> None:
    sem = asyncio.Semaphore(concurrency)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Append mode: if script is re-run, only missing are fetched.
    fout = CACHE_PATH.open("a")
    t0 = time.time()
    n_ok = 0
    n_err = 0
    async with httpx.AsyncClient(headers={"User-Agent": "IDPro-E1/1.0"}) as client:
        tasks = [fetch_one(client, acc, sem) for acc in missing]
        for i, task in enumerate(asyncio.as_completed(tasks)):
            r = await task
            if r is None:
                continue
            if "_error" in r:
                n_err += 1
            else:
                n_ok += 1
            fout.write(json.dumps(r) + "\n")
            fout.flush()
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (len(missing) - (i + 1)) / max(rate, 1e-6)
                log.info(f"  {i + 1}/{len(missing)} done, "
                         f"ok={n_ok} err={n_err}, "
                         f"{rate:.1f}/s, eta {eta/60:.1f} min")
    fout.close()
    log.info(f"DONE: ok={n_ok} err={n_err} in {(time.time()-t0)/60:.1f} min")


def main() -> int:
    all_accs = load_accessions()
    cached = load_cached()
    missing = [a for a in all_accs if a not in cached]
    log.info(f"Total accessions: {len(all_accs)}")
    log.info(f"Already cached:   {len(cached)}")
    log.info(f"To fetch:         {len(missing)}")
    if not missing:
        log.info("Nothing to do.")
        return 0
    asyncio.run(fetch_all(missing, concurrency=20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
