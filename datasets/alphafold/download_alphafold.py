#!/usr/bin/env python3
"""
Download AlphaFold DB v6 PDB structures for the bacterial UniProt training pool.

Reads accessions from a flat text file (one per line), then fetches
  https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v6.pdb
in parallel via aiohttp. Idempotent — skips accessions whose PDB file already
exists in the output dir. 404s are logged to misses.txt and not retried.
Transient failures (timeouts, 5xx, 429) are retried with exponential backoff.

Usage:
  python download_alphafold.py --concurrency 24

Defaults are anchored at <IDPRO_DATA_ROOT or repo/datasets>/alphafold/, so a
bare invocation works after `source env.sh`. Override with --accessions /
--out-dir / --log-dir to point elsewhere.
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import aiohttp

# Resolve the alphafold root via env var (overrides) or the in-repo
# datasets/alphafold layout the script lives under.
DEFAULT_BASE = Path(os.environ.get(
    "IDPRO_DATA_ROOT",
    Path(__file__).resolve().parents[1],
)) / "alphafold"
URL_TEMPLATE = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.pdb"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--accessions", type=Path,
                   default=DEFAULT_BASE / "accessions.txt")
    p.add_argument("--out-dir", type=Path,
                   default=DEFAULT_BASE / "pdbs")
    p.add_argument("--log-dir", type=Path,
                   default=DEFAULT_BASE / "logs")
    p.add_argument("--concurrency", type=int, default=24,
                   help="Max concurrent HTTP requests")
    p.add_argument("--max-retries", type=int, default=4,
                   help="Retries per accession on transient failures")
    p.add_argument("--timeout", type=int, default=60,
                   help="Per-request timeout in seconds")
    p.add_argument("--progress-every", type=int, default=500,
                   help="Print progress every N completions")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after this many accessions (0 = no limit)")
    return p.parse_args()


class Counters:
    __slots__ = ("done", "skipped", "missed", "failed", "retries", "bytes", "started")

    def __init__(self):
        self.done = 0
        self.skipped = 0
        self.missed = 0
        self.failed = 0
        self.retries = 0
        self.bytes = 0
        self.started = time.time()


async def fetch_one(session, sem, acc, out_dir, missed_f, failed_f, retries, timeout, c):
    out_path = out_dir / f"AF-{acc}-F1-model_v6.pdb"
    if out_path.exists() and out_path.stat().st_size > 0:
        c.skipped += 1
        return

    url = URL_TEMPLATE.format(acc=acc)
    backoff = 1.0
    last_err = None

    async with sem:
        for attempt in range(retries + 1):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        # atomic-ish write: tmp then rename
                        tmp = out_path.with_suffix(".pdb.tmp")
                        tmp.write_bytes(data)
                        tmp.rename(out_path)
                        c.done += 1
                        c.bytes += len(data)
                        return
                    elif resp.status == 404:
                        missed_f.write(f"{acc}\n")
                        missed_f.flush()
                        c.missed += 1
                        return
                    elif resp.status in (429, 500, 502, 503, 504):
                        last_err = f"HTTP {resp.status}"
                        # Retry-After header if present
                        retry_after = resp.headers.get("Retry-After")
                        wait = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                        await asyncio.sleep(wait)
                        backoff = min(backoff * 2, 30.0)
                        c.retries += 1
                        continue
                    else:
                        last_err = f"HTTP {resp.status}"
                        break
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_err = f"{type(e).__name__}: {e}"
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                c.retries += 1
                continue

        # Exhausted retries
        failed_f.write(f"{acc}\t{last_err}\n")
        failed_f.flush()
        c.failed += 1


async def main_async(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    accessions = [a.strip() for a in args.accessions.read_text().splitlines() if a.strip()]
    if args.limit:
        accessions = accessions[: args.limit]
    total = len(accessions)
    print(f"[download_alphafold] {total} accessions, concurrency={args.concurrency}, "
          f"out={args.out_dir}", flush=True)

    misses_path = args.log_dir / "misses.txt"
    failed_path = args.log_dir / "failed.txt"

    c = Counters()
    sem = asyncio.Semaphore(args.concurrency)

    # connector with per-host limit a bit above concurrency to avoid contention
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2,
                                     limit_per_host=args.concurrency,
                                     ttl_dns_cache=300)
    headers = {"User-Agent": "idpro/0.1 (research; sahu lab)"}

    async def reporter(stop_event):
        """Periodic progress reporter — runs independently of task creation."""
        interval = 10.0  # seconds
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                pass
            completed = c.done + c.skipped + c.missed + c.failed
            elapsed = time.time() - c.started
            rate = completed / max(elapsed, 1e-3)
            eta_min = (total - completed) / max(rate, 1e-3) / 60.0
            print(f"  [{completed}/{total}] done={c.done} skip={c.skipped} "
                  f"miss={c.missed} fail={c.failed} retries={c.retries} "
                  f"GB={c.bytes/1e9:.2f} rate={rate:.1f}/s eta={eta_min:.1f}m",
                  flush=True)

    with open(misses_path, "a") as missed_f, open(failed_path, "a") as failed_f:
        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            stop_event = asyncio.Event()
            reporter_task = asyncio.create_task(reporter(stop_event))

            tasks = [
                asyncio.create_task(
                    fetch_one(session, sem, acc, args.out_dir,
                              missed_f, failed_f,
                              args.max_retries, args.timeout, c)
                )
                for acc in accessions
            ]
            await asyncio.gather(*tasks, return_exceptions=False)

            stop_event.set()
            await reporter_task

    elapsed = time.time() - c.started
    print(f"[download_alphafold] DONE in {elapsed/60:.1f}m  "
          f"done={c.done} skip={c.skipped} miss={c.missed} fail={c.failed} "
          f"retries={c.retries} GB={c.bytes/1e9:.2f}", flush=True)
    print(f"  misses logged to {misses_path}")
    print(f"  failures logged to {failed_path}")


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
