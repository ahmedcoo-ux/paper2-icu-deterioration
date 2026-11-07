# Source: s5 split.ipynb, code cells [0, 1] (outputs not included)

# %% ---- cell 0 ----
"""
============================================================
S5_PRESPLIT — Split hourly_signals_with_labs into per-stay files
============================================================
Run this ONCE before s5_partA.py and s5_partB.py.

Reads:
    pipeline_outputs/s3/hourly_signals_with_labs.parquet

Writes:
    pipeline_outputs/s5_signals/<stay_id>.parquet
        ~88,713 small files, ~90 rows each
        Total disk: ~2–3 GB

Why:
    parquet filter pushdown on a single flat file still
    reads the entire file into RAM on most systems.
    Pre-splitting means each stay read in S5 is ~90 rows —
    peak RAM drops from 5 GB to ~50 MB.

Runtime: ~5–10 minutes (one read + ~88K small writes)
Safe to re-run: skips stay files that already exist.
============================================================
"""

import logging
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

# ─────────────────────────────────────────────────────────────
#  PATHS — edit to match your machine
# ─────────────────────────────────────────────────────────────
S3_DIR      = Path(r"<PHD_ROOT>\PHD Research\DATAset\s3")

SIGNALS_DIR = Path(r"<PHD_ROOT>\PHD Research\DATAset\s5_signals")
OUTPUT_DIR  = Path(r"<PHD_ROOT>\PHD Research\DATAset\s5")

BATCH_SIZE  = 500_000   # rows per pyarrow batch — safe for 8 GB RAM


# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "s5_presplit.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("PRESPLIT")


def run_presplit():
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  S5 Pre-split — One parquet file per stay_id    ║")
    log.info("╚══════════════════════════════════════════════════╝")

    src = S3_DIR / "hourly_signals_with_labs.parquet"
    if not src.exists():
        raise FileNotFoundError(
            f"Cannot find: {src.resolve()}\nRun S3 first."
        )

    # Which stays are already split (allows safe resume)
    already_done = {int(p.stem) for p in SIGNALS_DIR.glob("*.parquet")}
    log.info("Already split: %d stays — will skip these", len(already_done))

    pf        = pq.ParquetFile(src)
    stay_bufs = {}       # { stay_id: [chunk_df, ...] }
    n_written = len(already_done)
    batch_num = 0

    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        batch_num += 1
        chunk      = batch.to_pandas()
        chunk["stay_id"] = chunk["stay_id"].astype(int)

        # Accumulate rows per stay
        for sid, grp in chunk.groupby("stay_id", sort=False):
            if sid in already_done:
                continue
            if sid not in stay_bufs:
                stay_bufs[sid] = []
            stay_bufs[sid].append(grp)

        # Flush all stays except the last one in this chunk
        # (the last stay_id may continue into the next batch)
        last_sid = int(chunk["stay_id"].iloc[-1])
        to_flush = [s for s in list(stay_bufs) if s != last_sid]

        for sid in to_flush:
            df_stay = pd.concat(stay_bufs.pop(sid), ignore_index=True)
            df_stay.to_parquet(SIGNALS_DIR / f"{sid}.parquet", index=False)
            n_written += 1

        log.info("  Batch %3d | stays written: %6d | buffer: %d stays",
                 batch_num, n_written, len(stay_bufs))
        del chunk

    # Flush remaining buffer
    for sid, dfs in stay_bufs.items():
        df_stay = pd.concat(dfs, ignore_index=True)
        df_stay.to_parquet(SIGNALS_DIR / f"{sid}.parquet", index=False)
        n_written += 1

    log.info("Pre-split complete. %d stay files in %s", n_written, SIGNALS_DIR)


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  S5 Pre-split")
    print("=" * 60)
    print(f"  Source : {(S3_DIR / 'hourly_signals_with_labs.parquet').resolve()}")
    print(f"  Output : {SIGNALS_DIR.resolve()}")
    print()
    run_presplit()
    print("Done — now run s5_partA.py")

# %% ---- cell 1 ----

