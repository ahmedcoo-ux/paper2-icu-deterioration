# Source: S5 merge.ipynb, code cells [0, 1, 2, 3, 4, 5] (outputs not included)

# %% ---- cell 0 ----

"""
============================================================
S5 Merge — Ultra Memory-Safe (Batch-by-Batch)
============================================================
Reads Part A and Part B in small row batches.
Never loads a full part into RAM.
Peak RAM: ~200-300 MB regardless of file size.

Strategy:
    PyArrow ParquetFile.iter_batches(batch_size=50_000)
    reads 50,000 rows at a time from each part file,
    writes each batch immediately to output parquet,
    then discards it before reading the next batch.
============================================================
"""

import logging
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
S4_DIR      = Path(r"<PHD_ROOT>\PHD Research\DATAset\s4")
#SIGNALS_DIR = Path(r"<PHD_ROOT>\PHD Research\DATAset\s5_signals")   # per-stay files
OUTPUT_DIR  = Path(r"<PHD_ROOT>\PHD Research\DATAset\s5")


# Rows read per batch — lower = less RAM, more iterations
# 50,000 rows × 188 cols × 8 bytes ≈ 75 MB per batch
BATCH_SIZE = 50_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "s5_merge.log",
                            mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("S5_MERGE")


def stream_file_to_writer(src_path: Path,
                           writer: pq.ParquetWriter,
                           label: str) -> int:
    """
    Reads src_path in batches of BATCH_SIZE rows.
    Writes each batch to writer immediately.
    Returns total rows written.
    Never loads the full file into RAM.
    """
    pf         = pq.ParquetFile(str(src_path))
    total_rows = 0
    batch_num  = 0

    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        batch_num  += 1
        total_rows += batch.num_rows
        writer.write_batch(batch)
        del batch

        if batch_num % 10 == 0:
            log.info("  %s — batch %d | rows written: %d",
                     label, batch_num, total_rows)

    log.info("  %s done — %d rows in %d batches",
             label, total_rows, batch_num)
    return total_rows


def run_merge():
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  S5 Merge — Batch-by-Batch (Ultra Low RAM)      ║")
    log.info("╚══════════════════════════════════════════════════╝")
    log.info("BATCH_SIZE = %d rows (~%d MB per batch)",
             BATCH_SIZE, BATCH_SIZE * 188 * 8 // 1_048_576)

    path_a   = OUTPUT_DIR / "window_features_partA.parquet"
    path_b   = OUTPUT_DIR / "window_features_partB.parquet"
    out_path = OUTPUT_DIR / "window_features.parquet"

    for p in [path_a, path_b]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p.name} not found at {p.resolve()}\n"
                f"Check OUTPUT_DIR at top of script."
            )

    # Read schema from Part A (small operation — metadata only)
    log.info("Reading schema from Part A ...")
    schema = pq.read_schema(str(path_a))
    log.info("  Schema: %d columns", len(schema.names))

    # Open output writer
    log.info("Opening output writer -> %s", out_path)
    writer = pq.ParquetWriter(str(out_path), schema)

    # Stream Part A
    log.info("Streaming Part A ...")
    n_a = stream_file_to_writer(path_a, writer, "Part A")

    # Stream Part B
    log.info("Streaming Part B ...")
    n_b = stream_file_to_writer(path_b, writer, "Part B")

    writer.close()
    total = n_a + n_b
    log.info("Writer closed. Total rows: %d (A=%d, B=%d)", total, n_a, n_b)

    # ── Sanity checks — column-by-column, low RAM ──────────
    log.info("Running sanity checks ...")
    failures = []

    # Load windows table (small — only needed for checks)
    windows = pd.read_parquet(S4_DIR / "windows.parquet")

    # Check 1: row count
    meta = pq.read_metadata(str(out_path))
    if meta.num_rows != len(windows):
        failures.append(
            f"Check 1 FAIL: {meta.num_rows:,} rows "
            f"vs {len(windows):,} expected"
        )
    else:
        log.info("  Check 1 passed: row count correct (%d)", meta.num_rows)

    # Check 2: no duplicate window_id
    # Read window_id column only (tiny)
    wids = pq.read_table(
        str(out_path), columns=["window_id"]
    ).to_pandas()["window_id"]
    n_dup = int(wids.duplicated().sum())
    if n_dup:
        failures.append(f"Check 2 FAIL: {n_dup} duplicate window_ids")
    else:
        log.info("  Check 2 passed: no duplicate window_ids")
    del wids

    # Check 3: no stay_id overlap between parts
    # Read stay_id column only from each part
    stays_a = set(
        pq.read_table(str(path_a), columns=["stay_id"])
        .to_pandas()["stay_id"]
    )
    stays_b = set(
        pq.read_table(str(path_b), columns=["stay_id"])
        .to_pandas()["stay_id"]
    )
    overlap = stays_a & stays_b
    if overlap:
        failures.append(
            f"Check 3 FAIL: {len(overlap)} stay_ids in both parts"
        )
    else:
        log.info("  Check 3 passed: no stay_id overlap")
    del stays_a, stays_b

    # Check 4: label integrity
    feat_lab = pq.read_table(
        str(out_path), columns=["window_id", "label"]
    ).to_pandas()
    mc  = feat_lab.merge(
        windows[["window_id", "label"]],
        on="window_id", suffixes=("_f", "_w")
    )
    bad = int((mc["label_f"] != mc["label_w"]).sum())
    if bad:
        failures.append(f"Check 4 FAIL: {bad} label mismatches")
    else:
        log.info("  Check 4 passed: labels correct")
    del feat_lab, mc, windows

    # Check 5: column count consistent
    n_cols_a = len(pq.read_schema(str(path_a)).names)
    n_cols_b = len(pq.read_schema(str(path_b)).names)
    if n_cols_a != n_cols_b:
        failures.append(
            f"Check 5 FAIL: Part A={n_cols_a}, Part B={n_cols_b} columns"
        )
    else:
        log.info("  Check 5 passed: %d columns consistent", n_cols_a)

    if failures:
        msg = "Merge FAILED:\n" + "\n".join(failures)
        log.error(msg)
        raise AssertionError(msg)

    log.info("All 5 checks passed")

    # ── Summary (read only label column) ───────────────────
    label_col = pq.read_table(
        str(out_path), columns=["label"]
    ).to_pandas()["label"]
    n_pos   = int(label_col.sum())
    n_total = int(len(label_col))
    del label_col

    all_cols  = pq.read_schema(str(out_path)).names
    feat_cols = [c for c in all_cols
                 if c not in ["window_id","stay_id","subject_id","label"]]

    summary = "\n".join([
        "",
        "=" * 60,
        "  S5 FEATURE ENGINEERING SUMMARY",
        "=" * 60,
        f"  Total windows  : {n_total:>10,}",
        f"  Total features : {len(feat_cols):>10,}",
        f"  Positive (1)   : {n_pos:>10,}  ({n_pos/n_total*100:.2f}%)",
        f"  Negative (0)   : {n_total-n_pos:>10,}  "
        f"({(n_total-n_pos)/n_total*100:.2f}%)",
        "",
        f"  Saved: {out_path.resolve()}",
        "=" * 60,
        "",
    ])
    log.info(summary)
    (OUTPUT_DIR / "s5_summary.txt").write_text(summary, encoding="utf-8")
    log.info("S5 Merge DONE.")
    return out_path


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  S5 Merge — Ultra Memory-Safe")
    print("=" * 60)
    print(f"  OUTPUT_DIR = {OUTPUT_DIR.resolve()}")
    print(f"  BATCH_SIZE = {BATCH_SIZE:,} rows")
    print(f"  Peak RAM   ~ {BATCH_SIZE * 188 * 8 // 1_048_576} MB per batch")
    print()

    ok = True
    for p in [
        OUTPUT_DIR / "window_features_partA.parquet",
        OUTPUT_DIR / "window_features_partB.parquet",
        S4_DIR     / "windows.parquet",
    ]:
        if not p.exists():
            print(f"  Not found: {p.resolve()}")
            ok = False
    if not ok:
        print()
        print("  Fix: check OUTPUT_DIR and S4_DIR at top of script.")
        raise SystemExit(1)

    out_path = run_merge()

    meta      = pq.read_metadata(str(out_path))
    all_cols  = pq.read_schema(str(out_path)).names
    feat_cols = [c for c in all_cols
                 if c not in ["window_id","stay_id","subject_id","label"]]
    print()
    print(f"Rows     : {meta.num_rows:,}")
    print(f"Features : {len(feat_cols)}")
    print(f"Saved -> {out_path.resolve()}")

# %% ---- cell 1 ----
import pandas as pd

windows = pd.read_parquet(r'<PHD_ROOT>\PHD Research\DATAset\s5\window_features.parquet')

print("Shape:", windows.shape)
print("Columns sample:", list(windows.columns[:20]))

# Check if map trend features already exist
map_cols = [c for c in windows.columns if 'map' in c.lower()]
print("\nMAP features in S5:", map_cols)

# Check if fio2 trend features exist
fio2_cols = [c for c in windows.columns if 'fio2' in c.lower()]
print("FiO2 features in S5:", fio2_cols)

# Check if gcs sub-score trend features exist
gcs_cols = [c for c in windows.columns if 'gcs' in c.lower()]
print("GCS features in S5:", gcs_cols)

# %% ---- cell 2 ----
import pandas as pd
windows = pd.read_parquet(r'<PHD_ROOT>\PHD Research\DATAset\s5\window_features.parquet')
print(windows.columns.tolist())

# %% ---- cell 3 ----
import pandas as pd

s2b    = pd.read_parquet(r'<PHD_ROOT>\PHD Research\DATAset\s2b\hourly_signals_with_gcs.parquet')
s3     = pd.read_parquet(r'<PHD_ROOT>\PHD Research\DATAset\s3\hourly_signals_with_labs.parquet')
s4     = pd.read_parquet(r'<PHD_ROOT>\PHD Research\DATAset\s4\windows.parquet')

print("=== S2b columns ===")
print(list(s2b.columns))

print("\n=== S3 columns ===")
print(list(s3.columns))

print("\n=== S4 columns ===")
print(list(s4.columns))

print("\n=== S4 sample (first 3 rows) ===")
print(s4.head(3))

print("\n=== Window ID sample from S5 ===")
windows = pd.read_parquet(r'<PHD_ROOT>\PHD Research\DATAset\s5\window_features.parquet')
print(windows['window_id'].head(10).tolist())

# %% ---- cell 4 ----
# ── RELOAD S3, S4, WINDOWS ───────────────────────────────────────────────────
print("Loading files...")
s3      = pd.read_parquet(r'<PHD_ROOT>\PHD Research\DATAset\s3\hourly_signals_with_labs.parquet')
s4      = pd.read_parquet(r'<PHD_ROOT>\PHD Research\DATAset\s4\windows.parquet')
windows = pd.read_parquet(r'<PHD_ROOT>\PHD Research\DATAset\s5\window_features.parquet')

# ── ADD HOUR_OFFSET TO FIO2 AND GCS ─────────────────────────────────────────
print("Adding hour_offset...")
icustays = pd.read_csv(r'<PHD_ROOT>\PHD Research\DATAset\mimic-iv-3.1\icu\icustays.csv',
                       usecols=['stay_id','intime'])
icustays['intime'] = pd.to_datetime(icustays['intime'])

fio2_raw['charttime'] = pd.to_datetime(fio2_raw['charttime'])
fio2_h = fio2_raw.merge(icustays, on='stay_id', how='left')
fio2_h['hour_offset'] = ((fio2_h['charttime'] - fio2_h['intime'])
                          .dt.total_seconds() / 3600).round().astype(int)
fio2_h = fio2_h.groupby(['stay_id','hour_offset'])['fio2'].mean().reset_index()

gcs_pivot['charttime'] = pd.to_datetime(gcs_pivot['charttime'])
gcs_h = gcs_pivot.merge(icustays, on='stay_id', how='left')
gcs_h['hour_offset'] = ((gcs_h['charttime'] - gcs_h['intime'])
                         .dt.total_seconds() / 3600).round().astype(int)
gcs_h = gcs_h.groupby(['stay_id','hour_offset'])[['gcs_eye','gcs_verbal','gcs_motor']].mean().reset_index()

del icustays
gc.collect()

# ── MERGE INTO S3 ────────────────────────────────────────────────────────────
print("Merging into S3...")
s3 = s3.merge(fio2_h, on=['stay_id','hour_offset'], how='left')
s3 = s3.merge(gcs_h,  on=['stay_id','hour_offset'], how='left')

s3['fio2_obs']       = s3['fio2'].notna().astype(int)
s3['gcs_eye_obs']    = s3['gcs_eye'].notna().astype(int)
s3['gcs_verbal_obs'] = s3['gcs_verbal'].notna().astype(int)
s3['gcs_motor_obs']  = s3['gcs_motor'].notna().astype(int)
s3['intubated_flag'] = (s3['gcs_verbal'].notna() & (s3['gcs_verbal'] <= 1)).astype(int)
s3['sedated_flag']   = (s3['gcs_motor'].notna()  & (s3['gcs_motor']  <= 2)).astype(int)

for sig in ['fio2','gcs_eye','gcs_verbal','gcs_motor']:
    s3[sig] = s3.groupby('stay_id')[sig].transform(lambda x: x.ffill(limit=4))

print(f"S3 extended: {s3.shape}")
gc.collect()

# ── FEATURE SETUP ────────────────────────────────────────────────────────────
SIGNALS = ['fio2', 'gcs_eye', 'gcs_verbal', 'gcs_motor']
THRESHOLDS = {
    'fio2':       (0.40, 'above'),
    'gcs_eye':    (3.0,  'below'),
    'gcs_verbal': (4.0,  'below'),
    'gcs_motor':  (5.0,  'below'),
}

def process_stay(stay_hourly, stay_windows):
    rows = []
    for win in stay_windows.itertuples(index=False):
        w = stay_hourly[
            (stay_hourly['hour_offset'] >= win.window_start) &
            (stay_hourly['hour_offset'] <= win.window_end)
        ]
        row = {'window_id': win.window_id}
        for sig in SIGNALS:
            s        = w[sig]
            obs      = s.dropna()
            n        = len(obs)
            miss     = s.isna().mean()
            thr, dir = THRESHOLDS[sig]
            if n == 0:
                for k in ['mean','min','max','std','last','slope','delta','miss_rate','cross_pct']:
                    row[f'{sig}_{k}'] = np.nan
                row[f'{sig}_miss_rate'] = miss
                continue
            slope = float(np.polyfit(np.arange(n), obs.values, 1)[0]) if n > 1 else 0.0
            cross = float((obs > thr).mean()) if dir == 'above' else float((obs < thr).mean())
            row.update({
                f'{sig}_mean':      float(obs.mean()),
                f'{sig}_min':       float(obs.min()),
                f'{sig}_max':       float(obs.max()),
                f'{sig}_std':       float(obs.std()) if n > 1 else 0.0,
                f'{sig}_last':      float(obs.iloc[-1]),
                f'{sig}_slope':     slope,
                f'{sig}_delta':     float(obs.iloc[-1] - obs.iloc[0]),
                f'{sig}_miss_rate': miss,
                f'{sig}_cross_pct': cross,
            })
        row['intubated_flag'] = int(w['intubated_flag'].any()) if len(w) > 0 else 0
        row['sedated_flag']   = int(w['sedated_flag'].any())   if len(w) > 0 else 0
        row['fio2_obs_rate']  = float(w['fio2_obs'].mean())    if len(w) > 0 else np.nan
        rows.append(row)
    return rows

# ── MEMORY-SAFE LOOP ─────────────────────────────────────────────────────────
print("\nProcessing by stay_id...")
OUT_FEATURES = r'<PHD_ROOT>\PHD Research\DATAset\s5\new_features.parquet'
FLUSH_EVERY  = 50_000

s3_grouped = s3.groupby('stay_id')
s4_grouped = s4.groupby('stay_id')
stay_ids   = s3['stay_id'].unique()

buffer = []
writer = None
total  = 0

for i, sid in enumerate(stay_ids):
    try:
        stay_hourly  = s3_grouped.get_group(sid)
        stay_windows = s4_grouped.get_group(sid)
    except KeyError:
        continue
    buffer.extend(process_stay(stay_hourly, stay_windows))
    total += 1

    if len(buffer) >= FLUSH_EVERY:
        chunk = pd.DataFrame(buffer)
        for col in chunk.select_dtypes('float64').columns:
            chunk[col] = chunk[col].astype(np.float32)
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUT_FEATURES, table.schema)
        writer.write_table(table)
        buffer.clear()
        gc.collect()
        print(f"  Stays: {i+1:,} / {len(stay_ids):,} | Rows flushed: {total:,}")

if buffer:
    chunk = pd.DataFrame(buffer)
    for col in chunk.select_dtypes('float64').columns:
        chunk[col] = chunk[col].astype(np.float32)
    table = pa.Table.from_pandas(chunk, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(OUT_FEATURES, table.schema)
    writer.write_table(table)
    buffer.clear()

if writer:
    writer.close()

print(f"\nTotal stays processed: {total:,}")
gc.collect()

# ── MERGE WITH EXISTING WINDOWS ──────────────────────────────────────────────
print("Merging new features into window_features...")
new_features_df = pd.read_parquet(OUT_FEATURES)
new_features_df = new_features_df.sort_values('window_id').reset_index(drop=True)
windows_sorted  = windows.sort_values('window_id').reset_index(drop=True)

assert (windows_sorted['window_id'].values == new_features_df['window_id'].values).all(), \
    "window_id mismatch!"

for col in [c for c in new_features_df.columns if c != 'window_id']:
    windows_sorted[col] = new_features_df[col].values

del new_features_df
gc.collect()

# ── SAVE ─────────────────────────────────────────────────────────────────────
OUT_FINAL = r'<PHD_ROOT>\PHD Research\DATAset\s5\window_features_extended.parquet'
windows_sorted.to_parquet(OUT_FINAL, index=False)
print(f"Saved -> {OUT_FINAL}")
print(f"Final shape: {windows_sorted.shape}")
print(f"Features added: {windows_sorted.shape[1] - 188}")

# ── SANITY CHECK ─────────────────────────────────────────────────────────────
new_cols = [c for c in windows_sorted.columns
            if c.startswith(('fio2','gcs_eye','gcs_verbal','gcs_motor','intubated','sedated'))]
for c in new_cols:
    pct = windows_sorted[c].isna().mean() * 100
    print(f"  {c:<30} null: {pct:.1f}%")

print(f"\nLabel distribution:")
print(windows_sorted['label'].value_counts())
print("\nDone ✓")

# %% ---- cell 5 ----

