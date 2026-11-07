# Source: S5 patch.ipynb, code cells [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (outputs not included)

# %% ---- cell 0 ----
import pandas as pd
import numpy as np

print("Loading files...")
s3      = pd.read_parquet(r'<PHD_ROOT>\PHD Research\DATAset\s3\hourly_signals_with_labs.parquet')
s4      = pd.read_parquet(r'<PHD_ROOT>\PHD Research\DATAset\s4\windows.parquet')
windows = pd.read_parquet(r'<PHD_ROOT>\PHD Research\DATAset\s5\window_features.parquet')

# ─── MERGE NEW SIGNALS INTO S3 ───────────────────────────────────────────
print("Merging FiO2 and GCS into S3...")
s3 = s3.merge(fio2_raw,  on=['stay_id','hour_offset'], how='left')
s3 = s3.merge(gcs_pivot, on=['stay_id','hour_offset'], how='left')

s3['fio2_obs']       = s3['fio2'].notna().astype(int)
s3['gcs_eye_obs']    = s3['gcs_eye'].notna().astype(int)
s3['gcs_verbal_obs'] = s3['gcs_verbal'].notna().astype(int)
s3['gcs_motor_obs']  = s3['gcs_motor'].notna().astype(int)
s3['intubated_flag'] = (s3['gcs_verbal'].notna() & (s3['gcs_verbal'] <= 1)).astype(int)
s3['sedated_flag']   = (s3['gcs_motor'].notna()  & (s3['gcs_motor']  <= 2)).astype(int)

for sig in ['fio2', 'gcs_eye', 'gcs_verbal', 'gcs_motor']:
    s3[sig] = s3.groupby('stay_id')[sig].transform(lambda x: x.ffill(limit=4))

print(f"S3 extended: {s3.shape}")

# ─── SETUP ───────────────────────────────────────────────────────────────
SIGNALS = ['fio2', 'gcs_eye', 'gcs_verbal', 'gcs_motor']
THRESHOLDS = {
    'fio2':       (0.40, 'above'),
    'gcs_eye':    (3.0,  'below'),
    'gcs_verbal': (4.0,  'below'),
    'gcs_motor':  (5.0,  'below'),
}

def process_stay(stay_hourly, stay_windows):
    """Process all windows for one stay — fully vectorized within stay."""
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

# ─── BATCH LOOP BY STAY_ID ───────────────────────────────────────────────
print("\nProcessing by stay_id in batches...")

s3_grouped  = s3.groupby('stay_id')
s4_grouped  = s4.groupby('stay_id')
stay_ids    = s3['stay_id'].unique()

BATCH_SIZE  = 5000
all_rows    = []
total_stays = len(stay_ids)

for batch_start in range(0, total_stays, BATCH_SIZE):
    batch_ids = stay_ids[batch_start : batch_start + BATCH_SIZE]

    for sid in batch_ids:
        try:
            stay_hourly  = s3_grouped.get_group(sid)
            stay_windows = s4_grouped.get_group(sid)
        except KeyError:
            continue
        all_rows.extend(process_stay(stay_hourly, stay_windows))

    pct = min(batch_start + BATCH_SIZE, total_stays)
    print(f"  Stays processed: {pct:,} / {total_stays:,}  "
          f"| Rows so far: {len(all_rows):,}")

print(f"\nTotal rows computed: {len(all_rows):,}")

# ─── MERGE AND SAVE ───────────────────────────────────────────────────────
print("Merging into windows...")
new_features_df  = pd.DataFrame(all_rows)
windows_extended = windows.merge(new_features_df, on='window_id', how='left')

print(f"Final shape    : {windows_extended.shape}")
print(f"Features added : {windows_extended.shape[1] - 188}")

out_path = r'<PHD_ROOT>\PHD Research\DATAset\s5\window_features_extended.parquet'
windows_extended.to_parquet(out_path, index=False)
print(f"Saved -> {out_path}")

# ─── SANITY CHECK ─────────────────────────────────────────────────────────
print("\n── Sanity Check ──────────────────────────────────────────────────")
new_cols = [c for c in windows_extended.columns
            if c.startswith(('fio2','gcs_eye','gcs_verbal',
                             'gcs_motor','intubated','sedated'))]
for c in new_cols:
    pct = windows_extended[c].isna().mean() * 100
    print(f"  {c:<30} null: {pct:.1f}%")
print(f"\nLabel distribution:")
print(windows_extended['label'].value_counts())
print("\nDone ✓")

# %% ---- cell 1 ----
# ─── REPLACE JUST THE MERGE AND SAVE SECTION ─────────────────────────────
# The all_rows computation is done — do NOT re-run the loop

print("Building new features dataframe...")
new_features_df = pd.DataFrame(all_rows)
print(f"New features shape: {new_features_df.shape}")

# Sort both by window_id to ensure alignment
print("Sorting by window_id...")
new_features_df = new_features_df.sort_values('window_id').reset_index(drop=True)
windows_sorted  = windows.sort_values('window_id').reset_index(drop=True)

# Verify alignment before joining
assert (windows_sorted['window_id'].values == new_features_df['window_id'].values).all(), \
    "window_id mismatch — check sorting"

# Drop window_id from new features (already in windows)
new_cols_only = new_features_df.drop(columns=['window_id'])

# Assign new columns directly — no merge, no memory duplication
print("Assigning new columns...")
for col in new_cols_only.columns:
    windows_sorted[col] = new_cols_only[col].values

print(f"Final shape    : {windows_sorted.shape}")
print(f"Features added : {windows_sorted.shape[1] - 188}")

# Save
out_path = r'<PHD_ROOT>\PHD Research\DATAset\s5\window_features_extended.parquet'
windows_sorted.to_parquet(out_path, index=False)
print(f"Saved -> {out_path}")

# Sanity check
print("\n── Sanity Check ──────────────────────────────────────────────────")
new_cols = [c for c in windows_sorted.columns
            if c.startswith(('fio2','gcs_eye','gcs_verbal',
                             'gcs_motor','intubated','sedated'))]
for c in new_cols:
    pct = windows_sorted[c].isna().mean() * 100
    print(f"  {c:<30} null: {pct:.1f}%")

print(f"\nLabel distribution:")
print(windows_sorted['label'].value_counts())
print("\nDone ✓")

# %% ---- cell 2 ----
import pandas as pd
import numpy as np
import os, gc

# ── STEP 1: Stream all_rows to parquet in chunks instead of holding in RAM ──
# If all_rows is already computed as a list, do this:

CHUNK = 500_000
out_dir = r'<PHD_ROOT>\PHD Research\DATAset\s5\chunks'
os.makedirs(out_dir, exist_ok=True)

for i in range(0, len(all_rows), CHUNK):
    chunk_df = pd.DataFrame(all_rows[i:i+CHUNK])
    # Downcast floats to float32 to cut memory in half
    for col in chunk_df.select_dtypes('float64').columns:
        chunk_df[col] = chunk_df[col].astype(np.float32)
    chunk_df.to_parquet(f'{out_dir}/chunk_{i//CHUNK:04d}.parquet', index=False)
    print(f"Saved chunk {i//CHUNK}")

del all_rows
gc.collect()

# ── STEP 2: Read chunks back and merge with windows in pieces ──
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq

# Load new features lazily via pyarrow
new_feat_dataset = ds.dataset(out_dir, format='parquet')

# Read windows and sort
print("Loading windows...")
windows_sorted = windows.sort_values('window_id').reset_index(drop=True)

# Downcast windows floats too
for col in windows_sorted.select_dtypes('float64').columns:
    windows_sorted[col] = windows_sorted[col].astype(np.float32)

# Read new features in one shot via pyarrow (more memory-efficient than pd.read_parquet)
print("Reading new features via pyarrow...")
new_features_df = new_feat_dataset.to_table().to_pandas()
new_features_df = new_features_df.sort_values('window_id').reset_index(drop=True)

# Verify alignment
assert (windows_sorted['window_id'].values == new_features_df['window_id'].values).all()

# Assign column-by-column (avoids merge copy overhead)
new_cols = [c for c in new_features_df.columns if c != 'window_id']
for col in new_cols:
    windows_sorted[col] = new_features_df[col].values

del new_features_df
gc.collect()

# ── STEP 3: Save ──
out_path = r'<PHD_ROOT>\PHD Research\DATAset\s5\window_features_extended.parquet'
windows_sorted.to_parquet(out_path, index=False, engine='pyarrow')
print(f"Saved -> {out_path}  shape={windows_sorted.shape}")

# %% ---- cell 3 ----
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import gc, os

# ── CONFIG ──────────────────────────────────────────────────────────────────
OUT_FEATURES = r'<PHD_ROOT>\PHD Research\DATAset\s5\new_features.parquet'
FLUSH_EVERY  = 50_000   # rows buffered before flushing to disk

# ── LOOP ────────────────────────────────────────────────────────────────────
buffer  = []
writer  = None
total   = 0

stay_ids = s3['stay_id'].unique()   # adjust to match your original loop source

for i, stay_id in enumerate(stay_ids):

    # ── your existing per-stay computation goes here (unchanged) ──
    # e.g.:
    # stay_data = s3[s3['stay_id'] == stay_id]
    # row_dict  = compute_features(stay_data)   # whatever you did before
    # ─────────────────────────────────────────────────────────────
    
    buffer.append(row_dict)
    total += 1

    # Flush to disk every FLUSH_EVERY rows
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
        print(f"  Stays processed: {i+1:,} / {len(stay_ids):,}  | Rows flushed: {total:,}")

# Flush remainder
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

print(f"\nTotal rows written: {total:,}")
gc.collect()

# ── MERGE WITH WINDOWS ───────────────────────────────────────────────────────
print("Loading new features...")
new_features_df = pd.read_parquet(OUT_FEATURES)
new_features_df = new_features_df.sort_values('window_id').reset_index(drop=True)

print("Loading & sorting windows...")
windows_sorted = windows.sort_values('window_id').reset_index(drop=True)

# Downcast windows floats too
for col in windows_sorted.select_dtypes('float64').columns:
    windows_sorted[col] = windows_sorted[col].astype(np.float32)

# Verify alignment
assert (windows_sorted['window_id'].values == new_features_df['window_id'].values).all(), \
    "window_id mismatch!"

# Assign column-by-column — no merge, no internal copy
print("Assigning new columns...")
for col in [c for c in new_features_df.columns if c != 'window_id']:
    windows_sorted[col] = new_features_df[col].values

del new_features_df
gc.collect()

# ── SAVE FINAL ───────────────────────────────────────────────────────────────
OUT_FINAL = r'<PHD_ROOT>\PHD Research\DATAset\s5\window_features_extended.parquet'
windows_sorted.to_parquet(OUT_FINAL, index=False)
print(f"Saved -> {OUT_FINAL}")
print(f"Final shape: {windows_sorted.shape}")

# %% ---- cell 4 ----
import os

for root, dirs, files in os.walk(r'<PHD_ROOT>\PHD Research\DATAset'):
    dirs[:] = [d for d in dirs if d not in ['chunks']]
    for f in files:
        if any(x in f.lower() for x in ['fio2', 'gcs', 'chart', 'icu', 'ce']):
            full = os.path.join(root, f)
            size_mb = os.path.getsize(full) / 1024**2
            print(f"{size_mb:8.1f} MB   {full}")

# %% ---- cell 5 ----
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import gc, os

# ── ITEM IDS (MIMIC-IV) ──────────────────────────────────────────────────────
# FiO2: 223835
# GCS Eye: 220739, GCS Verbal: 223900, GCS Motor: 223901
FIO2_ITEMS = {223835}
GCS_ITEMS  = {220739: 'gcs_eye', 223900: 'gcs_verbal', 223901: 'gcs_motor'}
ALL_ITEMS  = FIO2_ITEMS | set(GCS_ITEMS.keys())

CHARTEVENTS = r'<PHD_ROOT>\PHD Research\DATAset\mimic-iv-3.1\icu\chartevents.csv'
CHUNK_SIZE  = 500_000

print("Extracting FiO2 and GCS from chartevents.csv (this will take a few minutes)...")

fio2_rows = []
gcs_rows  = []
chunks_read = 0

for chunk in pd.read_csv(CHARTEVENTS, usecols=['stay_id','charttime','itemid','valuenum'],
                         chunksize=CHUNK_SIZE, low_memory=False):
    chunk = chunk[chunk['itemid'].isin(ALL_ITEMS)].dropna(subset=['valuenum','stay_id'])
    
    fio2_chunk = chunk[chunk['itemid'] == 223835][['stay_id','charttime','valuenum']].copy()
    fio2_chunk.rename(columns={'valuenum':'fio2'}, inplace=True)
    if len(fio2_chunk): fio2_rows.append(fio2_chunk)

    gcs_chunk = chunk[chunk['itemid'].isin(GCS_ITEMS)][['stay_id','charttime','itemid','valuenum']].copy()
    gcs_chunk['component'] = gcs_chunk['itemid'].map(GCS_ITEMS)
    gcs_chunk.drop(columns='itemid', inplace=True)
    if len(gcs_chunk): gcs_rows.append(gcs_chunk)

    chunks_read += 1
    if chunks_read % 20 == 0:
        print(f"  Chunks read: {chunks_read:,}  (~{chunks_read*CHUNK_SIZE/1e6:.0f}M rows scanned)")

print("Assembling fio2_raw and gcs_pivot...")

# ── FIO2 ─────────────────────────────────────────────────────────────────────
fio2_raw = pd.concat(fio2_rows, ignore_index=True)
fio2_raw['charttime'] = pd.to_datetime(fio2_raw['charttime'])

# Scale FiO2 to 0-1 if stored as percentage
fio2_raw.loc[fio2_raw['fio2'] > 1, 'fio2'] /= 100

del fio2_rows; gc.collect()

# ── GCS ──────────────────────────────────────────────────────────────────────
gcs_raw = pd.concat(gcs_rows, ignore_index=True)
gcs_raw['charttime'] = pd.to_datetime(gcs_raw['charttime'])
gcs_pivot = gcs_raw.pivot_table(index=['stay_id','charttime'],
                                 columns='component', values='valuenum',
                                 aggfunc='first').reset_index()
gcs_pivot.columns.name = None

del gcs_rows, gcs_raw; gc.collect()

print(f"fio2_raw shape : {fio2_raw.shape}")
print(f"gcs_pivot shape: {gcs_pivot.shape}")
print("Extraction done ✓")

# %% ---- cell 6 ----
import pandas as pd

windows = pd.read_parquet(r'<PHD_ROOT>\PHD Research\DATAset\s5\window_features.parquet')
print(f"Shape: {windows.shape}")
print(f"\nColumns ({len(windows.columns)}):")
for c in windows.columns:
    print(" ", c)

# %% ---- cell 7 ----
# Confirm they are still in memory
print("fio2_raw" in dir(), "gcs_pivot" in dir())
print(fio2_raw.shape)
print(gcs_pivot.shape)

# %% ---- cell 8 ----
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

# %% ---- cell 9 ----

