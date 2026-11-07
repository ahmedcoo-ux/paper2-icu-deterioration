# Source: S7.ipynb, code cells [0, 1, 2] (outputs not included)

# %% ---- cell 0 ----
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import pyarrow.parquet as pq
import pickle, gc, os

OUT_DIR = r'<PHD_ROOT>\PHD Research\DATAset\s7'
os.makedirs(OUT_DIR, exist_ok=True)

NON_FEATURE_COLS = {'window_id', 'stay_id', 'subject_id', 'label', 'news2_max_horizon'}
PARQUET = r'<PHD_ROOT>\PHD Research\DATAset\s5\window_features_extended.parquet'

# ── STEP 1: Load index columns only ──────────────────────────────────────────
print("Loading index columns only...")
index_df = pd.read_parquet(PARQUET, columns=['subject_id', 'label'])
y      = index_df['label'].values.astype(np.int8)
groups = index_df['subject_id'].values
n_total = len(index_df)
del index_df; gc.collect()
print(f"Total rows: {n_total:,}")

# ── STEP 2: Get feature column names ─────────────────────────────────────────
pf_schema = pq.read_schema(PARQUET)
all_cols  = [f.name for f in pf_schema]
feature_cols = [c for c in all_cols if c not in NON_FEATURE_COLS]
print(f"Feature columns: {len(feature_cols)}")

with open(os.path.join(OUT_DIR, 'feature_cols.txt'), 'w') as f:
    f.write('\n'.join(feature_cols))

# ── STEP 3: Compute splits ────────────────────────────────────────────────────
gkf    = GroupKFold(n_splits=5)
splits = list(gkf.split(np.arange(n_total), y, groups))
print("Splits computed.")

# ── STEP 4: Per-fold processing ───────────────────────────────────────────────
CHUNK     = 200_000   # rows processed at a time
N_FEATS   = len(feature_cols)

for fold, (train_idx, val_idx) in enumerate(splits):
    print(f"\nFold {fold+1}/5 ──────────────────────────────")

    fold_dir = os.path.join(OUT_DIR, f'fold_{fold+1}')
    os.makedirs(fold_dir, exist_ok=True)

    y_train = y[train_idx]
    y_val   = y[val_idx]

    # Sort indices — required for row-group efficient parquet reading
    train_idx_sorted = np.sort(train_idx)
    val_idx_sorted   = np.sort(val_idx)

    # ── Fit scaler on 100k sample from train ─────────────────────────────────
    print("  Fitting scaler on sample...")
    sample_pos = np.random.choice(len(train_idx_sorted), size=100_000, replace=False)
    sample_row_ids = train_idx_sorted[np.sort(sample_pos)]

    pf     = pq.ParquetFile(PARQUET)
    sample_chunks = []
    rows_collected = 0
    row_offset = 0

    for batch in pf.iter_batches(batch_size=CHUNK, columns=feature_cols):
        batch_len = len(batch)
        batch_rows = np.arange(row_offset, row_offset + batch_len)
        mask = np.isin(batch_rows, sample_row_ids)
        if mask.any():
            sample_chunks.append(batch.to_pandas()[mask].values.astype(np.float32))
            rows_collected += mask.sum()
        row_offset += batch_len
        if rows_collected >= 100_000:
            break

    sample_data = np.vstack(sample_chunks).astype(np.float64)
    del sample_chunks; gc.collect()

    scaler = StandardScaler()
    scaler.fit(sample_data)
    del sample_data; gc.collect()

    mean_  = scaler.mean_.astype(np.float32)
    scale_ = scaler.scale_.astype(np.float32)

    with open(os.path.join(fold_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # ── Write train memmap ────────────────────────────────────────────────────
    print("  Writing X_train...")
    n_train  = len(train_idx_sorted)
    fp_train = np.lib.format.open_memmap(
        os.path.join(fold_dir, 'X_train.npy'),
        mode='w+', dtype=np.float32, shape=(n_train, N_FEATS))

    train_write_pos = 0
    row_offset = 0
    train_idx_set = set(train_idx_sorted.tolist())

    for batch in pq.ParquetFile(PARQUET).iter_batches(batch_size=CHUNK, columns=feature_cols):
        batch_len  = len(batch)
        batch_rows = np.arange(row_offset, row_offset + batch_len)
        mask = np.array([r in train_idx_set for r in batch_rows], dtype=bool)
        if mask.any():
            chunk = batch.to_pandas()[mask].values.astype(np.float32)
            chunk = (chunk - mean_) / scale_
            n = chunk.shape[0]
            fp_train[train_write_pos:train_write_pos + n] = chunk
            train_write_pos += n
        row_offset += batch_len

    del fp_train; gc.collect()
    print(f"  X_train written: {train_write_pos:,} rows")

    # ── Write val memmap ──────────────────────────────────────────────────────
    print("  Writing X_val...")
    n_val  = len(val_idx_sorted)
    fp_val = np.lib.format.open_memmap(
        os.path.join(fold_dir, 'X_val.npy'),
        mode='w+', dtype=np.float32, shape=(n_val, N_FEATS))

    val_write_pos = 0
    row_offset    = 0
    val_idx_set   = set(val_idx_sorted.tolist())

    for batch in pq.ParquetFile(PARQUET).iter_batches(batch_size=CHUNK, columns=feature_cols):
        batch_len  = len(batch)
        batch_rows = np.arange(row_offset, row_offset + batch_len)
        mask = np.array([r in val_idx_set for r in batch_rows], dtype=bool)
        if mask.any():
            chunk = batch.to_pandas()[mask].values.astype(np.float32)
            chunk = (chunk - mean_) / scale_
            n = chunk.shape[0]
            fp_val[val_write_pos:val_write_pos + n] = chunk
            val_write_pos += n
        row_offset += batch_len

    del fp_val; gc.collect()
    print(f"  X_val written: {val_write_pos:,} rows")

    # ── Save labels ───────────────────────────────────────────────────────────
    np.save(os.path.join(fold_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(fold_dir, 'y_val.npy'),   y_val)

    print(f"  Train: {n_train:,}  pos={y_train.mean()*100:.1f}%")
    print(f"  Val  : {n_val:,}   pos={y_val.mean()*100:.1f}%")
    print(f"  Saved -> {fold_dir}")
    gc.collect()

print(f"\nS7 Done ✓")

# %% ---- cell 1 ----
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle, os, time

FOLD_DIR = r'<PHD_ROOT>\PHD Research\DATAset\s7\fold_1'
S7_DIR   = r'<PHD_ROOT>\PHD Research\DATAset\s7'
OUT_DIR  = r'<PHD_ROOT>\PHD Research\DATAset\s7b'
os.makedirs(OUT_DIR, exist_ok=True)

# ── LOAD ─────────────────────────────────────────────────────────────────────
print("Loading fold 1...")
X_train = np.load(os.path.join(FOLD_DIR, 'X_train.npy'), mmap_mode='r')
X_val   = np.load(os.path.join(FOLD_DIR, 'X_val.npy'),   mmap_mode='r')
y_train = np.load(os.path.join(FOLD_DIR, 'y_train.npy'))
y_val   = np.load(os.path.join(FOLD_DIR, 'y_val.npy'))

with open(os.path.join(S7_DIR, 'feature_cols.txt')) as f:
    feature_cols = [l.strip() for l in f.readlines()]

print(f"Train: {X_train.shape}  Val: {X_val.shape}")

# ── FEATURE GROUPS — matched exactly to your column names ────────────────────
FEATURE_GROUPS = {
    'Heart Rate':                ['hr_'],
    'Blood Pressure (SBP/DBP)':  ['sbp_', 'dbp_'],
    'MAP':                       ['map_'],
    'Respiratory Rate':          ['rr_'],
    'SpO2':                      ['spo2_'],
    'Temperature':               ['temp_'],
    'Urine Output':              ['urine_output_'],
    'FiO2':                      ['fio2_'],
    'GCS Total':                 ['gcs_mean', 'gcs_last', 'gcs_obs_rate'],
    'GCS Eye':                   ['gcs_eye_'],
    'GCS Verbal':                ['gcs_verbal_'],
    'GCS Motor':                 ['gcs_motor_'],
    'AVPU':                      ['avpu_sum'],
    'Intubation/Sedation':       ['intubated_flag', 'sedated_flag'],
    'Lactate':                   ['lactate_'],
    'Blood Gas (pH/pCO2/pO2)':   ['ph_', 'pco2_', 'po2_'],
    'Renal (Creatinine/BUN)':    ['creatinine_', 'bun_'],
    'Electrolytes (K/Na)':       ['potassium_', 'sodium_'],
    'Glucose':                   ['glucose_'],
    'Haematology (WBC/Hgb/Hct/Plt)': ['wbc_', 'hemoglobin_', 'hematocrit_', 'platelets_'],
    'Hepatic (Bili/Albumin)':    ['bilirubin_', 'albumin_'],
    'Bicarbonate':               ['bicarbonate_'],
    'Lab Flags':                 ['lactate_flag', 'creatinine_flag', 'potassium_flag',
                                  'wbc_flag', 'glucose_flag', 'bilirubin_flag'],
    'Missingness (hrs_since)':   ['_hrs_since'],
    'Missingness (miss_rate)':   ['_miss_rate'],
    'Trend (slope)':             ['_slope'],
    'Trend (delta)':             ['_delta'],
    'Threshold Crossing':        ['_cross_pct'],
    'Abnormal % flags':          ['_pct_below', '_pct_abnormal'],
    'Age':                       ['age'],
}

def get_col_indices(patterns, feature_cols):
    indices = []
    for i, col in enumerate(feature_cols):
        for pat in patterns:
            if pat in col:
                indices.append(i)
                break
    return indices

# ── TRAIN BASELINE ────────────────────────────────────────────────────────────
print("\nTraining baseline LightGBM...")
t0 = time.time()

X_tr = np.array(X_train)
X_vl = np.array(X_val)

params = dict(
    objective         = 'binary',
    metric            = 'auc',
    n_estimators      = 500,
    learning_rate     = 0.05,
    num_leaves        = 63,
    min_child_samples = 50,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    n_jobs            = -1,
    random_state      = 42,
    verbose           = -1,
)

model = lgb.LGBMClassifier(**params)
model.fit(
    X_tr, y_train,
    eval_set=[(X_vl, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False),
               lgb.log_evaluation(100)]
)

baseline_pred  = model.predict_proba(X_vl)[:, 1]
baseline_auroc = roc_auc_score(y_val, baseline_pred)
baseline_auprc = average_precision_score(y_val, baseline_pred)
print(f"Baseline AUROC={baseline_auroc:.4f}  AUPRC={baseline_auprc:.4f}  "
      f"({time.time()-t0:.0f}s)")

with open(os.path.join(OUT_DIR, 'baseline_model_fold1.pkl'), 'wb') as f:
    pickle.dump(model, f, protocol=5)
print(f"Model saved -> {OUT_DIR}")

# ── ABLATION ─────────────────────────────────────────────────────────────────
print("\nRunning ablation...")
results = []

for group_name, patterns in FEATURE_GROUPS.items():
    col_ids = get_col_indices(patterns, feature_cols)
    if not col_ids:
        print(f"  SKIP (no cols): {group_name}")
        continue

    X_abl = X_vl.copy()
    X_abl[:, col_ids] = 0.0

    pred_abl  = model.predict_proba(X_abl)[:, 1]
    auroc_abl = roc_auc_score(y_val, pred_abl)
    auprc_abl = average_precision_score(y_val, pred_abl)

    d_auroc = baseline_auroc - auroc_abl
    d_auprc = baseline_auprc - auprc_abl

    results.append({
        'feature_group':  group_name,
        'n_cols':         len(col_ids),
        'auroc_baseline': round(baseline_auroc, 4),
        'auroc_ablated':  round(auroc_abl,      4),
        'delta_auroc':    round(d_auroc,         4),
        'auprc_baseline': round(baseline_auprc, 4),
        'auprc_ablated':  round(auprc_abl,      4),
        'delta_auprc':    round(d_auprc,         4),
    })
    print(f"  {group_name:<35}  n={len(col_ids):>3}  "
          f"ΔAUROC={d_auroc:+.4f}  ΔAUPRC={d_auprc:+.4f}")

# ── SAVE CSV ──────────────────────────────────────────────────────────────────
res_df   = pd.DataFrame(results).sort_values('delta_auroc', ascending=False)
csv_path = os.path.join(OUT_DIR, 'ablation_results.csv')
res_df.to_csv(csv_path, index=False)
print(f"\nSaved -> {csv_path}")
print(res_df[['feature_group','n_cols','delta_auroc','delta_auprc']].to_string(index=False))

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, max(6, len(res_df) * 0.45)))
colors = ['#d62728' if d > 0 else '#1f77b4' for d in res_df['delta_auroc']]
bars   = ax.barh(res_df['feature_group'], res_df['delta_auroc'], color=colors)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('ΔAUROC  (positive = group contributes, negative = group hurts)', fontsize=11)
ax.set_title(f'S7b Feature Group Ablation  |  Baseline AUROC={baseline_auroc:.4f}  '
             f'AUPRC={baseline_auprc:.4f}', fontsize=12)
for bar, val in zip(bars, res_df['delta_auroc']):
    ax.text(val + 0.0003 if val >= 0 else val - 0.0003,
            bar.get_y() + bar.get_height() / 2,
            f'{val:+.4f}', va='center',
            ha='left' if val >= 0 else 'right', fontsize=8)
plt.tight_layout()
plot_path = os.path.join(OUT_DIR, 'ablation_plot.png')
plt.savefig(plot_path, dpi=150)
print(f"Plot saved -> {plot_path}")
print("\nS7b Done ✓")

# %% ---- cell 2 ----

