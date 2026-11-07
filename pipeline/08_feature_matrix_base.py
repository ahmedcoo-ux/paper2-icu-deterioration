# Source: LGB AUROC 0.8064 retrained on new s7 features.ipynb, code cells [0, 4] (outputs not included)

# %% ---- cell 0 ----
import warnings, numpy as np, pandas as pd, pyarrow.parquet as pq
import pickle, time, gc
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
np.seterr(all='ignore')

S3_PATH = Path(r"<PHD_ROOT>\PHD Research\DATAset\s3\hourly_signals_with_labs.parquet")
S4_PATH = Path(r"<PHD_ROOT>\PHD Research\DATAset\s4\windows.parquet")
S7_DIR  = Path(r"<PHD_ROOT>\PHD Research\DATAset\s7_new")
OUT_DIR = Path(r"<PHD_ROOT>\s7_rebuilt")
OUT_DIR.mkdir(exist_ok=True)

VITALS = ['hr','sbp','dbp','map','rr','spo2','temp','urine_output']
LABS   = ['lactate','ph','pco2','po2','creatinine','bun','potassium',
          'sodium','glucose','wbc','hemoglobin','hematocrit',
          'platelets','bilirubin','bicarbonate','albumin']
NEWS2  = {'hr':(51,90),'sbp':(111,219),'rr':(12,20),
          'spo2':(96,100),'temp':(36.1,38.0)}

with open(S7_DIR/"feature_cols.txt") as f:
    feature_cols = [l.strip() for l in f.readlines()]
N_FEAT = len(feature_cols)
feat_idx = {c:i for i,c in enumerate(feature_cols)}

def slope_vec(mat):
    x=np.arange(mat.shape[1],dtype=np.float32)
    valid=~np.isnan(mat); n_v=valid.sum(axis=1).astype(np.float32)
    m=np.where(valid,mat,0.0)
    sx=(x*valid).sum(axis=1); sy=m.sum(axis=1)
    sx2=(x**2*valid).sum(axis=1); sxy=(x*m).sum(axis=1)
    denom=n_v*sx2-sx**2
    return np.where(n_v<2,0.0,np.where(denom!=0,(n_v*sxy-sx*sy)/denom,0.0)).astype(np.float32)

def last_vec(mat):
    has_v=~np.all(np.isnan(mat),axis=1)
    rev=mat[:,::-1]; idx=np.argmax(~np.isnan(rev),axis=1)
    return np.where(has_v,rev[np.arange(mat.shape[0]),idx],np.nan).astype(np.float32)

def delta_vec(mat):
    has_v=~np.all(np.isnan(mat),axis=1); n=mat.shape[0]
    idx_f=np.argmax(~np.isnan(mat),axis=1)
    idx_l=mat.shape[1]-1-np.argmax(~np.isnan(mat[:,::-1]),axis=1)
    return np.where(has_v,mat[np.arange(n),idx_l]-mat[np.arange(n),idx_f],0.0).astype(np.float32)

def get_mat(s3_df,col,mask):
    raw=s3_df[col].values.astype(np.float32) if col in s3_df.columns \
        else np.full(mask.shape[1],np.nan,dtype=np.float32)
    return np.where(mask,raw[np.newaxis,:],np.nan)

def set_feat(out, s4_rows, col, vals):
    """Write vals array into correct column of output matrix."""
    if col in feat_idx:
        out[s4_rows, feat_idx[col]] = vals

def process_stay_into(s3_df, stay_wins, age, out):
    """Write features directly into pre-allocated output array."""
    s4_rows  = stay_wins['s4_row'].values
    w_starts = stay_wins['window_start'].values
    w_ends   = stay_wins['window_end'].values
    n_wins   = len(s4_rows)

    # Static features
    out[s4_rows, feat_idx['age']]            = age
    out[s4_rows, feat_idx['intubated_flag']] = 0
    out[s4_rows, feat_idx['sedated_flag']]   = 0
    out[s4_rows, feat_idx['fio2_obs_rate']]  = 0.0
    for k in ['fio2_mean','fio2_min','fio2_max','fio2_std','fio2_last',
              'fio2_slope','fio2_delta','fio2_miss_rate','fio2_cross_pct']:
        out[s4_rows, feat_idx[k]] = np.nan
    for sig in ['gcs_eye','gcs_verbal','gcs_motor']:
        for k in ['mean','min','max','std','last','slope','delta','miss_rate','cross_pct']:
            out[s4_rows, feat_idx[f'{sig}_{k}']] = np.nan

    all_hours = s3_df.index.values if len(s3_df) else np.array([],dtype=np.int64)

    if len(all_hours) == 0:
        for sig in VITALS:
            for k in ['mean','min','max','std','last','slope','delta','miss_rate','cross_pct']:
                out[s4_rows, feat_idx[f'{sig}_{k}']] = np.nan
        for k in ['spo2_pct_below94','rr_pct_abnormal','sbp_pct_below100',
                  'hr_pct_abnormal','temp_pct_abnormal']:
            out[s4_rows, feat_idx[k]] = 0.0
        for sig in LABS:
            for k in ['mean','last','slope','delta','miss_rate','hrs_since']:
                out[s4_rows, feat_idx[f'{sig}_{k}']] = np.nan
        for sig in ['lactate','creatinine','potassium','wbc','glucose','bilirubin']:
            out[s4_rows, feat_idx[f'{sig}_flag']] = 0
        out[s4_rows, feat_idx['gcs_mean']]     = np.nan
        out[s4_rows, feat_idx['gcs_last']]     = np.nan
        out[s4_rows, feat_idx['gcs_obs_rate']] = 0.0
        out[s4_rows, feat_idx['avpu_sum']]     = 0.0
        return

    mask = ((all_hours[np.newaxis,:] >= w_starts[:,np.newaxis]) &
            (all_hours[np.newaxis,:] <  w_ends[:,np.newaxis]))

    for sig in VITALS:
        mat=get_mat(s3_df,sig,mask)
        lo,hi=NEWS2.get(sig,(None,None))
        out[s4_rows,feat_idx[f'{sig}_mean']]      = np.nanmean(mat,axis=1)
        out[s4_rows,feat_idx[f'{sig}_min']]       = np.nanmin(mat,axis=1)
        out[s4_rows,feat_idx[f'{sig}_max']]       = np.nanmax(mat,axis=1)
        out[s4_rows,feat_idx[f'{sig}_std']]       = np.nanstd(mat,axis=1)
        out[s4_rows,feat_idx[f'{sig}_last']]      = last_vec(mat)
        out[s4_rows,feat_idx[f'{sig}_slope']]     = slope_vec(mat)
        out[s4_rows,feat_idx[f'{sig}_delta']]     = delta_vec(mat)
        out[s4_rows,feat_idx[f'{sig}_miss_rate']] = np.mean(np.isnan(mat),axis=1)
        out[s4_rows,feat_idx[f'{sig}_cross_pct']] = \
            np.nanmean((mat<lo)|(mat>hi),axis=1) if lo else np.zeros(n_wins)

    out[s4_rows,feat_idx['spo2_pct_below94']]  = np.nanmean(get_mat(s3_df,'spo2',mask)<94,axis=1)
    out[s4_rows,feat_idx['rr_pct_abnormal']]   = np.nanmean((get_mat(s3_df,'rr',mask)<12)|(get_mat(s3_df,'rr',mask)>25),axis=1)
    out[s4_rows,feat_idx['sbp_pct_below100']]  = np.nanmean(get_mat(s3_df,'sbp',mask)<100,axis=1)
    out[s4_rows,feat_idx['hr_pct_abnormal']]   = np.nanmean((get_mat(s3_df,'hr',mask)<51)|(get_mat(s3_df,'hr',mask)>110),axis=1)
    out[s4_rows,feat_idx['temp_pct_abnormal']] = np.nanmean((get_mat(s3_df,'temp',mask)<36.1)|(get_mat(s3_df,'temp',mask)>38.0),axis=1)

    for sig in LABS:
        mat=get_mat(s3_df,sig,mask); hs=get_mat(s3_df,f'{sig}_hrs_since',mask)
        out[s4_rows,feat_idx[f'{sig}_mean']]      = np.nanmean(mat,axis=1)
        out[s4_rows,feat_idx[f'{sig}_last']]      = last_vec(mat)
        out[s4_rows,feat_idx[f'{sig}_slope']]     = slope_vec(mat)
        out[s4_rows,feat_idx[f'{sig}_delta']]     = delta_vec(mat)
        out[s4_rows,feat_idx[f'{sig}_miss_rate']] = np.mean(np.isnan(mat),axis=1)
        out[s4_rows,feat_idx[f'{sig}_hrs_since']] = last_vec(hs)

    for sig in ['lactate','creatinine','potassium','wbc','glucose','bilirubin']:
        mat=get_mat(s3_df,f'{sig}_obs',mask)
        flags=np.nanmax(mat,axis=1)
        out[s4_rows,feat_idx[f'{sig}_flag']] = np.where(np.isnan(flags),0,(flags>0).astype(np.float32))

    gcs_mat=get_mat(s3_df,'gcs_total',mask)
    avpu_mat=get_mat(s3_df,'avpu_score',mask)
    out[s4_rows,feat_idx['gcs_mean']]     = np.nanmean(gcs_mat,axis=1)
    out[s4_rows,feat_idx['gcs_last']]     = last_vec(gcs_mat)
    out[s4_rows,feat_idx['gcs_obs_rate']] = np.mean(~np.isnan(gcs_mat),axis=1)
    out[s4_rows,feat_idx['avpu_sum']]     = np.nansum(avpu_mat,axis=1)

# ── Load S4 ────────────────────────────────────────────────────────────────
print("Loading S4...")
wins  = pd.read_parquet(S4_PATH).reset_index(drop=True)
wins['s4_row'] = wins.index
y_all = wins['label'].values.astype(np.int8)
grps  = wins['stay_id'].values
wins_by_stay = {sid: grp for sid, grp in wins.groupby('stay_id')}
N_WIN = len(wins)
print(f"  {N_WIN:,} windows  {len(wins_by_stay):,} stays")

# ── Age ────────────────────────────────────────────────────────────────────
print("Loading age...")
icu = pd.read_csv(r"<PHD_ROOT>\PHD Research\DATAset\mimic-iv-3.1\icu\icustays.csv",
                  usecols=['stay_id','intime'])
pts = pd.read_csv(r"<PHD_ROOT>\PHD Research\DATAset\mimic-iv-3.1\hosp\patients.csv",
                  usecols=['subject_id','anchor_age','anchor_year'])
icu = icu.merge(wins[['stay_id','subject_id']].drop_duplicates(),on='stay_id')
icu = icu.merge(pts,on='subject_id')
icu['intime'] = pd.to_datetime(icu['intime'])
icu['age']    = icu['anchor_age']+(icu['intime'].dt.year-icu['anchor_year'])
age_map       = icu.set_index('stay_id')['age'].to_dict()

# ── Pre-allocate output memmap — written directly to disk ──────────────────
MEMMAP_PATH = OUT_DIR / "X_full.npy"
print(f"Allocating memmap {N_WIN:,} × {N_FEAT} → {OUT_DIR/'X_full.npy'}...")
X = np.memmap(str(MEMMAP_PATH), dtype=np.float32, mode='w+',
              shape=(N_WIN, N_FEAT))
X[:] = np.nan  # init all to nan
print("  Done")

# ── Process S3 row-group by row-group ─────────────────────────────────────
print("Processing S3...")
t  = time.time()
pf = pq.ParquetFile(str(S3_PATH))
n_rg = pf.metadata.num_row_groups
processed_stays = set()

for rg_i in range(n_rg):
    t_rg = time.time()
    rg_df = pf.read_row_group(rg_i).to_pandas()
    rg_stays = rg_df['stay_id'].unique()
    n_processed = 0

    for stay_id in rg_stays:
        if stay_id not in wins_by_stay:
            continue
        s3_stay   = rg_df[rg_df['stay_id']==stay_id].set_index('hour_offset')
        stay_wins = wins_by_stay[stay_id]
        age       = age_map.get(stay_id, np.nan)
        process_stay_into(s3_stay, stay_wins, age, X)
        processed_stays.add(stay_id)
        n_processed += 1

    X.flush()  # write to disk
    del rg_df; gc.collect()
    print(f"  RG {rg_i+1}/{n_rg}: {n_processed:,} stays  "
          f"({time.time()-t_rg:.0f}s)  total {(time.time()-t)/60:.1f} min")

# Handle stays with no S3 data
missing = set(wins_by_stay.keys()) - processed_stays
print(f"  {len(missing):,} stays with no S3 data — filling nan")
for stay_id in missing:
    stay_wins = wins_by_stay[stay_id]
    process_stay_into(pd.DataFrame(), stay_wins, age_map.get(stay_id,np.nan), X)
X.flush()

print(f"\nFeatures done in {(time.time()-t)/60:.1f} min")

# ── Impute NaN ─────────────────────────────────────────────────────────────
print("Imputing NaN (column medians)...")
# Process column by column to avoid loading full matrix
for j in range(N_FEAT):
    col = X[:, j]
    msk = np.isnan(col)
    if msk.any():
        med = np.nanmedian(col)
        X[msk, j] = med
X.flush()
print(f"  NaN remaining: {np.isnan(X[:]).sum()}")

# ── Save folds ─────────────────────────────────────────────────────────────
print("Saving folds...")
splits = list(GroupKFold(n_splits=5).split(
    np.arange(N_WIN), y_all, grps))

for fold, (train_idx, val_idx) in enumerate(splits, 1):
    fold_dir = OUT_DIR/f"fold_{fold}"
    fold_dir.mkdir(exist_ok=True)

    # Load slices from memmap
    X_tr = np.array(X[train_idx])
    X_vl = np.array(X[val_idx])
    y_tr = y_all[train_idx]
    y_vl = y_all[val_idx]

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr).astype(np.float32)
    X_vl   = scaler.transform(X_vl).astype(np.float32)

    np.save(fold_dir/"X_train.npy",     X_tr)
    np.save(fold_dir/"X_val.npy",       X_vl)
    np.save(fold_dir/"y_train.npy",     y_tr)
    np.save(fold_dir/"y_val.npy",       y_vl)
    np.save(fold_dir/"val_indices.npy", val_idx)
    with open(fold_dir/"scaler.pkl","wb") as f:
        pickle.dump(scaler, f)
    print(f"  fold_{fold}: train={X_tr.shape} val={X_vl.shape} "
          f"pos={y_vl.mean()*100:.1f}%")
    del X_tr, X_vl; gc.collect()

print(f"\nDone. Output: {OUT_DIR}")

# %% ---- cell 4 ----
import pandas as pd
import numpy as np
from pathlib import Path
import time

CHART_PATH = Path(r"<PHD_ROOT>\PHD Research\DATAset\mimic-iv-3.1\icu\chartevents.csv")
OUT_PATH   = Path(r"<PHD_ROOT>\PHD Research\DATAset\s3\chartevents_subset.parquet")
S4_PATH    = Path(r"<PHD_ROOT>\PHD Research\DATAset\s4\windows.parquet")
S7_DIR     = Path(r"<PHD_ROOT>\PHD Research\DATAset\s7_new")
REB_DIR    = Path(r"<PHD_ROOT>\s7_rebuilt")

ITEMIDS = {
    220739: 'gcs_eye',
    223900: 'gcs_verbal',
    223901: 'gcs_motor',
    223835: 'fio2',
    224385: 'intubated',
    227021: 'intubated',
    228096: 'sedated',
}
TARGET_IDS = set(ITEMIDS.keys())

# ── Step 1: Extract from chartevents ──────────────────────────────────────
print("="*60)
print("STEP 1: Extracting chartevents subset...")
print("="*60)
t = time.time()
chunks = []
n_chunks = n_kept = 0

for chunk in pd.read_csv(
    CHART_PATH,
    usecols=['stay_id','charttime','itemid','valuenum'],
    chunksize=500000,
    dtype={'itemid':'int32','valuenum':'float32'},
    low_memory=False,
):
    filtered = chunk[chunk['itemid'].isin(TARGET_IDS)]
    if len(filtered):
        chunks.append(filtered.copy())
        n_kept += len(filtered)
    n_chunks += 1
    if n_chunks % 100 == 0:
        print(f"  Chunk {n_chunks}  kept {n_kept:,}  ({(time.time()-t)/60:.1f} min)")

print(f"Done in {(time.time()-t)/60:.1f} min  kept {n_kept:,} rows")
ce = pd.concat(chunks, ignore_index=True)
del chunks
ce['signal'] = ce['itemid'].map(ITEMIDS)
ce.to_parquet(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}  shape={ce.shape}")
print(ce['signal'].value_counts())

# ── Step 2: Load icustays for intime (to compute hour_offset) ─────────────
print("\n" + "="*60)
print("STEP 2: Computing hour offsets...")
print("="*60)
icu = pd.read_csv(r"<PHD_ROOT>\PHD Research\DATAset\mimic-iv-3.1\icu\icustays.csv",
                  usecols=['stay_id','intime'])
icu['intime'] = pd.to_datetime(icu['intime'])
intime_map    = icu.set_index('stay_id')['intime'].to_dict()

ce['charttime'] = pd.to_datetime(ce['charttime'])
ce['stay_id']   = ce['stay_id'].astype('int64')
ce['intime']    = ce['stay_id'].map(intime_map)
ce['hour_offset'] = ((ce['charttime'] - ce['intime'])
                     .dt.total_seconds() / 3600).round().astype(int)
ce = ce[ce['hour_offset'] >= 0].copy()
print(f"  After filtering negative offsets: {len(ce):,} rows")

# ── Step 3: Pivot to wide format per stay/hour ────────────────────────────
print("\n" + "="*60)
print("STEP 3: Pivoting to wide format...")
print("="*60)

# For intubated/sedated: binary (was it charted in this hour?)
# For fio2/gcs: numeric mean per hour
ce_numeric = ce[ce['signal'].isin(['gcs_eye','gcs_verbal','gcs_motor','fio2'])].copy()
ce_binary  = ce[ce['signal'].isin(['intubated','sedated'])].copy()

# Numeric: mean per stay/hour/signal
num_wide = (ce_numeric
            .groupby(['stay_id','hour_offset','signal'])['valuenum']
            .mean()
            .unstack('signal')
            .reset_index())
num_wide.columns.name = None
print(f"  Numeric wide shape: {num_wide.shape}")
print(f"  Numeric columns: {num_wide.columns.tolist()}")

# Binary: max per stay/hour/signal (1 if any charted)
bin_wide = (ce_binary
            .groupby(['stay_id','hour_offset','signal'])['valuenum']
            .count()
            .unstack('signal')
            .reset_index())
bin_wide.columns.name = None
bin_wide = bin_wide.rename(columns={'intubated':'intubated_raw','sedated':'sedated_raw'})
bin_wide['intubated_flag'] = (bin_wide.get('intubated_raw', 0) > 0).astype(np.float32)
bin_wide['sedated_flag']   = (bin_wide.get('sedated_raw',   0) > 0).astype(np.float32)
bin_wide = bin_wide[['stay_id','hour_offset','intubated_flag','sedated_flag']]
print(f"  Binary wide shape: {bin_wide.shape}")

# Merge numeric and binary
ce_wide = num_wide.merge(bin_wide, on=['stay_id','hour_offset'], how='outer')
print(f"  Combined wide shape: {ce_wide.shape}")
print(f"  Columns: {ce_wide.columns.tolist()}")

# ── Step 4: Load S4 windows and compute per-window features ───────────────
print("\n" + "="*60)
print("STEP 4: Computing per-window chart features...")
print("="*60)
import warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')

wins = pd.read_parquet(S4_PATH).reset_index(drop=True)
wins['s4_row'] = wins.index
N_WIN = len(wins)
print(f"  {N_WIN:,} windows")

# Index ce_wide by stay/hour
ce_by_stay = {sid: grp.set_index('hour_offset')
              for sid, grp in ce_wide.groupby('stay_id')}
print(f"  Chart data for {len(ce_by_stay):,} stays")

from scipy import stats

def slope_vec(mat):
    x=np.arange(mat.shape[1],dtype=np.float32)
    valid=~np.isnan(mat); n_v=valid.sum(axis=1).astype(np.float32)
    m=np.where(valid,mat,0.0)
    sx=(x*valid).sum(axis=1); sy=m.sum(axis=1)
    sx2=(x**2*valid).sum(axis=1); sxy=(x*m).sum(axis=1)
    denom=n_v*sx2-sx**2
    return np.where(n_v<2,0.0,np.where(denom!=0,(n_v*sxy-sx*sy)/denom,0.0)).astype(np.float32)

def last_vec(mat):
    has_v=~np.all(np.isnan(mat),axis=1)
    rev=mat[:,::-1]; idx=np.argmax(~np.isnan(rev),axis=1)
    return np.where(has_v,rev[np.arange(mat.shape[0]),idx],np.nan).astype(np.float32)

def delta_vec(mat):
    has_v=~np.all(np.isnan(mat),axis=1); n=mat.shape[0]
    idx_f=np.argmax(~np.isnan(mat),axis=1)
    idx_l=mat.shape[1]-1-np.argmax(~np.isnan(mat[:,::-1]),axis=1)
    return np.where(has_v,mat[np.arange(n),idx_l]-mat[np.arange(n),idx_f],0.0).astype(np.float32)

def get_mat(df, col, mask):
    raw = df[col].values.astype(np.float32) if col in df.columns \
          else np.full(mask.shape[1], np.nan, dtype=np.float32)
    return np.where(mask, raw[np.newaxis,:], np.nan)

# Feature columns we need to fill
GCS_COMP = ['gcs_eye','gcs_verbal','gcs_motor']
FIO2_AGG = ['fio2_mean','fio2_min','fio2_max','fio2_std','fio2_last',
            'fio2_slope','fio2_delta','fio2_miss_rate','fio2_cross_pct','fio2_obs_rate']
GCS_AGG  = [f'{s}_{k}' for s in GCS_COMP
            for k in ['mean','min','max','std','last','slope','delta','miss_rate','cross_pct']]

# Load feature_cols
with open(S7_DIR / "feature_cols.txt") as f:
    feature_cols = [l.strip() for l in f.readlines()]
feat_idx = {c:i for i,c in enumerate(feature_cols)}

# For each fold, load X memmap and fill in the chart features
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import pickle, gc

y_all  = wins['label'].values.astype(np.int8)
grps   = wins['stay_id'].values
splits = list(GroupKFold(n_splits=5).split(np.arange(N_WIN), y_all, grps))

# Build chart features for ALL windows first
print("  Building chart feature array...")
t = time.time()

# Allocate chart-only feature array (just the new cols)
CHART_COLS = FIO2_AGG + GCS_AGG + ['intubated_flag','sedated_flag']
n_chart    = len(CHART_COLS)
chart_idx  = {c:i for i,c in enumerate(CHART_COLS)}

chart_arr  = np.full((N_WIN, n_chart), np.nan, dtype=np.float32)

stays = list(wins.groupby('stay_id'))
for s_i, (stay_id, stay_wins) in enumerate(stays):
    if s_i % 5000 == 0:
        print(f"    Stay {s_i:,}/{len(stays):,}  ({(time.time()-t)/60:.1f} min)")

    ce_df    = ce_by_stay.get(stay_id, pd.DataFrame())
    s4_rows  = stay_wins['s4_row'].values
    w_starts = stay_wins['window_start'].values
    w_ends   = stay_wins['window_end'].values
    n_wins   = len(s4_rows)

    if len(ce_df) == 0:
        chart_arr[s4_rows, chart_idx['intubated_flag']] = 0.0
        chart_arr[s4_rows, chart_idx['sedated_flag']]   = 0.0
        continue

    all_hours = ce_df.index.values
    mask = ((all_hours[np.newaxis,:] >= w_starts[:,np.newaxis]) &
            (all_hours[np.newaxis,:] <  w_ends[:,np.newaxis]))

    # FIO2
    mat = get_mat(ce_df, 'fio2', mask)
    chart_arr[s4_rows, chart_idx['fio2_mean']]      = np.nanmean(mat, axis=1)
    chart_arr[s4_rows, chart_idx['fio2_min']]       = np.nanmin(mat,  axis=1)
    chart_arr[s4_rows, chart_idx['fio2_max']]       = np.nanmax(mat,  axis=1)
    chart_arr[s4_rows, chart_idx['fio2_std']]       = np.nanstd(mat,  axis=1)
    chart_arr[s4_rows, chart_idx['fio2_last']]      = last_vec(mat)
    chart_arr[s4_rows, chart_idx['fio2_slope']]     = slope_vec(mat)
    chart_arr[s4_rows, chart_idx['fio2_delta']]     = delta_vec(mat)
    chart_arr[s4_rows, chart_idx['fio2_miss_rate']] = np.mean(np.isnan(mat), axis=1)
    chart_arr[s4_rows, chart_idx['fio2_cross_pct']] = 0.0
    chart_arr[s4_rows, chart_idx['fio2_obs_rate']]  = np.mean(~np.isnan(mat), axis=1)

    # GCS components
    for sig in GCS_COMP:
        mat = get_mat(ce_df, sig, mask)
        lo,hi = (1,4) if 'eye' in sig else (1,5) if 'verbal' in sig else (1,6)
        chart_arr[s4_rows, chart_idx[f'{sig}_mean']]      = np.nanmean(mat, axis=1)
        chart_arr[s4_rows, chart_idx[f'{sig}_min']]       = np.nanmin(mat,  axis=1)
        chart_arr[s4_rows, chart_idx[f'{sig}_max']]       = np.nanmax(mat,  axis=1)
        chart_arr[s4_rows, chart_idx[f'{sig}_std']]       = np.nanstd(mat,  axis=1)
        chart_arr[s4_rows, chart_idx[f'{sig}_last']]      = last_vec(mat)
        chart_arr[s4_rows, chart_idx[f'{sig}_slope']]     = slope_vec(mat)
        chart_arr[s4_rows, chart_idx[f'{sig}_delta']]     = delta_vec(mat)
        chart_arr[s4_rows, chart_idx[f'{sig}_miss_rate']] = np.mean(np.isnan(mat), axis=1)
        chart_arr[s4_rows, chart_idx[f'{sig}_cross_pct']] = np.nanmean(
            (mat<lo)|(mat>hi), axis=1)

    # Intubated / Sedated
    int_mat = get_mat(ce_df, 'intubated_flag', mask)
    sed_mat = get_mat(ce_df, 'sedated_flag',   mask)
    chart_arr[s4_rows, chart_idx['intubated_flag']] = np.nanmax(int_mat, axis=1)
    chart_arr[s4_rows, chart_idx['sedated_flag']]   = np.nanmax(sed_mat, axis=1)

# Fill remaining NaN for intubated/sedated with 0
for col in ['intubated_flag','sedated_flag']:
    ci = chart_idx[col]
    chart_arr[np.isnan(chart_arr[:,ci]), ci] = 0.0

print(f"  Chart features done in {(time.time()-t)/60:.1f} min")

# ── Step 5: Merge into rebuilt X and save updated folds ───────────────────
print("\n" + "="*60)
print("STEP 5: Merging chart features into rebuilt X and saving folds...")
print("="*60)

# Load full rebuilt X
MEMMAP_PATH = REB_DIR / "X_full.npy"
X_full = np.memmap(str(MEMMAP_PATH), dtype=np.float32, mode='r+',
                   shape=(N_WIN, len(feature_cols)))

# Write chart features into correct columns of X_full
for col in CHART_COLS:
    if col in feat_idx:
        X_full[:, feat_idx[col]] = chart_arr[:, chart_idx[col]]
X_full.flush()
print("  Chart features merged into X_full memmap")

# Re-impute NaN
print("  Imputing NaN...")
for j in range(len(feature_cols)):
    col = X_full[:, j]
    msk = np.isnan(col)
    if msk.any():
        med = np.nanmedian(col[~msk])
        X_full[msk, j] = med if not np.isnan(med) else 0.0
X_full.flush()
print(f"  NaN remaining: {np.isnan(X_full[:]).sum()}")

# Save updated folds
for fold, (train_idx, val_idx) in enumerate(splits, 1):
    fold_dir = REB_DIR / f"fold_{fold}"
    fold_dir.mkdir(exist_ok=True)

    X_tr = np.array(X_full[train_idx])
    X_vl = np.array(X_full[val_idx])
    y_tr = y_all[train_idx]
    y_vl = y_all[val_idx]

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr).astype(np.float32)
    X_vl   = scaler.transform(X_vl).astype(np.float32)

    np.save(fold_dir/"X_train.npy",     X_tr)
    np.save(fold_dir/"X_val.npy",       X_vl)
    np.save(fold_dir/"y_train.npy",     y_tr)
    np.save(fold_dir/"y_val.npy",       y_vl)
    np.save(fold_dir/"val_indices.npy", val_idx)
    with open(fold_dir/"scaler.pkl","wb") as f:
        pickle.dump(scaler, f)

    del X_tr, X_vl; gc.collect()
    print(f"  fold_{fold}: saved  pos={y_vl.mean()*100:.1f}%")

print(f"\nAll done. Updated folds saved to {REB_DIR}")
