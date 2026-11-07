# Source: temporal holdout validation.ipynb, code cells [0] (outputs not included)

# %% ---- cell 0 ----
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle, time, gc
import warnings
warnings.filterwarnings('ignore')

S4_PATH  = Path(r"<PHD_ROOT>\PHD Research\DATAset\s4\windows.parquet")
REB_DIR  = Path(r"<PHD_ROOT>\s7_rebuilt")
S9_DIR   = Path(r"<PHD_ROOT>\s9_temporal")
XGBF_DIR = Path(r"<PHD_ROOT>\s10_xgboost\flattened")
ICU_PATH = Path(r"<PHD_ROOT>\PHD Research\DATAset\mimic-iv-3.1\icu\icustays.csv")
OUT_DIR  = Path(r"<PHD_ROOT>\temporal_holdout")
OUT_DIR.mkdir(exist_ok=True)

S7_DIR   = Path(r"<PHD_ROOT>\PHD Research\DATAset\s7_new")
with open(S7_DIR / "feature_cols.txt") as f:
    feature_cols = [l.strip() for l in f.readlines()]
N_FEAT = len(feature_cols)

# ── TCN Architecture ───────────────────────────────────────────────────────
class TCNBlock(nn.Module):
    def __init__(self, channels, kernel, dilation):
        super().__init__()
        pad = (kernel-1)*dilation
        self.conv1 = nn.Conv1d(channels,channels,kernel,dilation=dilation,padding=pad)
        self.conv2 = nn.Conv1d(channels,channels,kernel,dilation=dilation,padding=pad)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(0.1)
    def forward(self, x):
        T   = x.size(2)
        out = self.relu(self.norm1(self.conv1(x)[..., :T]))
        out = self.drop(out)
        out = self.relu(self.norm2(self.conv2(out)[..., :T]))
        out = self.drop(out)
        return out + x

class TCNAttention(nn.Module):
    def __init__(self,n_signals=67,n_steps=6,d_model=128,dilations=(1,2,4)):
        super().__init__()
        self.proj=nn.Conv1d(n_signals,d_model,kernel_size=1)
        self.tcn_blocks=nn.ModuleList([TCNBlock(d_model,2,d) for d in dilations])
        self.attn=nn.Linear(d_model,1)
        self.classifier=nn.Sequential(nn.Linear(d_model,64),nn.ReLU(),
                                       nn.Dropout(0.1),nn.Linear(64,1))
    def forward(self,x):
        x=x.permute(0,2,1); x=self.proj(x)
        for b in self.tcn_blocks: x=b(x)
        a=x.permute(0,2,1)
        w=torch.softmax(self.attn(a).squeeze(-1),dim=-1)
        c=(a*w.unsqueeze(-1)).sum(dim=1)
        return self.classifier(c).squeeze(-1)

# ── Step 1: Build temporal split ──────────────────────────────────────────
print("="*60)
print("STEP 1: Building temporal holdout split")
print("="*60)

wins = pd.read_parquet(S4_PATH, columns=['stay_id','label'])
wins = wins.reset_index(drop=True)
wins['s4_row'] = wins.index
y_all = wins['label'].values.astype(np.int8)

icu = pd.read_csv(ICU_PATH, usecols=['stay_id','intime'])
icu['intime'] = pd.to_datetime(icu['intime'])
intime_map    = icu.set_index('stay_id')['intime'].to_dict()
wins['intime'] = wins['stay_id'].map(intime_map)
wins = wins.dropna(subset=['intime'])

stays_sorted  = wins.drop_duplicates('stay_id').sort_values('intime')
n_stays       = len(stays_sorted)
cutoff_idx    = int(n_stays * 0.80)
cutoff_date   = stays_sorted.iloc[cutoff_idx]['intime']
print(f"  Total stays: {n_stays:,}")
print(f"  Cutoff date: {cutoff_date.date()}")

holdout_stays = set(stays_sorted.iloc[cutoff_idx:]['stay_id'])
train_stays   = set(stays_sorted.iloc[:cutoff_idx]['stay_id'])

holdout_mask    = wins['stay_id'].isin(holdout_stays)
holdout_s4_rows = wins.loc[holdout_mask,'s4_row'].values
y_holdout       = wins.loc[holdout_mask,'label'].values.astype(np.int8)
train_mask      = wins['stay_id'].isin(train_stays)
y_train         = wins.loc[train_mask,'label'].values.astype(np.int8)

print(f"  Train:   {train_mask.sum():,} windows  pos={y_train.mean()*100:.1f}%")
print(f"  Holdout: {len(holdout_s4_rows):,} windows  pos={y_holdout.mean()*100:.1f}%")
np.save(OUT_DIR/"holdout_s4_rows.npy", holdout_s4_rows)
np.save(OUT_DIR/"y_holdout.npy",       y_holdout)
del wins, icu; gc.collect()

# ── Helper: chunked inference ──────────────────────────────────────────────
CHUNK = 50000  # process 50k windows at a time — ~45MB per chunk

def predict_lgb_chunked(model, X_memmap, row_idx, scaler, chunk=CHUNK):
    probs = np.zeros(len(row_idx), dtype=np.float32)
    for i in range(0, len(row_idx), chunk):
        idx_b = row_idx[i:i+chunk]
        X_b   = np.array(X_memmap[idx_b])
        X_b   = scaler.transform(X_b).astype(np.float32)
        probs[i:i+chunk] = model.predict(X_b)
        if i % 200000 == 0:
            print(f"    chunk {i//chunk+1}/{(len(row_idx)+chunk-1)//chunk}")
    return probs

def predict_xgb_chunked(model, X_memmap, row_idx, scaler, chunk=CHUNK):
    probs = np.zeros(len(row_idx), dtype=np.float32)
    for i in range(0, len(row_idx), chunk):
        idx_b = row_idx[i:i+chunk]
        X_b   = np.array(X_memmap[idx_b])
        X_b   = scaler.transform(X_b).astype(np.float32)
        dm    = xgb.DMatrix(X_b)
        probs[i:i+chunk] = model.predict(dm)
    return probs

def predict_xgbf_chunked(model, seq_memmap, row_idx, chunk=CHUNK,
                          n_steps=6, n_sig=67):
    probs = np.zeros(len(row_idx), dtype=np.float32)
    for i in range(0, len(row_idx), chunk):
        idx_b = row_idx[i:i+chunk]
        X_b   = np.array(seq_memmap[idx_b]).reshape(len(idx_b), -1)
        dm    = xgb.DMatrix(X_b)
        probs[i:i+chunk] = model.predict(dm)
    return probs

def predict_tcn_chunked(model, seq_memmap, row_idx, scaler,
                         chunk=CHUNK, n_steps=6, n_sig=67):
    probs  = np.zeros(len(row_idx), dtype=np.float32)
    n_feat = scaler.n_features_in_
    device = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        for i in range(0, len(row_idx), chunk):
            idx_b = row_idx[i:i+chunk]
            X_b   = np.array(seq_memmap[idx_b])  # (B, 6, 67)
            X_norm = X_b.copy()
            for t in range(n_steps):
                X_norm[:,t,:n_feat] = scaler.transform(X_b[:,t,:n_feat])
            t_b = torch.tensor(X_norm, dtype=torch.float32)
            probs[i:i+chunk] = torch.sigmoid(model(t_b)).numpy()
    return probs

# ── Load memmaps (read-only, minimal RAM) ─────────────────────────────────
print("\nLoading memmaps (read-only)...")
MEMMAP_PATH = REB_DIR / "X_full.npy"
N_WIN       = len(y_all)
X_full      = np.memmap(str(MEMMAP_PATH), dtype=np.float32,
                         mode='r', shape=(N_WIN, N_FEAT))

S9_MEMMAP   = S9_DIR / "X_sequences.npy"
X_seq       = np.memmap(str(S9_MEMMAP), dtype=np.float32,
                         mode='r', shape=(N_WIN, 6, 67))
print(f"  X_full:  {X_full.shape}")
print(f"  X_seq:   {X_seq.shape}")

# Load fold 1 scaler for LGB/XGB
with open(REB_DIR/"fold_1"/"scaler.pkl","rb") as f:
    scaler_lgb = pickle.load(f)

# ── Step 2: LightGBM holdout ───────────────────────────────────────────────
print(f"\n{'='*60}")
print("LightGBM holdout inference...")
lgb_probs_folds = []
for fold in range(1,6):
    print(f"  Fold {fold}...")
    model = lgb.Booster(model_file=str(REB_DIR/f"fold_{fold}"/"lgb_model.txt"))
    p     = predict_lgb_chunked(model, X_full, holdout_s4_rows, scaler_lgb)
    lgb_probs_folds.append(p)
    del model; gc.collect()
lgb_probs = np.mean(lgb_probs_folds, axis=0)
lgb_auroc = roc_auc_score(y_holdout, lgb_probs)
lgb_auprc = average_precision_score(y_holdout, lgb_probs)
print(f"  LGB: AUROC={lgb_auroc:.4f}  AUPRC={lgb_auprc:.4f}")
np.save(OUT_DIR/"lgb_holdout_probs.npy", lgb_probs)
del lgb_probs_folds; gc.collect()

# ── Step 3: XGBoost Agg holdout ───────────────────────────────────────────
print(f"\n{'='*60}")
print("XGBoost Agg holdout inference...")
xgb_probs_folds = []
for fold in range(1,6):
    print(f"  Fold {fold}...")
    model = xgb.Booster()
    model.load_model(str(REB_DIR/f"fold_{fold}"/"xgb_model.json"))
    p     = predict_xgb_chunked(model, X_full, holdout_s4_rows, scaler_lgb)
    xgb_probs_folds.append(p)
    del model; gc.collect()
xgb_probs = np.mean(xgb_probs_folds, axis=0)
xgb_auroc = roc_auc_score(y_holdout, xgb_probs)
xgb_auprc = average_precision_score(y_holdout, xgb_probs)
print(f"  XGB_Agg: AUROC={xgb_auroc:.4f}  AUPRC={xgb_auprc:.4f}")
np.save(OUT_DIR/"xgb_holdout_probs.npy", xgb_probs)
del xgb_probs_folds; gc.collect()

# ── Step 4: XGBoost Flattened holdout ────────────────────────────────────
print(f"\n{'='*60}")
print("XGBoost Flattened holdout inference...")
xgbf_probs_folds = []
for fold in range(1,6):
    print(f"  Fold {fold}...")
    model = xgb.Booster()
    model.load_model(str(XGBF_DIR/f"fold_{fold}"/"model.json"))
    p     = predict_xgbf_chunked(model, X_seq, holdout_s4_rows)
    xgbf_probs_folds.append(p)
    del model; gc.collect()
xgbf_probs = np.mean(xgbf_probs_folds, axis=0)
xgbf_auroc = roc_auc_score(y_holdout, xgbf_probs)
xgbf_auprc = average_precision_score(y_holdout, xgbf_probs)
print(f"  XGB_Flt: AUROC={xgbf_auroc:.4f}  AUPRC={xgbf_auprc:.4f}")
np.save(OUT_DIR/"xgbf_holdout_probs.npy", xgbf_probs)
del xgbf_probs_folds; gc.collect()

# ── Step 5: TCN holdout ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print("TCN holdout inference...")
tcn_probs_folds = []
for fold in range(1,6):
    print(f"  Fold {fold}...")
    fold_dir = S9_DIR/f"fold_{fold}"
    with open(fold_dir/"scaler.pkl","rb") as f:
        scaler_tcn = pickle.load(f)
    model = TCNAttention()
    ckpt  = torch.load(fold_dir/"best_model.pt", map_location="cpu")
    model.load_state_dict(ckpt)
    p     = predict_tcn_chunked(model, X_seq, holdout_s4_rows, scaler_tcn)
    tcn_probs_folds.append(p)
    del model, scaler_tcn; gc.collect()
tcn_probs = np.mean(tcn_probs_folds, axis=0)
tcn_auroc = roc_auc_score(y_holdout, tcn_probs)
tcn_auprc = average_precision_score(y_holdout, tcn_probs)
print(f"  TCN: AUROC={tcn_auroc:.4f}  AUPRC={tcn_auprc:.4f}")
np.save(OUT_DIR/"tcn_holdout_probs.npy", tcn_probs)
del tcn_probs_folds; gc.collect()

# ── Step 6: Ensemble ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Ensemble...")
avg_probs  = (lgb_probs+xgb_probs+xgbf_probs+tcn_probs)/4
avg_auroc  = roc_auc_score(y_holdout, avg_probs)
avg_auprc  = average_precision_score(y_holdout, avg_probs)
print(f"  Ensemble (avg): AUROC={avg_auroc:.4f}  AUPRC={avg_auprc:.4f}")

# Logistic stacking meta-learner
from sklearn.linear_model import LogisticRegression
cv_lgb  = np.concatenate([np.load(REB_DIR/f"fold_{f}"/"val_probs.npy")
                           for f in range(1,6)])
cv_xgb  = np.concatenate([np.load(REB_DIR/f"fold_{f}"/"val_probs_xgb.npy")
                           for f in range(1,6)])
cv_xgbf = np.concatenate([np.load(XGBF_DIR/f"fold_{f}"/"val_probs.npy")
                           for f in range(1,6)])
cv_tcn  = np.concatenate([np.load(S9_DIR/f"fold_{f}"/"val_probs.npy")
                           for f in range(1,6)])
cv_y    = np.concatenate([np.load(REB_DIR/f"fold_{f}"/"val_labels.npy")
                           for f in range(1,6)])

meta = LogisticRegression(C=1.0, max_iter=1000)
meta.fit(np.column_stack([cv_lgb,cv_xgb,cv_xgbf,cv_tcn]), cv_y)
X_stack     = np.column_stack([lgb_probs,xgb_probs,xgbf_probs,tcn_probs])
stack_probs = meta.predict_proba(X_stack)[:,1]
stack_auroc = roc_auc_score(y_holdout, stack_probs)
stack_auprc = average_precision_score(y_holdout, stack_probs)
print(f"  Ensemble (stack): AUROC={stack_auroc:.4f}  AUPRC={stack_auprc:.4f}")
print(f"  Stacking weights: lgb={meta.coef_[0][0]:.3f}  "
      f"xgb={meta.coef_[0][1]:.3f}  "
      f"xgbf={meta.coef_[0][2]:.3f}  "
      f"tcn={meta.coef_[0][3]:.3f}")

# ── Step 7: Summary table ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print("TEMPORAL HOLDOUT RESULTS")
print(f"{'='*60}")
cv_aurocs = {'lgb':0.8064,'xgb_agg':0.8067,'xgb_flt':0.8116,
             'tcn':0.8077,'ensemble':0.8149,'ensemble_stack':0.8151}
names  = ['lgb','xgb_agg','xgb_flt','tcn','ensemble','ensemble_stack']
h_aurocs=[lgb_auroc,xgb_auroc,xgbf_auroc,tcn_auroc,avg_auroc,stack_auroc]
h_auprcs=[lgb_auprc,xgb_auprc,xgbf_auprc,tcn_auprc,avg_auprc,stack_auprc]

print(f"\n{'Model':20s} {'Hold AUROC':12s} {'CV AUROC':12s} "
      f"{'Gap':8s} {'Hold AUPRC':12s}")
print("-"*65)
for name,ha,hp in zip(names,h_aurocs,h_auprcs):
    cv = cv_aurocs[name]
    print(f"  {name:18s} {ha:.4f}       {cv:.4f}       "
          f"{ha-cv:+.4f}   {hp:.4f}")

df = pd.DataFrame({'model':names,'holdout_auroc':h_aurocs,
                   'holdout_auprc':h_auprcs,
                   'cv_auroc':[cv_aurocs[n] for n in names],
                   'gap':[ha-cv_aurocs[n] for n,ha in zip(names,h_aurocs)]})
df.to_csv(OUT_DIR/"holdout_results.csv", index=False)

# ── Step 8: Calibration + ROC plots ──────────────────────────────────────
print("\nPlotting figures...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve

model_probs = {
    'LGB':              lgb_probs,
    'XGB_Agg':          xgb_probs,
    'XGB_Flt':          xgbf_probs,
    'TCN':              tcn_probs,
    'Ensemble (avg)':   avg_probs,
    'Ensemble (stack)': stack_probs,
}

# ROC curves
fig, ax = plt.subplots(figsize=(8,7))
colors  = ['#e74c3c','#e67e22','#2ecc71','#3498db','#9b59b6','#1a4f8a']
for (name,probs),color in zip(model_probs.items(),colors):
    fpr,tpr,_ = roc_curve(y_holdout,probs)
    auroc     = roc_auc_score(y_holdout,probs)
    ax.plot(fpr,tpr,color=color,linewidth=2,
            label=f'{name} ({auroc:.4f})')
ax.plot([0,1],[0,1],'k--',linewidth=1,label='Random (0.5000)')
ax.set_xlabel('False Positive Rate',fontsize=12)
ax.set_ylabel('True Positive Rate',fontsize=12)
ax.set_title('ROC Curves — Temporal Holdout\n(Last 20% ICU Admissions by Date)',
             fontsize=12,fontweight='bold')
ax.legend(fontsize=9,loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR/"roc_curves_holdout.png",dpi=300,bbox_inches='tight')
plt.savefig(OUT_DIR/"roc_curves_holdout.pdf",bbox_inches='tight')
plt.close()

# Calibration curves
fig,axes = plt.subplots(2,3,figsize=(15,10))
for ax,(name,probs) in zip(axes.flat,model_probs.items()):
    frac,mean = calibration_curve(y_holdout,probs,n_bins=20,strategy='uniform')
    ax.plot(mean,frac,'s-',color='#2e75b6',linewidth=2,markersize=4)
    ax.plot([0,1],[0,1],'k--',linewidth=1)
    ax.set_title(f'{name}\nAUROC={roc_auc_score(y_holdout,probs):.4f}',
                 fontsize=10,fontweight='bold')
    ax.set_xlabel('Mean predicted prob',fontsize=8)
    ax.set_ylabel('Fraction positives',fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle('Calibration Curves — Temporal Holdout Validation',
             fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR/"calibration_holdout.png",dpi=300,bbox_inches='tight')
plt.savefig(OUT_DIR/"calibration_holdout.pdf",bbox_inches='tight')
plt.close()

print(f"\nAll saved to {OUT_DIR}")
print("Temporal holdout validation complete.")
