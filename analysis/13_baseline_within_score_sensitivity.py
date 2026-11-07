# Source: paper2.ipynb, code cells [16] (outputs not included)

# %% ---- cell 16 ----
# ============================================================================
# Paper 2 — within-score baseline, red scores, pooled 0-4, logit-Platt sensitivity
#   Baseline model: NEWS-2 inputs only (13 features at window_end, final carry-forward limits):
#     6 component values (RR, SpO2, SBP, HR, temperature, GCS), 6 sub-scores, oxygen points (NEWS-2_7 - NEWS-2_6)
#     LightGBM, same hyperparameters, same folds (asserted identical to the corrected ladder), Platt on pooled OOF.
#     Note: the baseline uses hour window_end; the full model uses only hours before it (conservative comparison).
#   Red score: any single component scoring 3 at window_end (oxygen cannot score 3).
#   Bootstrap: same patient codes, seed and draws as the P2 bootstrap (B = 2,000) -> paired with all P2 results.
#   Logit-Platt: Platt refitted on logit(OOF raw probability), applied to raw held-out predictions of the full model.
# Output: <PHD_ROOT>\Paper2\corrected\P2_baseline_* .csv/.npy, P2_calibration_logitplatt*.csv
# ============================================================================
import os, sys, time, gc, json
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, r"<PHD_ROOT>")
from paths.paper2 import *
from paths.registry import register
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

N_TOTAL, K, B, SEED, CHUNK = 3_498_146, 100_000, 2000, 42, 200_000
OUT = P2_OUT; PVC = os.path.join(CORRECTED_DIR, "prediction_vectors")
FINAL_CAPS = {"rr": 4, "spo2": 4, "sbp": 4, "hr": 8, "temp": 4, "cons": 48}
LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, max_depth=-1, min_child_samples=50, subsample=0.8,
                  colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, n_jobs=4, verbose=-1, force_row_wise=True,
                  objective="binary", metric="auc", seed=SEED, bagging_seed=SEED, feature_fraction_seed=SEED)
t00 = time.time()
def log(m): print(f"[{time.time()-t00:6.0f}s] {m}", flush=True)

def s_rr(v):   return np.select([np.isnan(v), v <= 8, v <= 11, v <= 20, v <= 24, v >= 25], [0, 3, 1, 0, 2, 3], default=0)
def s_spo2(v): return np.select([np.isnan(v), v <= 91, v <= 93, v <= 95, v >= 96], [0, 3, 2, 1, 0], default=0)
def s_sbp(v):  return np.select([np.isnan(v), v <= 90, v <= 100, v <= 110, v <= 219, v >= 220], [0, 3, 2, 1, 0, 3], default=0)
def s_hr(v):   return np.select([np.isnan(v), v <= 40, v <= 50, v <= 90, v <= 110, v <= 130, v >= 131], [0, 3, 1, 0, 1, 2, 3], default=0)
def s_temp(v): return np.select([np.isnan(v), v <= 35.0, v <= 36.0, v <= 38.0, v <= 39.0, v >= 39.1], [0, 3, 1, 0, 1, 2], default=0)
def s_cons(v): return np.where(~np.isnan(v) & (v < 15), 3, 0)
COMP = {"rr": ("rr", "rr_obs", s_rr), "spo2": ("spo2", "spo2_obs", s_spo2), "sbp": ("sbp", "sbp_obs", s_sbp),
        "hr": ("hr", "hr_obs", s_hr), "temp": ("temp", "temp_obs", s_temp), "cons": ("gcs_total", "gcs_obs", s_cons)}

# ---------------------------------------------------------------- windows, labels, splits
W = pd.read_parquet(WINDOWS, columns=["window_id", "subject_id", "stay_id", "window_end", "label"]).sort_values("window_id")
assert len(W) == N_TOTAL
subj = W.subject_id.to_numpy(np.int64); sid = W.stay_id.to_numpy(np.int64); we = W.window_end.to_numpy(np.int64)
y_orig = W.label.to_numpy().astype(np.int8); del W; gc.collect()
yR = np.load(Y_REPAIRED).astype(np.int8); keep = np.load(KEEP_REPAIRED)
train_all, hold_all = np.load(TRAIN_ROWS), np.load(HOLDOUT_ROWS)
rows = np.load(CORRECTED_HO_ROWS); assert np.array_equal(rows, hold_all[keep[hold_all]])
dev = train_all[keep[train_all]]; assert len(dev) == 2_647_017

# ---------------------------------------------------------------- baseline features at window_end
cols = ["stay_id", "hour_offset"] + [c for v in COMP.values() for c in v[:2]]
hs = pd.read_parquet(HOURLY, columns=cols)
hs = hs[hs.stay_id.astype(np.int64).isin(np.unique(sid))].copy()
hs["stay_id"], hs["hour_offset"] = hs.stay_id.astype(np.int64), hs.hour_offset.astype(np.int64)
hs = hs.sort_values(["stay_id", "hour_offset"]).reset_index(drop=True)
idx = pd.Index(hs.stay_id.to_numpy() * K + hs.hour_offset.to_numpy()); assert idx.is_unique
grp = hs.stay_id.to_numpy(); p_we = idx.get_indexer(sid * K + we)
VAL, SUB = {}, {}
for c, (vc, oc, fn) in COMP.items():
    ch = hs[vc].where(hs[oc].fillna(0).astype(int) == 1)
    ff = ch.groupby(grp).ffill(limit=FINAL_CAPS[c]).to_numpy(float)
    VAL[c] = np.where(p_we >= 0, ff[np.maximum(p_we, 0)], np.nan); SUB[c] = fn(VAL[c]).astype(float)
del hs, idx, grp; gc.collect()
s6 = sum(SUB.values())
assert np.array_equal(s6, np.load(NEWS2_6_REP).astype(float)), "sub-scores do not reproduce the repaired NEWS-2_6"
n7 = np.load(NEWS7_REP).astype(float); o2 = n7 - s6
XB = np.column_stack([VAL[c] for c in COMP] + [SUB[c] for c in COMP] + [o2]).astype(np.float32)
FEATS = [f"{c}_value" for c in COMP] + [f"{c}_subscore" for c in COMP] + ["oxygen_points"]
red_all = np.zeros(N_TOTAL, bool)
for c in COMP: red_all |= SUB[c] == 3
log(f"baseline features built ({XB.shape[1]}): {FEATS}")

# ---------------------------------------------------------------- train baseline (same folds as the corrected ladder)
splits_all = list(GroupKFold(n_splits=5).split(np.zeros(len(train_all)), y_orig[train_all], groups=subj[train_all]))
cv_rows = [(train_all[tr][keep[train_all[tr]]], train_all[va][keep[train_all[va]]]) for tr, va in splits_all]
for kf, (_, va) in enumerate(cv_rows, 1):
    assert np.array_equal(va, np.load(os.path.join(PVC, f"full_cvfold{kf}_rows.npy"))), f"fold {kf} differs from the ladder"
def fit(tr):
    yt = yR[tr]; prm = dict(LGB_PARAMS); prm["scale_pos_weight"] = (yt == 0).sum() / (yt == 1).sum()
    ds = lgb.Dataset(XB[tr], label=yt, params={"max_bin": 63, "feature_pre_filter": False})
    return lgb.train(prm, ds, num_boost_round=300)
oof_p, oof_y, fold_auc = [], [], []
for kf, (tr, va) in enumerate(cv_rows, 1):
    bst = fit(tr); p = bst.predict(XB[va]); oof_p.append(p); oof_y.append(yR[va])
    fold_auc.append(roc_auc_score(yR[va], p)); log(f"baseline fold {kf}: AUROC {fold_auc[-1]:.4f}")
bst = fit(dev); raw_ho = bst.predict(XB[rows])
platt = LogisticRegression(C=1e10, max_iter=1000).fit(np.concatenate(oof_p).reshape(-1, 1), np.concatenate(oof_y))
pb = platt.predict_proba(raw_ho.reshape(-1, 1))[:, 1]
f_pb = os.path.join(OUT, "P2_baseline_holdout_probs_CALIBRATED.npy"); np.save(f_pb, pb.astype(np.float32))
register("paper2", "P2_BASELINE_PRED", f_pb)
imp = pd.DataFrame({"feature": FEATS, "gain": bst.feature_importance("gain")}).sort_values("gain", ascending=False)
log(f"baseline CV {np.mean(fold_auc):.4f} | held-out AUROC {roc_auc_score(yR[rows], pb):.4f}")

# ---------------------------------------------------------------- held-out arrays
y = yR[rows].astype(int)
pf = np.load(PRED_FULL_CORR).astype(float).ravel(); pr = np.load(PRED_REDUCED_CORR).astype(float).ravel()
assert abs(roc_auc_score(y, pf) - 0.80350) < 5e-5
n7h = np.rint(n7[rows]).astype(int); red = red_all[rows]; pid = subj[rows]; N = len(rows)
SC = {"full": pf, "no_both": pr, "baseline": pb}
GROUPS = {"all": np.ones(N, bool), "pool0_4": n7h <= 4, "pool0_4_nored": (n7h <= 4) & ~red}
for k in range(5):
    GROUPS[f"s{k}"] = n7h == k; GROUPS[f"s{k}_nored"] = (n7h == k) & ~red

# ---------------------------------------------------------------- paired bootstrap
def prep(s, yy):
    o = np.argsort(s, kind="mergesort"); ss = s[o]
    return o, np.r_[0, np.cumsum(ss[1:] != ss[:-1])], yy[o].astype(float)
def wauc(pre, wt):
    o, gid, yo = pre; wo = wt[o]; G = gid[-1] + 1
    pos = np.bincount(gid, weights=wo * yo, minlength=G); neg = np.bincount(gid, weights=wo * (1 - yo), minlength=G)
    den = pos.sum() * neg.sum()
    return np.nan if den == 0 else float(np.sum(pos * (np.cumsum(neg) - neg + 0.5 * neg)) / den)
pre = {(g, m): prep(SC[m][msk], y[msk]) for g, msk in GROUPS.items() for m in SC}
def stats(wt): return {(g, m): wauc(p, wt[GROUPS[g]]) for (g, m), p in pre.items()}
codes, _ = pd.factorize(pid); P = codes.max() + 1
point = stats(np.ones(N)); keys = list(point); boot = np.empty((B, len(keys)))
rng = np.random.default_rng(SEED)
for bi in range(B):
    wt = np.bincount(rng.integers(0, P, P), minlength=P).astype(float)[codes]
    r = stats(wt); boot[bi] = [r[k] for k in keys]
    if (bi + 1) % 250 == 0: log(f"bootstrap {bi+1}/{B}")
bd = {k: boot[:, i] for i, k in enumerate(keys)}
Q = []
def add(name, g, p, b):
    Q.append({"quantity": name, "group": g, "point": p, "lo95": np.nanpercentile(b, 2.5), "hi95": np.nanpercentile(b, 97.5),
              "lo99": np.nanpercentile(b, 0.5), "hi99": np.nanpercentile(b, 99.5)})
for (g, m) in keys: add(f"auroc {m}", g, point[(g, m)], bd[(g, m)])
for g in GROUPS:
    for a, b in [("full", "baseline"), ("no_both", "baseline"), ("full", "no_both")]:
        add(f"auroc {a} - {b}", g, point[(g, a)] - point[(g, b)], bd[(g, a)] - bd[(g, b)])
Qd = pd.DataFrame(Q)
sizes = pd.DataFrame([{"group": g, "windows": int(m.sum()), "patients": int(len(np.unique(pid[m]))), "positives": int(y[m].sum()),
                       "prevalence": float(y[m].mean()), "red_score_windows": int((m & red).sum()),
                       "red_share": float((m & red).sum() / max(m.sum(), 1))} for g, m in GROUPS.items()])

# ---------------------------------------------------------------- logit-Platt sensitivity (full model)
def logit(p): p = np.clip(p, 1e-7, 1 - 1e-7); return np.log(p / (1 - p))
def calib(x, yy):
    a = 0.0
    for _ in range(8):
        mu = 1 / (1 + np.exp(-(a + x))); a += np.sum(yy - mu) / np.sum(mu * (1 - mu))
    b = np.array([0.0, 1.0])
    for _ in range(8):
        mu = 1 / (1 + np.exp(-(b[0] + b[1] * x))); v = mu * (1 - mu); r = yy - mu
        H = np.array([[v.sum(), (v * x).sum()], [(v * x).sum(), (v * x * x).sum()]])
        b = b + np.linalg.solve(H, np.array([r.sum(), (r * x).sum()]))
    return a, b[1]
oof_f = np.concatenate([np.load(os.path.join(PVC, f"full_cvfold{k}_probs.npy")) for k in range(1, 6)]).astype(float)
oofy_f = np.concatenate([np.load(os.path.join(PVC, f"full_cvfold{k}_yval.npy")) for k in range(1, 6)]).astype(int)
raw_f = np.load(os.path.join(PVC, "full_holdout_probs_RAW.npy")).astype(float)
lp = LogisticRegression(C=1e10, max_iter=1000).fit(logit(oof_f).reshape(-1, 1), oofy_f)
pf_lp = lp.predict_proba(logit(raw_f).reshape(-1, 1))[:, 1]
cal_rows, dec_rows = [], []
for name, p in [("Platt on probability (primary)", pf), ("Platt on logit (sensitivity)", pf_lp)]:
    a, s = calib(logit(p), y)
    cal_rows.append({"method": name, "brier": float(np.mean((p - y) ** 2)), "cal_intercept": a, "cal_slope": s,
                     "mean_predicted": float(p.mean()), "observed": float(y.mean())})
    d = pd.qcut(p, 10, labels=False, duplicates="drop")
    t = pd.DataFrame({"p": p, "y": y, "d": d}).groupby("d").agg(n=("y", "size"), mean_predicted=("p", "mean"), observed=("y", "mean")).reset_index()
    t.insert(0, "method", name); dec_rows.append(t)
CAL = pd.DataFrame(cal_rows); DEC = pd.concat(dec_rows, ignore_index=True)

# ---------------------------------------------------------------- save and print
for fn, df, key in [("P2_baseline_bootstrap.csv", Qd, "P2_BASELINE_BOOTSTRAP"), ("P2_baseline_set_sizes.csv", sizes, "P2_BASELINE_SET_SIZES"),
                    ("P2_baseline_importance.csv", imp, "P2_BASELINE_IMPORTANCE"), ("P2_calibration_logitplatt.csv", CAL, "P2_CAL_LOGITPLATT"),
                    ("P2_calibration_logitplatt_deciles.csv", DEC, "P2_CAL_LOGITPLATT_DECILES")]:
    f = os.path.join(OUT, fn); df.to_csv(f, index=False); register("paper2", key, f)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(f"\nbaseline CV fold AUROC: {np.round(fold_auc, 4).tolist()}")
print("\n=== baseline feature importance (gain) ===\n", imp.to_string(index=False))
print("\n=== groups: sizes and red scores ===\n", sizes.round(4).to_string(index=False))
print("\n=== bootstrap ===\n", Qd.round(4).to_string(index=False))
print("\n=== calibration: primary vs logit-Platt ===\n", CAL.round(4).to_string(index=False))
print("\n", DEC.round(4).to_string(index=False))
log("done")
