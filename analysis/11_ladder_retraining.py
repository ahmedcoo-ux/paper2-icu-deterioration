# Source: paper2.ipynb, code cells [14] (outputs not included)

# %% ---- cell 14 ----
# ============================================================================
# Corrected inputs, step 2 (v4): retraining ladder on leak-free, numerically stable features
#   Identical to run_ablation_consolidated.ipynb (LightGBM 300 trees, lr 0.05, 63 leaves, min_child 50, colsample 0.8,
#   alpha 0.1, lambda 1.0, max_bin 63, scale_pos_weight = neg/pos, seed 42, no bagging; refit on all development rows;
#   Platt on pooled out-of-fold predictions).
#   Folds: GroupKFold(5) on ALL development rows (original labels) — asserted identical to the valid run by its saved
#   per-fold validation labels — then restricted to kept rows.
#   Experiments:
#     leakfix   : X_full_v3,          original label, all windows         -> full, no_both
#     labelonly : X_full_v3,          repaired label, rebuilt gate        -> full
#     corrected : X_full_v3_repaired, repaired label, rebuilt gate        -> full, no_both, full_avail48
#   Per experiment: count assertions, per-fold balance, NEWS-2_6 / NEWS-2_7 comparators on its own held-out set.
# Output: <PHD_ROOT>\ewp_results_v3\<experiment>\...
# ============================================================================
import numpy as np, pandas as pd, lightgbm as lgb, sklearn, time, gc, json, os, psutil
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

PHD = r"<PHD_ROOT>"
N_TOTAL, N_FEATURES, SEED, CHUNK = 3_498_146, 223, 42, 200_000
BASE = Path(rf"{PHD}\ewp_results_v3")
VALID_RUN = Path(rf"{PHD}\ewp_results\prediction_vectors")
EXPERIMENTS = [
    {"name": "leakfix",   "X": rf"{PHD}\s7_rebuilt\X_full_v3.npy",          "label": "original", "gate": False, "variants": ["full", "no_both"]},
    {"name": "labelonly", "X": rf"{PHD}\s7_rebuilt\X_full_v3.npy",          "label": "repaired", "gate": True,  "variants": ["full"]},
    {"name": "corrected", "X": rf"{PHD}\s7_rebuilt\X_full_v3_repaired.npy", "label": "repaired", "gate": True,  "variants": ["full", "no_both", "full_avail48"]},
]
LGB_PARAMS_BASE = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, max_depth=-1,
                       min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=0.1, reg_lambda=1.0, n_jobs=4, verbose=-1, max_bin=63, force_row_wise=True)
_proc = psutil.Process(os.getpid())
def mem(): return f"RSS {_proc.memory_info().rss/1e9:4.1f}GB / free {psutil.virtual_memory().available/1e9:4.1f}GB"

windows = pd.read_parquet(rf"{PHD}\PHD Research\DATAset\s4\windows.parquet", columns=["window_id", "subject_id", "label"])
windows = windows.sort_values("window_id").reset_index(drop=True)
subject_groups = windows["subject_id"].values; y_orig = windows["label"].values.astype(np.int8); del windows; gc.collect()
y_rep = np.load(BASE / "y_repaired.npy"); keep = np.load(BASE / "keep_repaired.npy")
obs_avail48 = np.load(BASE / "gcs_obs_rate_avail48.npy")
N6 = {"original": np.load(BASE / "news2_6_original_we.npy"), "repaired": np.load(BASE / "news2_6_repaired_we.npy")}
N7 = {"original": np.load(BASE / "news7_original.npy"), "repaired": np.load(BASE / "news7_repaired.npy")}
summ1 = json.load(open(BASE / "corrected_inputs_summary.json"))
train_all = np.load(rf"{PHD}\temporal_holdout\train_s4_rows.npy"); hold_all = np.load(rf"{PHD}\temporal_holdout\holdout_s4_rows.npy")
assert len(train_all) == 2_799_482 and len(hold_all) == 698_664

# folds: must reproduce the valid run exactly (its saved per-fold validation labels, in order)
splits_all = list(GroupKFold(n_splits=5).split(np.zeros(len(train_all)), y_orig[train_all], groups=subject_groups[train_all]))
for kf, (_, va) in enumerate(splits_all, 1):
    saved = np.load(VALID_RUN / f"full_cvfold{kf}_yval.npy")
    assert len(saved) == len(va) and (saved == y_orig[train_all[va]]).all(), f"fold {kf} differs from the valid run"
print(f"folds identical to the valid run (sklearn {sklearn.__version__}, lightgbm {lgb.__version__})", flush=True)

feat_names = pd.read_csv(rf"{PHD}\shap\feature_cols.csv")["0"].tolist()
hrs_since_cols = [i for i, f in enumerate(feat_names) if "hrs_since" in f.lower()]
obs_flag_cols = [i for i, f in enumerate(feat_names) if any(x in f.lower() for x in ["_obs_rate", "_miss_rate", "obs_rate", "miss_rate"])]
monitoring_cols = sorted(set(hrs_since_cols + obs_flag_cols))
clinical_cols = [i for i in range(N_FEATURES) if i not in monitoring_cols]
assert len(hrs_since_cols) == 16 and len(obs_flag_cols) == 30 and len(clinical_cols) == 177
J_OBS = feat_names.index("gcs_obs_rate")
VARIANT_DEF = {"full": (list(range(N_FEATURES)), False), "no_both": (clinical_cols, False), "full_avail48": (list(range(N_FEATURES)), True)}

for ex in EXPERIMENTS:
    print(f"\n{'#'*80}\nEXPERIMENT {ex['name']}  ({Path(ex['X']).name}, {ex['label']} label, gate {'rebuilt' if ex['gate'] else 'original'})\n{'#'*80}", flush=True)
    X_full = np.memmap(ex["X"], dtype=np.float32, mode="r", shape=(N_TOTAL, N_FEATURES))
    y_full = y_rep if ex["label"] == "repaired" else y_orig
    k = keep if ex["gate"] else np.ones(N_TOTAL, bool)
    cv_rows = [(train_all[tr][k[train_all[tr]]], train_all[va][k[train_all[va]]]) for tr, va in splits_all]
    for tr_r, va_r in cv_rows:
        assert len(set(subject_groups[tr_r]) & set(subject_groups[va_r])) == 0
    train_rows = train_all[k[train_all]]; holdout_rows = hold_all[k[hold_all]]; y_holdout = y_full[holdout_rows]
    if ex["gate"]:
        assert len(holdout_rows) == summ1["holdout_kept"] == 658_615 and int(y_holdout.sum()) == summ1["holdout_pos_kept"] == 128_844
        assert len(train_rows) == summ1["train_kept"], "train set differs from Script 1"
    else:
        assert len(holdout_rows) == 698_664 and len(train_rows) == 2_799_482
    OUT = BASE / ex["name"]; PROB_DIR = OUT / "prediction_vectors"; PROB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "holdout_rows.npy", holdout_rows)
    OUT_JSON = OUT / "FINAL_results.json"
    results = json.load(open(OUT_JSON)) if OUT_JSON.exists() else {}
    results["_meta"] = {"sklearn": sklearn.__version__, "lightgbm": lgb.__version__, "train_rows": int(len(train_rows)),
                        "holdout_rows": int(len(holdout_rows)), "holdout_positives": int(y_holdout.sum()),
                        "folds": [{"rows": int(len(va)), "positives": int(y_full[va].sum()), "prevalence": float(y_full[va].mean())} for _, va in cv_rows]}
    n7 = N7[ex["label"]][holdout_rows]; ok7 = ~np.isnan(n7)
    results["_comparators"] = {"news2_6_at_window_end_auroc": float(roc_auc_score(y_holdout, N6[ex["label"]][holdout_rows])),
                               "news2_7_at_window_end_auroc": float(roc_auc_score(y_holdout[ok7], n7[ok7]))}
    print(f"train {len(train_rows):,} | holdout {len(holdout_rows):,} (pos {y_holdout.sum():,}) | comparators {results['_comparators']}", flush=True)
    for kf, f in enumerate(results["_meta"]["folds"], 1): print(f"  fold {kf}: {f}", flush=True)
    json.dump(results, open(OUT_JSON, "w"), indent=2)

    def load_rows(row_idx, cols, swap):
        cols = np.asarray(cols, dtype=np.int64); out = np.empty((len(row_idx), len(cols)), dtype=np.float32)
        pos_obs = np.where(cols == J_OBS)[0]
        for s in range(0, len(row_idx), CHUNK):
            e = min(s + CHUNK, len(row_idx)); r = row_idx[s:e]; blk = X_full[r, :][:, cols]
            if swap and len(pos_obs): blk[:, pos_obs[0]] = obs_avail48[r]
            out[s:e, :] = blk; del blk
        return out
    def fit_lgb(train_idx, cols, swap):
        params = dict(LGB_PARAMS_BASE); n_est = params.pop("n_estimators"); max_bin = params.pop("max_bin")
        X_tr = load_rows(train_idx, cols, swap); y_tr = y_full[train_idx]
        spw = (y_tr == 0).sum() / (y_tr == 1).sum()
        ds = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True, params={"max_bin": max_bin, "feature_pre_filter": False})
        ds.construct(); del X_tr; gc.collect()
        params.update({"scale_pos_weight": spw, "objective": "binary", "metric": "auc", "seed": SEED,
                       "bagging_seed": SEED, "feature_fraction_seed": SEED})
        booster = lgb.train(params, ds, num_boost_round=n_est); del ds; gc.collect()
        return booster
    def predict(booster, idx_, cols, swap):
        out = np.empty(len(idx_), dtype=np.float32)
        for s in range(0, len(idx_), CHUNK):
            e = min(s + CHUNK, len(idx_)); out[s:e] = booster.predict(load_rows(idx_[s:e], cols, swap))
        return out

    for vname in ex["variants"]:
        cols, swap = VARIANT_DEF[vname]
        print(f"\n{'='*70}\n{ex['name']} | {vname} — {len(cols)} features\n{'='*70}", flush=True)
        vres = results.get(vname, {"cv_folds": []})
        while len(vres["cv_folds"]) < 5:
            kf = len(vres["cv_folds"]); tr_rows, val_rows = cv_rows[kf]
            t0 = time.time(); booster = fit_lgb(tr_rows, cols, swap); probs = predict(booster, val_rows, cols, swap); y_val = y_full[val_rows]
            np.save(PROB_DIR / f"{vname}_cvfold{kf+1}_probs.npy", probs); np.save(PROB_DIR / f"{vname}_cvfold{kf+1}_yval.npy", y_val)
            np.save(PROB_DIR / f"{vname}_cvfold{kf+1}_rows.npy", val_rows)
            vres["cv_folds"].append({"auroc": float(roc_auc_score(y_val, probs)), "auprc": float(average_precision_score(y_val, probs)),
                                     "rows": int(len(val_rows)), "prevalence": float(y_val.mean())})
            print(f"  fold {kf+1}: AUROC={vres['cv_folds'][-1]['auroc']:.4f} rows={len(val_rows):,} prev={y_val.mean():.4f} "
                  f"({(time.time()-t0)/60:.1f} min) {mem()}", flush=True)
            del booster, probs; gc.collect(); results[vname] = vres; json.dump(results, open(OUT_JSON, "w"), indent=2)
        if "holdout" not in vres:
            t0 = time.time(); booster_full = fit_lgb(train_rows, cols, swap); probs_ho = predict(booster_full, holdout_rows, cols, swap)
            np.save(PROB_DIR / f"{vname}_holdout_probs_RAW.npy", probs_ho)
            vres["holdout"] = {"auroc": float(roc_auc_score(y_holdout, probs_ho)), "auprc": float(average_precision_score(y_holdout, probs_ho))}
            print(f"  HOLDOUT: AUROC={vres['holdout']['auroc']:.4f} AUPRC={vres['holdout']['auprc']:.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)
            oof_p = np.concatenate([np.load(PROB_DIR / f"{vname}_cvfold{j}_probs.npy") for j in range(1, 6)])
            oof_y = np.concatenate([np.load(PROB_DIR / f"{vname}_cvfold{j}_yval.npy") for j in range(1, 6)])
            platt = LogisticRegression(C=1e10, max_iter=1000).fit(oof_p.reshape(-1, 1), oof_y)
            probs_cal = platt.predict_proba(probs_ho.reshape(-1, 1))[:, 1]
            np.save(PROB_DIR / f"{vname}_holdout_probs_CALIBRATED.npy", probs_cal.astype(np.float32))
            vres["calibration"] = {"platt_a": float(platt.coef_[0][0]), "platt_b": float(platt.intercept_[0]), "n_oof": int(len(oof_y))}
            print(f"  calibrated: mean {probs_cal.mean():.4f} (observed {y_holdout.mean():.4f})", flush=True)
            del booster_full, probs_ho, oof_p, oof_y, probs_cal; gc.collect(); results[vname] = vres; json.dump(results, open(OUT_JSON, "w"), indent=2)
    del X_full; gc.collect()

print(f"\n{'='*80}\nLADDER (holdout AUROC) — original valid run: full 0.8037, no_both 0.7985, delta 0.0052\n{'='*80}")
for ex in EXPERIMENTS:
    r = json.load(open(BASE / ex["name"] / "FINAL_results.json"))
    print(f"{ex['name']:10s} comparators: {r['_comparators']}")
    for v in ex["variants"]:
        d = r[v]; a = np.array([f["auroc"] for f in d["cv_folds"]])
        print(f"{ex['name']:10s} {v:14s} CV {a.mean():.4f} ± {a.std(ddof=1):.4f}   holdout {d['holdout']['auroc']:.4f}")
    if "full" in r and "no_both" in r:
        print(f"{ex['name']:10s} delta (full - no_both) holdout: {r['full']['holdout']['auroc'] - r['no_both']['holdout']['auroc']:.4f}")
