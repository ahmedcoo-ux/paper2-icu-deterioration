# Source: paper2.ipynb, code cells [13] (outputs not included)

# %% ---- cell 13 ----
# ============================================================================
# Corrected inputs, step 1 (v6.2): leak-free, numerically stable window features + consciousness repair + labels/gate/strata
#   Changes from v6.1: truncation test deletes hours >= window_end (features are defined on [window_start, window_end));
#   check (b1) requires near-exact agreement for same-definition slope/std features, reported overall and for early
#   windows (window_start < 48 h, where float32 rounding cannot matter); computed features stored as float32.
#   Rebuilt per window over hours [window_start, window_end) (5 hours, as built), computed in float64:
#     40 stay-denominator features + 3 constant GCS cross_pct -> window denominators (cross/pct: / observed hours; miss/obs: / 5)
#     28 slope features (x = hours from window start) + 12 std features (np.nanstd, ddof = 0, as the original code)
#   Checks (nothing written unless all pass):
#     (a)  leaky formula (window / stay hours) reproduces X_full for the 40 features (>= 99%, outside split stays)
#     (b1) same-definition slopes/std (vitals, labs; all std): agreement with X_full in early windows >= 99%
#     (b2) chart slopes (FiO2, GCS components): original row-position definition reproduces X_full (r >= 0.99);
#          written values use x = hours (stated correction)
#     (c)  consciousness value features (4 h) reproduce X_full (>= 99%)
#     (d)  truncation test on the new code: hours >= window_end deleted for 3,000 windows -> 0 features change
# Output: <PHD_ROOT>\ewp_results_v3\*  and  <PHD_ROOT>\s7_rebuilt\X_full_v3*.npy
# ============================================================================
import os, gc, json, time, warnings
import numpy as np, pandas as pd, pyarrow.parquet as pq
warnings.filterwarnings("ignore", category=RuntimeWarning)

PHD = r"<PHD_ROOT>"
WINDOWS = rf"{PHD}\PHD Research\DATAset\s4\windows.parquet"
HOURLY = rf"{PHD}\PHD Research\DATAset\s3\hourly_signals_with_labs.parquet"
CE_SUBSET = rf"{PHD}\PHD Research\DATAset\s3\chartevents_subset.parquet"
ICUSTAYS = rf"{PHD}\PHD Research\DATAset\mimic-iv-3.1\icu\icustays.csv"
LABELS = rf"{PHD}\SAL\01_feasibility\organ_support_window_labels.npz"
X_IN = rf"{PHD}\s7_rebuilt\X_full.npy"
X_V3 = rf"{PHD}\s7_rebuilt\X_full_v3.npy"
X_V3R = rf"{PHD}\s7_rebuilt\X_full_v3_repaired.npy"
OUT = rf"{PHD}\ewp_results_v3"; os.makedirs(OUT, exist_ok=True)
FEAT_NAMES = pd.read_csv(rf"{PHD}\shap\feature_cols.csv")["0"].tolist()
N_TOTAL, N_FEAT, HZ, K, CHUNK, NW, N_TRUNC, EARLY_H = 3_498_146, 223, 6, 100_000, 200_000, 5, 3000, 48
FINAL_CAPS = {"rr": 4, "spo2": 4, "sbp": 4, "hr": 8, "temp": 4, "cons": 48}
S4_CAPS = {c: 4 for c in FINAL_CAPS}
VITALS = ["hr", "sbp", "dbp", "map", "rr", "spo2", "temp", "urine_output"]
LABS = ["lactate", "ph", "pco2", "po2", "creatinine", "bun", "potassium", "sodium", "glucose", "wbc", "hemoglobin",
        "hematocrit", "platelets", "bilirubin", "bicarbonate", "albumin"]
NEWS2_RANGE = {"hr": (51, 90), "sbp": (111, 219), "rr": (12, 20), "spo2": (96, 100), "temp": (36.1, 38.0)}
PCT = {"spo2_pct_below94": ("spo2", lambda v: v < 94), "rr_pct_abnormal": ("rr", lambda v: (v < 12) | (v > 25)),
       "sbp_pct_below100": ("sbp", lambda v: v < 100), "hr_pct_abnormal": ("hr", lambda v: (v < 51) | (v > 110)),
       "temp_pct_abnormal": ("temp", lambda v: (v < 36.1) | (v > 38.0))}
CE_IDS = {220739: "gcs_eye", 223900: "gcs_verbal", 223901: "gcs_motor", 223835: "fio2",
          224385: "intubated", 227021: "intubated", 228096: "sedated"}
CHART_SIG = ["gcs_eye", "gcs_verbal", "gcs_motor", "fio2"]
GCS_COMP_RANGE = {"gcs_eye": (1, 4), "gcs_verbal": (1, 5), "gcs_motor": (1, 6)}
CONS_VALUE = ["gcs_mean", "gcs_last", "avpu_sum"]
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
HS_COLS = ["stay_id", "hour_offset"] + sorted(set(VITALS + LABS + [c for v in COMP.values() for c in v[:2]]))

def slope64(M, Xpos=None):
    x = np.broadcast_to(np.arange(M.shape[1], dtype=np.float64), M.shape) if Xpos is None else Xpos.astype(np.float64)
    valid = ~np.isnan(M) & ~np.isnan(x); n_v = valid.sum(1).astype(np.float64)
    xm = np.where(valid, x, 0.0); m = np.where(valid, M, 0.0)
    sx = xm.sum(1); sy = m.sum(1); sx2 = (xm ** 2).sum(1); sxy = (xm * m).sum(1)
    den = n_v * sx2 - sx ** 2
    return np.where(n_v < 2, 0.0, np.where(den != 0, (n_v * sxy - sx * sy) / np.where(den != 0, den, 1.0), 0.0))

def compute_features(hs, ce_h, sid, ws, we, with_leaky):
    hs = hs.sort_values(["stay_id", "hour_offset"]).reset_index(drop=True)
    assert (hs.groupby("stay_id").hour_offset.diff().dropna() == 1).all(), "hour_offset not contiguous"
    key = hs.stay_id.to_numpy() * K + hs.hour_offset.to_numpy(); idx = pd.Index(key); assert idx.is_unique
    grp = hs.stay_id.to_numpy()
    PW = [idx.get_indexer(sid * K + ws + j) for j in range(NW)]
    win = lambda arr: np.column_stack([np.where(p >= 0, arr[np.maximum(p, 0)], np.nan) for p in PW])
    charted = {c: hs[vc].where(hs[oc].fillna(0).astype(int) == 1) for c, (vc, oc, f) in COMP.items()}
    ffill = lambda c, cap: (charted[c].groupby(grp).ffill() if cap is None else charted[c].groupby(grp).ffill(limit=cap)).to_numpy(float)
    F, LK = {}, {}
    nr = hs.groupby("stay_id").size().reindex(sid).to_numpy(float) if with_leaky else None
    for v in VITALS + LABS:
        M = win(hs[v].to_numpy(float)); avail = (~np.isnan(M)).sum(1)
        F[f"{v}_miss_rate"] = 1 - avail / NW; F[f"{v}_slope"] = slope64(M)
        if v in VITALS: F[f"{v}_std"] = np.nanstd(M, axis=1)              # ddof = 0, as np.nanstd in the original code
        if with_leaky: LK[f"{v}_miss_rate"] = 1 - avail / nr
    for v, (lo, hi) in NEWS2_RANGE.items():
        M = win(hs[v].to_numpy(float)); avail = (~np.isnan(M)).sum(1); cond = ((M < lo) | (M > hi)).sum(1)
        F[f"{v}_cross_pct"] = np.where(avail > 0, cond / np.maximum(avail, 1), np.nan)
        if with_leaky: LK[f"{v}_cross_pct"] = cond / nr
    for name, (v, fn) in PCT.items():
        M = win(hs[v].to_numpy(float)); avail = (~np.isnan(M)).sum(1); cond = fn(M).sum(1)
        F[name] = np.where(avail > 0, cond / np.maximum(avail, 1), np.nan)
        if with_leaky: LK[name] = cond / nr
    num = ce_h[ce_h.signal.isin(CHART_SIG)].groupby(["stay_id", "hour_offset", "signal"]).valuenum.mean().unstack("signal")
    ce_hours = ce_h[["stay_id", "hour_offset"]].drop_duplicates()
    ce_idx = pd.Index(np.sort(ce_hours.stay_id.to_numpy() * K + ce_hours.hour_offset.to_numpy()))
    n_ce = pd.Series(ce_hours.stay_id.to_numpy()).value_counts().reindex(sid).to_numpy(float)
    nkey = num.index.get_level_values(0).to_numpy() * K + num.index.get_level_values(1).to_numpy()
    PC = [ce_idx.get_indexer(sid * K + ws + j) for j in range(NW)]
    cewin = lambda arr: np.column_stack([np.where(p >= 0, arr[np.maximum(p, 0)], np.nan) for p in PC])
    POSM = np.column_stack([np.where(p >= 0, p, np.nan) for p in PC]).astype(float)
    if len(POSM): POSM = POSM - np.nanmin(np.where(np.isnan(POSM), np.inf, POSM), axis=1, keepdims=True)
    for s in CHART_SIG:
        arr = pd.Series(num[s].to_numpy(float) if s in num.columns else np.full(len(num), np.nan), index=nkey).reindex(ce_idx).to_numpy(float)
        M = cewin(arr); avail = (~np.isnan(M)).sum(1)
        F[f"{s}_miss_rate"] = 1 - avail / NW
        F[f"{s}_slope"] = slope64(M)
        F[f"{s}_slope_rowpos"] = slope64(M, POSM)
        F[f"{s}_std"] = np.nanstd(M, axis=1)
        if with_leaky: LK[f"{s}_miss_rate"] = np.where(np.isnan(n_ce), np.nan, 1 - avail / n_ce)
        if s in GCS_COMP_RANGE:
            lo, hi = GCS_COMP_RANGE[s]; cond = ((M < lo) | (M > hi)).sum(1)
            F[f"{s}_cross_pct"] = np.where(avail > 0, cond / np.maximum(avail, 1), np.nan)
        else:
            F["fio2_obs_rate"] = avail / NW
            if with_leaky: LK["fio2_obs_rate"] = np.where(np.isnan(n_ce), np.nan, avail / n_ce)
    g4 = ffill("cons", 4); G4 = win(g4); avail4 = (~np.isnan(G4)).sum(1)
    F["gcs_obs_rate_v3"] = avail4 / NW
    if with_leaky: LK["gcs_obs_rate"] = avail4 / nr
    g48 = ffill("cons", FINAL_CAPS["cons"]); G48 = win(g48)
    F["gcs_obs_rate_rep"] = np.nansum(win(charted["cons"].notna().to_numpy().astype(float)), axis=1) / NW
    F["gcs_obs_rate_avail48"] = (~np.isnan(G48)).sum(1) / NW
    for tag, G, gv in [("4", G4, g4), ("48", G48, g48)]:
        A = win(s_cons(gv).astype(float)); rev = G[:, ::-1]; has = ~np.all(np.isnan(G), axis=1)
        F[f"gcs_mean_{tag}"] = np.nanmean(G, axis=1)
        F[f"gcs_last_{tag}"] = np.where(has, rev[np.arange(len(G)), np.argmax(~np.isnan(rev), axis=1)], np.nan)
        F[f"avpu_sum_{tag}"] = np.nansum(A, axis=1)
    return ({k: np.asarray(v, np.float32) for k, v in F.items()}, {k: np.asarray(v, np.float32) for k, v in LK.items()})

# ---------------------------------------------------------------- windows, splits, news7
W = pd.read_parquet(WINDOWS, columns=["window_id", "stay_id", "window_start", "window_end", "label"])
assert len(W) == N_TOTAL and W.window_id.is_monotonic_increasing
sid = W.stay_id.to_numpy().astype(np.int64); ws = W.window_start.to_numpy(np.int64); we = W.window_end.to_numpy(np.int64)
assert (we - ws == NW).all()
y_s4 = W.label.to_numpy().astype(np.int8); del W; gc.collect()
train_rows = np.load(rf"{PHD}\temporal_holdout\train_s4_rows.npy"); holdout_rows = np.load(rf"{PHD}\temporal_holdout\holdout_s4_rows.npy")
is_dev = np.zeros(N_TOTAL, bool); is_dev[train_rows] = True
is_ho = np.zeros(N_TOTAL, bool); is_ho[holdout_rows] = True
L = np.load(LABELS, allow_pickle=True); fr_ = L["file_row"].astype(np.int64)
news7 = np.full(N_TOTAL, np.nan); news7[fr_] = L["news7"].astype(float)
split_all = np.full(N_TOTAL, "", dtype=object); split_all[fr_] = np.char.lower(L["split"].astype(str)); del L, fr_
assert (is_ho == (split_all == "held-out")).all()
stays = np.unique(sid)
pf = pq.ParquetFile(HOURLY); seen = {}
for i in range(pf.metadata.num_row_groups):
    for s in np.unique(pf.read_row_group(i, columns=["stay_id"]).column(0).to_numpy()):
        seen[int(s)] = seen.get(int(s), 0) + 1
in_split = np.isin(sid, np.array([s for s, n in seen.items() if n > 1], dtype=np.int64))
early = ws < EARLY_H

# ---------------------------------------------------------------- data
hs = pd.read_parquet(HOURLY, columns=HS_COLS)
hs = hs[hs.stay_id.astype(np.int64).isin(stays)].copy()
hs["stay_id"], hs["hour_offset"] = hs.stay_id.astype(np.int64), hs.hour_offset.astype(np.int64)
hs = hs.sort_values(["stay_id", "hour_offset"]).reset_index(drop=True)
icu = pd.read_csv(ICUSTAYS, usecols=["stay_id", "intime"], parse_dates=["intime"])
ce = pd.read_parquet(CE_SUBSET, columns=["stay_id", "charttime", "itemid", "valuenum"])
ce = ce[ce.itemid.isin(list(CE_IDS)) & ce.stay_id.isin(stays)].copy()
ce["stay_id"] = ce.stay_id.astype(np.int64)
ce["hour_offset"] = ((pd.to_datetime(ce.charttime) - ce.stay_id.map(icu.set_index("stay_id").intime)).dt.total_seconds() / 3600).round().astype(int)
ce = ce[ce.hour_offset >= 0]; ce["signal"] = ce.itemid.map(CE_IDS)
ce_h = ce[["stay_id", "hour_offset", "signal", "valuenum"]].reset_index(drop=True); del ce; gc.collect()
log("data loaded")

# ---------------------------------------------------------------- labels / gate / strata
key = hs.stay_id.to_numpy() * K + hs.hour_offset.to_numpy(); idx = pd.Index(key); grp = hs.stay_id.to_numpy()
charted_l = {c: hs[vc].where(hs[oc].fillna(0).astype(int) == 1) for c, (vc, oc, f) in COMP.items()}
def ffill_cap(c, cap):
    return (charted_l[c].groupby(grp).ffill() if cap is None else charted_l[c].groupby(grp).ffill(limit=cap)).to_numpy(float)
def total_score(caps): return sum(COMP[c][2](ffill_cap(c, cap)) for c, cap in caps.items()).astype(float)
def at_we(arr, k):
    p = idx.get_indexer(sid * K + we + k); return np.where(p >= 0, arr[np.maximum(p, 0)], np.nan)
def score_gate_label(t):
    s_we = np.nan_to_num(at_we(t, 0), nan=0); lab = np.zeros(N_TOTAL, bool)
    for k in range(1, HZ + 1): lab |= np.nan_to_num(at_we(t, k), nan=-1) >= 7
    return s_we, s_we < 5, lab
s6_orig, g4, l4 = score_gate_label(total_score(S4_CAPS))
assert g4.all() and (l4.astype(np.int8) == y_s4).all(), "S4 gate/label not reproduced — run in a FRESH kernel"
s6_rep, keep, y_rep = score_gate_label(total_score(FINAL_CAPS))
o2 = news7 - s6_orig; o2_ok = np.isin(o2, [0, 2]); assert o2_ok[is_ho].all()
n7_rep = np.where(o2_ok, s6_rep + o2, np.nan)
del charted_l, idx, key, grp; gc.collect()
log(f"labels: held-out kept {keep[is_ho].sum():,}, positives {y_rep[is_ho & keep].sum():,} (expect 658,615 / 128,844)")

# ---------------------------------------------------------------- (d) truncation test on the NEW code: delete hours >= window_end
rng = np.random.default_rng(11); samp = np.sort(rng.choice(N_TOTAL, N_TRUNC, replace=False))
pseudo = (10_000_000 + np.arange(N_TRUNC)).astype(np.int64)
hs_by = {s: g for s, g in hs[hs.stay_id.isin(sid[samp])].groupby("stay_id")}
ce_by = {s: g for s, g in ce_h[ce_h.stay_id.isin(sid[samp])].groupby("stay_id")}
def pseudo_data(truncate):
    H, C = [], []
    for i, r in enumerate(samp):
        h = hs_by[sid[r]]; c = ce_by.get(sid[r], ce_h.iloc[:0])
        if truncate: h = h[h.hour_offset < we[r]]; c = c[c.hour_offset < we[r]]
        H.append(h.assign(stay_id=pseudo[i])); C.append(c.assign(stay_id=pseudo[i]))
    return pd.concat(H, ignore_index=True), pd.concat(C, ignore_index=True)
Ff, _ = compute_features(*pseudo_data(False), pseudo, ws[samp], we[samp], with_leaky=False)
Ft, _ = compute_features(*pseudo_data(True), pseudo, ws[samp], we[samp], with_leaky=False)
trunc_changes = {f: int((~np.isclose(Ff[f], Ft[f], atol=0, rtol=0, equal_nan=True)).sum()) for f in Ff if not f.endswith("_rowpos")}
n_changed = sum(v > 0 for v in trunc_changes.values())
log(f"TRUNCATION TEST (new code, hours >= window_end deleted): {n_changed} of {len(trunc_changes)} rebuilt features change")
if n_changed:
    raise SystemExit(f"truncation test failed: { {f: v for f, v in trunc_changes.items() if v} } — nothing written")
del Ff, Ft, hs_by, ce_by; gc.collect()

# ---------------------------------------------------------------- full computation + reproduction checks
F, LK = compute_features(hs, ce_h, sid, ws, we, with_leaky=True)
del hs, ce_h; gc.collect()
log("features computed")
def imputed(v, rows):
    v = np.asarray(v, float).copy(); m = np.isnan(v)
    if m.any(): v[m] = np.nanmedian(v[rows])
    return v.astype(np.float32)
X = np.memmap(X_IN, dtype=np.float32, mode="r", shape=(N_TOTAL, N_FEAT))
SLOPE_STD = [f for f in F if (f.endswith("_slope") or f.endswith("_std"))]
assert len(SLOPE_STD) == 40, len(SLOPE_STD)
CHART_SLOPES = [f"{s}_slope" for s in CHART_SIG]
SAME_DEF = [f for f in SLOPE_STD if f not in CHART_SLOPES]           # vital/lab slopes + all std (incl. chart std)
check = list(LK) + SLOPE_STD + CONS_VALUE
for f in check: assert f in FEAT_NAMES, f
cidx = [FEAT_NAMES.index(f) for f in check]
Xc = np.empty((N_TOTAL, len(cidx)), dtype=np.float32)
for s in range(0, N_TOTAL, CHUNK):
    e = min(s + CHUNK, N_TOTAL); Xc[s:e] = np.asarray(X[s:e, :])[:, cidx]
allrows = np.ones(N_TOTAL, bool); m_all, m_early = ~in_split, ~in_split & early
rep_rows, fail = [], []
for i, f in enumerate(check):
    x = Xc[:, i].astype(float); row = {"feature": f}
    if f in LK:
        ok_ = np.isclose(x, imputed(LK[f], allrows), atol=1e-4, rtol=1e-5)
        row.update(check="(a) leaky formula", agree_all=float(ok_[m_all].mean())); ok = row["agree_all"] >= 0.99
    elif f in SAME_DEF:
        r = imputed(F[f], allrows).astype(float); ok_ = np.isclose(x, r, atol=1e-4, rtol=1e-3)
        row.update(check="(b1) same definition", agree_all=float(ok_[m_all].mean()), agree_early=float(ok_[m_early].mean()),
                   r=float(np.corrcoef(x[m_all], r[m_all])[0, 1]))
        ok = row["agree_early"] >= 0.99
    elif f in CHART_SLOPES:
        r0 = imputed(F[f"{f}_rowpos"], allrows).astype(float); r1 = imputed(F[f], allrows).astype(float)
        row.update(check="(b2) chart slope, original row-position x", r=float(np.corrcoef(x[m_all], r0[m_all])[0, 1]),
                   r_corrected_hours=float(np.corrcoef(x[m_all], r1[m_all])[0, 1]))
        ok = row["r"] >= 0.99
    else:
        ok_ = np.isclose(x, imputed(F[f"{f}_4"], allrows), atol=1e-4, rtol=1e-5)
        row.update(check="(c) consciousness 4 h", agree_all=float(ok_[m_all].mean())); ok = row["agree_all"] >= 0.99
    row["pass"] = bool(ok); rep_rows.append(row)
    if not ok: fail.append(f)
R = pd.DataFrame(rep_rows); R.to_csv(os.path.join(OUT, "reproduction_checks.csv"), index=False)
pd.set_option("display.width", 220); print(R.round(4).to_string(index=False))
if fail:
    raise SystemExit(f"reproduction checks failed for: {fail} — paste this output; nothing has been written")
del Xc; gc.collect()

# ---------------------------------------------------------------- write corrected matrices
dev, devk = is_dev, is_dev & keep
BASE_FEATS = [f for f in F if f in FEAT_NAMES]
v3_cols = {f: imputed(F[f], dev) for f in BASE_FEATS}
v3_cols["gcs_obs_rate"] = imputed(F["gcs_obs_rate_v3"], dev)
rep_cols = {f: imputed(F[f], devk) for f in BASE_FEATS}
rep_cols["gcs_obs_rate"] = imputed(F["gcs_obs_rate_rep"], devk)
for f in CONS_VALUE: rep_cols[f] = imputed(F[f"{f}_48"], devk)
def write(path, cols_):
    Xo = np.memmap(path, dtype=np.float32, mode="w+", shape=(N_TOTAL, N_FEAT)); jj = {f: FEAT_NAMES.index(f) for f in cols_}
    for s in range(0, N_TOTAL, CHUNK):
        e = min(s + CHUNK, N_TOTAL); blk = np.array(X[s:e, :])
        for f, j in jj.items(): blk[:, j] = cols_[f][s:e]
        Xo[s:e, :] = blk
    Xo.flush(); del Xo
write(X_V3, v3_cols); log(f"written {X_V3}")
write(X_V3R, rep_cols); log(f"written {X_V3R}")

np.save(os.path.join(OUT, "y_repaired.npy"), y_rep.astype(np.int8))
np.save(os.path.join(OUT, "keep_repaired.npy"), keep)
np.save(os.path.join(OUT, "news2_6_original_we.npy"), s6_orig.astype(np.float32))
np.save(os.path.join(OUT, "news2_6_repaired_we.npy"), s6_rep.astype(np.float32))
np.save(os.path.join(OUT, "news7_original.npy"), news7.astype(np.float32))
np.save(os.path.join(OUT, "news7_repaired.npy"), n7_rep.astype(np.float32))
np.save(os.path.join(OUT, "news7_mismatch_mask.npy"), ~o2_ok)
np.save(os.path.join(OUT, "gcs_obs_rate_avail48.npy"), imputed(F["gcs_obs_rate_avail48"], devk))
summ = {"truncation_test_new_code": {"windows": N_TRUNC, "deleted": "hours >= window_end",
                                     "features_tested": len(trunc_changes), "features_changed": n_changed},
        "n_rebuilt": len(BASE_FEATS) + 1, "rebuilt_features": sorted(BASE_FEATS + ["gcs_obs_rate"]),
        "definitions": {"window": "hours [window_start, window_end)", "miss_rate/obs_rate": "/ 5 window hours",
                        "cross_pct/pct": "/ observed window hours", "slope": "float64, x = hours from window start",
                        "std": "np.nanstd, ddof = 0", "chart hours": "rounded to nearest hour (hourly table: floor)"},
        "repaired_extra_features": CONS_VALUE,
        "holdout_kept": int(keep[is_ho].sum()), "holdout_pos_kept": int(y_rep[is_ho & keep].sum()),
        "train_kept": int(keep[is_dev].sum()), "news7_mismatch_development": int((~o2_ok & is_dev).sum())}
json.dump(summ, open(os.path.join(OUT, "corrected_inputs_summary.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in summ.items() if k != "rebuilt_features"}, indent=2))
log("done")
