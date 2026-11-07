# Source: paper2.ipynb, code cells [17, 18, 19] (outputs not included)

# %% ---- cell 17 ----
# ============================================================================
# Paper 2 — table script (corrected model): Table 2, Table 3 additions, S1, R3, R4, R5, admission periods
#   Same held-out set, patient codes, seed and resample order as P2 bootstrap -> intervals are paired with it.
#   Table 2 : development / held-out (all) / held-out primary (7-component NEWS-2 0-4):
#             stays, patients, windows, positives, episodes, windows per stay
#   Table 3 : AUPRC within 7-component strata (full, no_both, delta, prevalence baseline)
#   S1      : 6-component strata 0-4 (all held-out windows): AUROC / AUPRC full, no_both, delta_s, delta_s - delta_global
#   R3      : in-hospital death during the same admission (admission-level): AUROC of the full model within
#             7-component strata and pooled over 0-4; death rate per stratum
#   R4      : calibration within 7-component strata: O:E, calibration intercept, slope, Brier
#   R5      : top 10% of windows by predicted risk within each 7-component stratum: PPV, sensitivity, lift
#   Periods : anchor_year_group of development vs held-out stays
# Output: <PHD_ROOT>\Paper2\corrected\P2_table2_cohort.csv, P2_tables_bootstrap.csv, P2_tables_set_sizes.csv,
#         P2_anchor_year_group.csv, P2_tables_boot_replicates.npy (+ keys)
# ============================================================================
import os, sys, time, gc, json
import numpy as np, pandas as pd
sys.path.insert(0, r"<PHD_ROOT>")
from paths.paper2 import *
from paths.registry import register
from sklearn.metrics import roc_auc_score, average_precision_score

B, SEED = 2000, 42
OUT = P2_OUT
HOSP = os.path.join(os.path.dirname(os.path.dirname(ICUSTAYS)), "hosp")
ADMISSIONS, PATIENTS = os.path.join(HOSP, "admissions.csv"), os.path.join(HOSP, "patients.csv")
t00 = time.time()
def log(m): print(f"[{time.time()-t00:6.0f}s] {m}", flush=True)

# ---------------------------------------------------------------- inputs
rows = np.load(CORRECTED_HO_ROWS); N = len(rows); assert N == 658_615
yA = np.load(Y_REPAIRED).astype(np.int8); keep = np.load(KEEP_REPAIRED)
y = yA[rows].astype(int); assert int(y.sum()) == 128_844
pf = np.load(PRED_FULL_CORR).astype(float).ravel(); pr = np.load(PRED_REDUCED_CORR).astype(float).ravel()
assert len(pf) == N and len(pr) == N
assert abs(roc_auc_score(y, pf) - 0.80350) < 5e-5, "full-model predictions differ from the bootstrap run"
n6 = np.rint(np.load(NEWS2_6_REP)[rows]).astype(int)
n7f = np.load(NEWS7_REP)[rows].astype(float); assert not np.isnan(n7f).any(); n7 = np.rint(n7f).astype(int)
assert n6.min() >= 0 and n6.max() <= 4, "6-component score outside 0-4 in the gated held-out set"
W = pd.read_parquet(WINDOWS, columns=["window_id", "subject_id", "stay_id", "window_start"]).sort_values("window_id")
subj = W.subject_id.to_numpy(np.int64); stay = W.stay_id.to_numpy(np.int64); wstart = W.window_start.to_numpy(np.int64); del W; gc.collect()
train_all, hold_all = np.load(TRAIN_ROWS), np.load(HOLDOUT_ROWS)
assert np.array_equal(rows, hold_all[keep[hold_all]])
dev = train_all[keep[train_all]]; assert len(dev) == 2_647_017
pid = subj[rows]
icu = pd.read_csv(ICUSTAYS, usecols=["stay_id", "hadm_id"])
adm = pd.read_csv(ADMISSIONS, usecols=["hadm_id", "hospital_expire_flag"])
death_by_stay = icu.merge(adm, on="hadm_id", how="left").set_index("stay_id").hospital_expire_flag
dth = death_by_stay.reindex(stay[rows]).to_numpy(float)
assert not np.isnan(dth).any(), "in-hospital death missing for some held-out windows"
dth = dth.astype(int)
log("inputs loaded")

# ---------------------------------------------------------------- Table 2
def episode_ids(r):
    d = pd.DataFrame({"stay": stay[r], "ws": wstart[r], "y": yA[r].astype(int), "i": np.arange(len(r))}).sort_values(["stay", "ws"])
    yy, ss = d.y.to_numpy(), d.stay.to_numpy()
    prev_y = np.r_[0, yy[:-1]]; prev_s = np.r_[-1, ss[:-1]]
    new = (yy == 1) & ~((prev_y == 1) & (prev_s == ss))
    ep = np.where(yy == 1, np.cumsum(new), 0)
    out = np.zeros(len(r), np.int64); out[d.i.to_numpy()] = ep
    return out
def describe(name, r, ep, sel=None):
    s = np.ones(len(r), bool) if sel is None else sel
    rr = r[s]; per = pd.Series(stay[rr]).value_counts().to_numpy()
    q25, q50, q75 = np.percentile(per, [25, 50, 75])
    pos = yA[rr].astype(int)
    return {"set": name, "icu_stays": int(len(np.unique(stay[rr]))), "patients": int(len(np.unique(subj[rr]))),
            "windows": int(len(rr)), "positive_windows": int(pos.sum()), "prevalence": float(pos.mean()),
            "episodes": int(len(np.unique(ep[s & (yA[r] == 1)]))),
            "windows_per_stay_median": float(q50), "windows_per_stay_q25": float(q25), "windows_per_stay_q75": float(q75)}
ep_dev, ep_ho = episode_ids(dev), episode_ids(rows)
T2 = pd.DataFrame([describe("development", dev, ep_dev),
                   describe("held-out (all)", rows, ep_ho),
                   describe("held-out primary (7-component NEWS-2 0-4)", rows, ep_ho, n7 <= 4)])
f_t2 = os.path.join(OUT, "P2_table2_cohort.csv"); T2.to_csv(f_t2, index=False); register("paper2", "P2_TABLE2", f_t2)
log("Table 2 done")

# ---------------------------------------------------------------- admission periods
pat = pd.read_csv(PATIENTS, usecols=["subject_id", "anchor_year_group"]).set_index("subject_id").anchor_year_group
per = []
for name, r in [("development", dev), ("held-out", rows)]:
    st = pd.DataFrame({"stay": stay[r], "subj": subj[r]}).drop_duplicates("stay")
    g = pat.reindex(st.subj.to_numpy()).fillna("missing").value_counts(normalize=True).sort_index()
    per += [{"set": name, "anchor_year_group": k, "share_of_stays": float(v)} for k, v in g.items()]
f_per = os.path.join(OUT, "P2_anchor_year_group.csv"); AP = pd.DataFrame(per); AP.to_csv(f_per, index=False)
register("paper2", "P2_ANCHOR_YEAR_GROUP", f_per)

# ---------------------------------------------------------------- weighted metrics
def prep(s, yy):
    o = np.argsort(s, kind="mergesort"); ss = s[o]
    return o, np.r_[0, np.cumsum(ss[1:] != ss[:-1])], yy[o].astype(float)
def wauc(pre, wt):
    o, gid, yo = pre; wo = wt[o]; G = gid[-1] + 1
    pos = np.bincount(gid, weights=wo * yo, minlength=G); neg = np.bincount(gid, weights=wo * (1 - yo), minlength=G)
    den = pos.sum() * neg.sum()
    return np.nan if den == 0 else float(np.sum(pos * (np.cumsum(neg) - neg + 0.5 * neg)) / den)
def wap(pre_desc, wt):
    o, gid, yo = pre_desc; wo = wt[o]; G = gid[-1] + 1
    pos = np.bincount(gid, weights=wo * yo, minlength=G); neg = np.bincount(gid, weights=wo * (1 - yo), minlength=G)
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    return np.nan if tp[-1] == 0 else float(np.sum(pos * tp / np.maximum(tp + fp, 1e-300)) / tp[-1])
def top10(pre_desc, wt):
    o, _, yo = pre_desc; wo = wt[o]; cw = np.cumsum(wo); flag = (cw - wo) < 0.10 * cw[-1]
    tp = np.sum(wo * yo * flag); fl = np.sum(wo * flag); P_ = np.sum(wo * yo); prev = P_ / cw[-1]
    ppv = tp / fl if fl > 0 else np.nan
    return ppv, (tp / P_ if P_ > 0 else np.nan), (ppv / prev if prev > 0 else np.nan)
def logit(p): p = np.clip(p, 1e-7, 1 - 1e-7); return np.log(p / (1 - p))
def calib(x, yy, w):
    a = 0.0
    for _ in range(8):
        mu = 1 / (1 + np.exp(-(a + x))); a += np.sum(w * (yy - mu)) / np.sum(w * mu * (1 - mu))
    b = np.array([0.0, 1.0])
    for _ in range(8):
        mu = 1 / (1 + np.exp(-(b[0] + b[1] * x))); v = w * mu * (1 - mu); r = w * (yy - mu)
        H = np.array([[v.sum(), (v * x).sum()], [(v * x).sum(), (v * x * x).sum()]])
        b = b + np.linalg.solve(H, np.array([r.sum(), (r * x).sum()]))
    return a, b[1]

# ---------------------------------------------------------------- groups and prepared orders
SC = {"full": pf, "no_both": pr}
G7 = {f"n7_s{k}": n7 == k for k in range(5)}
G6 = {f"n6_s{k}": n6 == k for k in range(5)}
GALL = {"all": np.ones(N, bool)}; GPOOL = {"n7_pool0_4": n7 <= 4}
pre_auc = {(g, m): prep(SC[m][msk], y[msk]) for G in (G6, GALL) for g, msk in G.items() for m in SC}
pre_ap = {(g, m): prep(-SC[m][msk], y[msk]) for G in (G7, G6) for g, msk in G.items() for m in SC}
pre_dth = {g: prep(pf[msk], dth[msk]) for G in (G7, GPOOL) for g, msk in G.items()}
xcal = {g: logit(pf[msk]) for g, msk in G7.items()}
ALLG = G7 | G6 | GALL | GPOOL
def stats(wt):
    r = {}
    for (g, m), p in pre_auc.items(): r[("auroc", g, m)] = wauc(p, wt[ALLG[g]])
    for (g, m), p in pre_ap.items():  r[("auprc", g, m)] = wap(p, wt[ALLG[g]])
    for g, p in pre_dth.items():      r[("auroc_death", g, "full")] = wauc(p, wt[ALLG[g]])
    for g, msk in G7.items():
        w, yy, pp = wt[msk], y[msk], pf[msk]
        r[("oe", g, "full")] = float(np.sum(w * yy) / np.sum(w * pp))
        r[("brier", g, "full")] = float(np.sum(w * (pp - yy) ** 2) / np.sum(w))
        a, b = calib(xcal[g], yy, w); r[("cal_intercept", g, "full")] = a; r[("cal_slope", g, "full")] = b
        ppv, sens, lift = top10(pre_ap[(g, "full")], w)
        r[("top10_ppv", g, "full")] = ppv; r[("top10_sensitivity", g, "full")] = sens; r[("top10_lift", g, "full")] = lift
    return r

# ---------------------------------------------------------------- bootstrap (same codes, seed and draws as the P2 bootstrap)
codes, _ = pd.factorize(pid); P = codes.max() + 1
point = stats(np.ones(N))
assert abs(point[("auroc", "all", "full")] - roc_auc_score(y, pf)) < 1e-7
m0 = G7["n7_s0"]; assert abs(point[("auprc", "n7_s0", "full")] - average_precision_score(y[m0], pf[m0])) < 1e-7
keys = list(point); boot = np.empty((B, len(keys)))
rng = np.random.default_rng(SEED)
for bi in range(B):
    wt = np.bincount(rng.integers(0, P, P), minlength=P).astype(float)[codes]
    r = stats(wt); boot[bi] = [r[k] for k in keys]
    if (bi + 1) % 250 == 0: log(f"bootstrap {bi+1}/{B}")
f_rep = os.path.join(OUT, "P2_tables_boot_replicates.npy"); np.save(f_rep, boot)
f_key = os.path.join(OUT, "P2_tables_boot_keys.csv"); pd.Series([str(k) for k in keys]).to_csv(f_key, index=False)
register("paper2", "P2_TABLES_BOOT_REPLICATES", f_rep); register("paper2", "P2_TABLES_BOOT_KEYS", f_key)
bd = {k: boot[:, i] for i, k in enumerate(keys)}
Q = []
def add(name, group, p, b):
    Q.append({"quantity": name, "group": group, "point": p, "lo95": np.nanpercentile(b, 2.5), "hi95": np.nanpercentile(b, 97.5),
              "lo99": np.nanpercentile(b, 0.5), "hi99": np.nanpercentile(b, 99.5)})
for k in keys: add(f"{k[0]} {k[2]}", k[1], point[k], bd[k])
for met, groups in [("auprc", G7), ("auprc", G6), ("auroc", G6)]:
    for g in groups:
        ka, kb = (met, g, "full"), (met, g, "no_both")
        add(f"{met} delta_s (full - no_both)", g, point[ka] - point[kb], bd[ka] - bd[kb])
dgp = point[("auroc", "all", "full")] - point[("auroc", "all", "no_both")]
dgb = bd[("auroc", "all", "full")] - bd[("auroc", "all", "no_both")]
for g in G6:
    ka, kb = ("auroc", g, "full"), ("auroc", g, "no_both")
    add("auroc delta_s - delta_global", g, point[ka] - point[kb] - dgp, bd[ka] - bd[kb] - dgb)
sizes = pd.DataFrame([{"group": g, "windows": int(msk.sum()), "patients": int(len(np.unique(pid[msk]))),
                       "positives": int(y[msk].sum()), "prevalence (AUPRC baseline)": float(y[msk].mean()),
                       "deaths": int(dth[msk].sum()), "death_rate": float(dth[msk].mean())} for g, msk in (G7 | G6 | GPOOL).items()])
Qd = pd.DataFrame(Q)
f_q = os.path.join(OUT, "P2_tables_bootstrap.csv"); Qd.to_csv(f_q, index=False); register("paper2", "P2_TABLES_BOOTSTRAP", f_q)
f_s = os.path.join(OUT, "P2_tables_set_sizes.csv"); sizes.to_csv(f_s, index=False); register("paper2", "P2_TABLES_SET_SIZES", f_s)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print("\n=== Table 2 ===\n", T2.to_string(index=False))
print("\n=== admission periods (anchor_year_group, share of stays) ===\n", AP.pivot(index="anchor_year_group", columns="set", values="share_of_stays").round(3).to_string())
print("\n=== strata sizes, prevalence, death ===\n", sizes.round(4).to_string(index=False))
print("\n=== bootstrap results ===\n", Qd.round(4).to_string(index=False))
log("done")

# %% ---- cell 18 ----
import sys, importlib, numpy as np, pandas as pd; sys.path.insert(0, r"<PHD_ROOT>")
import paths.paper2; importlib.reload(paths.paper2); from paths.paper2 import *
from paths.registry import register
A, T, Bb = pd.read_csv(P2_BOOTSTRAP), pd.read_csv(P2_TABLES_BOOTSTRAP), pd.read_csv(P2_BASELINE_BOOTSTRAP)
def f(x, d): s = f"{round(float(x), d):.{d}f}"; return s.replace("-", "−")
def c(df, q, g, d=3, lvl=99, an=None, m=1):
    s = df[(df.quantity == q) & (df.group == g)]
    if an is not None: s = s[s.analysis == an]
    r = s.iloc[0]; lo, hi = (r.lo99, r.hi99) if lvl == 99 else (r.lo95, r.hi95)
    return f"{f(r.point*m, d)} ({f(lo*m, d)} to {f(hi*m, d)})"
P = "primary"; X = "excluding overlap subjects"
t3 = []
for k in range(5):
    t3.append(["7c", k, c(A, "auroc full", f"s{k}", an=P), c(A, "auroc no_both", f"s{k}", an=P),
               c(A, "auroc delta_s (full - no_both)", f"s{k}", 4, an=P), c(Bb, "auroc baseline", f"s{k}"), c(T, "auprc full", f"n7_s{k}")])
t3.append(["7c", "pool", c(Bb, "auroc full", "pool0_4", lvl=95), c(Bb, "auroc no_both", "pool0_4", lvl=95),
           c(Bb, "auroc full - no_both", "pool0_4", 4, lvl=95), c(Bb, "auroc baseline", "pool0_4", lvl=95), "—"])
for k in range(5):
    t3.append(["6c", k, c(T, "auroc full", f"n6_s{k}"), c(T, "auroc no_both", f"n6_s{k}"),
               c(T, "auroc delta_s (full - no_both)", f"n6_s{k}", 4), "—", c(T, "auprc full", f"n6_s{k}")])
t3.append(["all", "all", c(A, "auroc full", "global", lvl=95, an=P), c(A, "auroc no_both", "global", lvl=95, an=P),
           c(A, "auroc full - no_both", "global", 4, lvl=95, an=P), c(Bb, "auroc baseline", "all", lvl=95), c(A, "auprc full", "global", lvl=95, an=P)])
T3 = pd.DataFrame(t3, columns=["spec", "score", "auroc_full", "auroc_no_monitoring", "delta", "auroc_news2_inputs", "auprc_full"])
t5 = [["R1", k, c(A, "auroc full", f"s{k}", an=X), c(A, "auroc delta_s (full - no_both)", f"s{k}", 4, an=X)] for k in range(5)]
t5.append(["R1", "all", c(A, "auroc full", "global", lvl=95, an=X), c(A, "auroc full - no_both", "global", 4, lvl=95, an=X)])
t5 += [["R2", k, c(T, "auroc_death full", f"n7_s{k}"), "—"] for k in range(5)]
t5.append(["R2", "pool", c(T, "auroc_death full", "n7_pool0_4", lvl=95), "—"])
t5 += [["no-red", k, c(Bb, "auroc full", f"s{k}_nored"), c(Bb, "auroc full - no_both", f"s{k}_nored", 4)] for k in (3, 4)]
t5 += [["no-red baseline", k, c(Bb, "auroc baseline", f"s{k}_nored"), "—"] for k in (3, 4)]
T5 = pd.DataFrame(t5, columns=["analysis", "score", "auroc_full", "delta"])
T6 = pd.DataFrame([[k, c(T, "oe full", f"n7_s{k}", 2), c(T, "cal_intercept full", f"n7_s{k}", 2), c(T, "cal_slope full", f"n7_s{k}", 2),
                    c(T, "brier full", f"n7_s{k}", 3), c(T, "top10_ppv full", f"n7_s{k}", 1, m=100),
                    c(T, "top10_sensitivity full", f"n7_s{k}", 1, m=100), c(T, "top10_lift full", f"n7_s{k}", 1)] for k in range(5)],
                  columns=["score", "OE", "intercept", "slope", "brier", "ppv_pct", "sens_pct", "lift"])
extra = {"news2_7 global": c(A, "auroc news2_7", "global", lvl=95, an=P), "news2_6 global": c(A, "auroc news2_6", "global", lvl=95, an=P)}
q = A[A.group.isin([f"s{k}" for k in range(5)]) & A.quantity.isin(["auroc full", "auroc no_both", "auroc delta_s (full - no_both)"])]
p = q.pivot_table(index=["quantity", "group"], columns="analysis", values="point"); p["chg"] = (p[X] - p[P]).abs()
extra["R1 max AUROC change"] = f(p.loc[["auroc full", "auroc no_both"], "chg"].max(), 4)
extra["R1 max delta_s change"] = f(p.loc["auroc delta_s (full - no_both)", "chg"].max(), 4)
for name, df in [("P2_table3_final.csv", T3), ("P2_table5_final.csv", T5), ("P2_table6_final.csv", T6)]:
    fn = os.path.join(P2_OUT, name) if "os" in dir() else P2_OUT + "\\" + name
    df.to_csv(fn, index=False, encoding="utf-8-sig"); register("paper2", name.replace(".csv", "").upper(), fn)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
for n, df in [("Table 3", T3), ("Table 5", T5), ("Table 6", T6)]: print(f"\n=== {n} ===\n", df.to_string(index=False))
print("\n", extra)

# %% ---- cell 19 ----
import sys, importlib, ast, json, numpy as np, pandas as pd; sys.path.insert(0, r"<PHD_ROOT>")
import paths.paper2; importlib.reload(paths.paper2); from paths.paper2 import *
R = np.load(P2_TABLES_BOOT_REPLICATES); K = [ast.literal_eval(k) for k in pd.read_csv(P2_TABLES_BOOT_KEYS).iloc[:, 0]]
T = pd.read_csv(P2_TABLES_BOOTSTRAP); col = {k: i for i, k in enumerate(K)}
pt = lambda q, g: T[(T.quantity == q) & (T.group == g)].point.iloc[0]
d = lambda g: R[:, col[("auroc", g, "full")]] - R[:, col[("auroc", g, "no_both")]]
gb = d("n6_s0") - d("n6_s4")
gp = (pt("auroc full", "n6_s0") - pt("auroc no_both", "n6_s0")) - (pt("auroc full", "n6_s4") - pt("auroc no_both", "n6_s4"))
print(f"S1 G = {gp:.4f} (95% CI {np.percentile(gb, 2.5):.4f} to {np.percentile(gb, 97.5):.4f})")
J = json.load(open(CORRECTED_RESULTS))
for m in ["full", "no_both"]:
    a = np.array([f["auroc"] for f in J[m]["cv_folds"]])
    print(f"{m}: folds {a.min():.3f} to {a.max():.3f}, mean {a.mean():.3f}")
