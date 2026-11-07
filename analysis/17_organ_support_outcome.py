# Source: paper2.ipynb, code cells [29, 30] (outputs not included)

# %% ---- cell 29 ----
# ============================================================================
# Paper 2 — clinically independent outcome (reviewer Major 1), using the SAL organ-support labels
#   Labels: organ_support_window_labels.npz (SAL.ipynb cell 1):
#     vasopressor (221906, 221289, 221749, 222315, 221662; inputevents, rate or amount > 0, not 'Rewritten')
#     invasive ventilation (procedure 225792 intervals + intubation 224385); episodes merged across gaps <= 1 h
#     *_active  = episode running at any time in the observation window
#     *_incident = episode START within the 6-h horizon
#   Eligible = composite not active in the observation window; outcome = composite incident
#   Re-scores saved held-out predictions (full, no_both, NEWS-2-inputs); no retraining
#   Bootstrap: same patient codes, seed and draws as the P2 bootstrap (B = 2,000)
# Output: <PHD_ROOT>\Paper2\corrected\P2_organ_support_*.csv
# ============================================================================
import os, sys, importlib, time, numpy as np, pandas as pd
sys.path.insert(0, r"<PHD_ROOT>")
import paths.shared, paths.paper2; importlib.reload(paths.shared); importlib.reload(paths.paper2); from paths.paper2 import *
from paths.registry import register
from sklearn.metrics import roc_auc_score

B, SEED, N_TOTAL = 2000, 42, 3_498_146
t00 = time.time()
def log(m): print(f"[{time.time()-t00:6.0f}s] {m}", flush=True)

W = pd.read_parquet(WINDOWS, columns=["window_id", "subject_id"]).sort_values("window_id")
assert len(W) == N_TOTAL; subj_all = W.subject_id.to_numpy(np.int64); del W
rows = np.load(CORRECTED_HO_ROWS); N = len(rows); assert N == 658_615
pid = subj_all[rows]
y_det = np.load(Y_REPAIRED)[rows].astype(int)
n7 = np.rint(np.load(NEWS7_REP)[rows]).astype(int)
SC = {"full": np.load(PRED_FULL_CORR).astype(float).ravel(), "no_both": np.load(PRED_REDUCED_CORR).astype(float).ravel(),
      "baseline": np.load(P2_BASELINE_PRED).astype(float).ravel()}
for v in SC.values(): assert len(v) == N
Lz = np.load(LABELS_NPZ, allow_pickle=True); fr = Lz["file_row"].astype(np.int64)
assert np.isin(rows, fr).all()
def lab(k): a = np.zeros(N_TOTAL, bool); a[fr] = Lz[k].astype(bool); return a[rows]
act = {e: lab(f"{e}_active") for e in ("composite", "vasopressor", "ventilation")}
inc = {e: lab(f"{e}_incident") for e in ("composite", "vasopressor", "ventilation")}
elig = ~act["composite"]; yc = (inc["composite"] & elig).astype(int)
log(f"eligible {elig.sum():,} ({elig.mean():.1%}); composite events {yc.sum():,} ({yc[elig].mean():.2%} of eligible); "
    f"vasopressor {int((inc['vasopressor'] & elig).sum()):,}; ventilation {int((inc['ventilation'] & elig).sum()):,}")

def prep(s, yy):
    o = np.argsort(s, kind="mergesort"); ss = s[o]
    return o, np.r_[0, np.cumsum(ss[1:] != ss[:-1])], yy[o].astype(float)
def wauc(pre, wt):
    o, gid, yo = pre; wo = wt[o]; G = gid[-1] + 1
    pos = np.bincount(gid, weights=wo * yo, minlength=G); neg = np.bincount(gid, weights=wo * (1 - yo), minlength=G)
    den = pos.sum() * neg.sum()
    return np.nan if den == 0 else float(np.sum(pos * (np.cumsum(neg) - neg + 0.5 * neg)) / den)
GROUPS = {"all_eligible": elig, "pool0_4": elig & (n7 <= 4)} | {f"s{k}": elig & (n7 == k) for k in range(5)}
pre = {(g, m): prep(SC[m][msk], yc[msk]) for g, msk in GROUPS.items() for m in SC}
codes, _ = pd.factorize(pid); P = codes.max() + 1
point = {k: wauc(p, np.ones(GROUPS[k[0]].sum())) for k, p in pre.items()}
assert abs(point[("all_eligible", "full")] - roc_auc_score(yc[elig], SC["full"][elig])) < 1e-7
keys = list(point); boot = np.empty((B, len(keys))); rng = np.random.default_rng(SEED)
for bi in range(B):
    wt = np.bincount(rng.integers(0, P, P), minlength=P).astype(float)[codes]
    boot[bi] = [wauc(pre[k], wt[GROUPS[k[0]]]) for k in keys]
    if (bi + 1) % 250 == 0: log(f"bootstrap {bi+1}/{B}")
np.save(os.path.join(P2_OUT, "P2_organ_support_boot_replicates.npy"), boot)
bd = {k: boot[:, i] for i, k in enumerate(keys)}
Q = []
def add(q, g, p, b): Q.append({"quantity": q, "group": g, "point": p, "lo95": np.nanpercentile(b, 2.5), "hi95": np.nanpercentile(b, 97.5),
                               "lo99": np.nanpercentile(b, 0.5), "hi99": np.nanpercentile(b, 99.5)})
for (g, m) in keys: add(f"auroc {m}", g, point[(g, m)], bd[(g, m)])
for g in GROUPS:
    for a, b in [("full", "baseline"), ("full", "no_both")]:
        add(f"auroc {a} - {b}", g, point[(g, a)] - point[(g, b)], bd[(g, a)] - bd[(g, b)])
Qd = pd.DataFrame(Q)
comp_pts = []
for e in ("vasopressor", "ventilation"):
    el = ~act[e]; ye = (inc[e] & el).astype(int)
    for g, msk in {"all_eligible": el, "pool0_4": el & (n7 <= 4)}.items():
        comp_pts.append({"endpoint": e, "group": g, "windows": int(msk.sum()), "events": int(ye[msk].sum()),
                         "auroc_full": roc_auc_score(ye[msk], SC["full"][msk])})
sizes = pd.DataFrame([{"group": g, "windows": int(m.sum()), "patients": int(len(np.unique(pid[m]))), "events": int(yc[m].sum()),
                       "event_rate": float(yc[m].mean()), "news2_deterioration_rate": float(y_det[m].mean()),
                       "events_also_news2_positive": int((yc[m] & y_det[m]).sum())} for g, m in GROUPS.items()])
for fn, df, key in [("P2_organ_support_bootstrap.csv", Qd, "P2_ORGAN_BOOTSTRAP"), ("P2_organ_support_sizes.csv", sizes, "P2_ORGAN_SIZES"),
                    ("P2_organ_support_components.csv", pd.DataFrame(comp_pts), "P2_ORGAN_COMPONENTS")]:
    f = os.path.join(P2_OUT, fn); df.to_csv(f, index=False); register("paper2", key, f)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print("\n=== groups ===\n", sizes.round(4).to_string(index=False))
print("\n=== components (point estimates) ===\n", pd.DataFrame(comp_pts).round(4).to_string(index=False))
print("\n=== bootstrap (new organ support within 6 h) ===\n", Qd.round(4).to_string(index=False))
log("done")

# %% ---- cell 30 ----
import sys, importlib, pandas as pd; sys.path.insert(0, r"<PHD_ROOT>")
import paths.paper2; importlib.reload(paths.paper2); from paths.paper2 import *
Q = pd.read_csv(P2_ORGAN_BOOTSTRAP)
f = lambda x, d=3: f"{round(float(x), d):.{d}f}".replace("-", "−")
for _, r in Q.iterrows():
    lv = (r.lo95, r.hi95) if r.group in ("all_eligible", "pool0_4") else (r.lo99, r.hi99)
    print(f"{r.quantity:24s} {r.group:13s} {f(r.point)} ({f(lv[0])} to {f(lv[1])})")
