# Source: paper2.ipynb, code cells [15] (outputs not included)

# %% ---- cell 15 ----
# ============================================================================
# Paper 2 — primary analysis on the corrected model (analysis plan emailed 2026-10-28 14:08 +03:00)
#   Input : <PHD_ROOT>\ewp_results_v3\corrected\  (full, no_both, full_avail48; Platt-calibrated held-out predictions)
#   Set   : held-out windows, repaired label, rebuilt gate (658,615 / 128,844); NEWS-2_7 mismatch windows excluded
#   Inference: patient-level paired bootstrap, B = 2,000, seed 42; every quantity on the same resamples;
#              percentile intervals at 95% and 99%
#   Quantities:
#     global : AUROC (primary) and AUPRC (secondary) for full, no_both, full_avail48, NEWS-2_6, NEWS-2_7;
#              full - no_both; full - NEWS-2_6; full - NEWS-2_7; full_avail48 - no_both; full_avail48 - full
#     strata : repaired NEWS-2_7 = 0..4: AUROC full / no_both, delta_s, delta_s - delta_global;
#              G = delta_0 - delta_4 (descriptive, continuity with R6)
#     calibration (full): Brier, calibration intercept (offset model) and slope; reliability by deciles
#   Analyses: primary; sensitivity excluding subjects who also have development windows
#   Decisions (plan): within-score claim = delta_s 99% lower bound > 0 in >= 4 of 5 strata;
#                     heterogeneity = any delta_s - delta_global 99% interval excludes 0;
#                     reported global gain = delta 95% lower bound > 0 and point >= 0.002
# Output: <PHD_ROOT>\Paper2\corrected\P2_*.csv / .json / .png / .pdf
# ============================================================================
import os, json, time, gc
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

PHD = r"<PHD_ROOT>"
BASE = rf"{PHD}\ewp_results_v3"; EXP = rf"{BASE}\corrected"; PV = rf"{EXP}\prediction_vectors"
OUT = rf"{PHD}\Paper2\corrected"; os.makedirs(OUT, exist_ok=True)
WINDOWS = rf"{PHD}\PHD Research\DATAset\s4\windows.parquet"
B, SEED, N_TOTAL = 2000, 42, 3_498_146
MODELS = ["full", "no_both", "full_avail48"]
t00 = time.time()
def log(m): print(f"[{time.time()-t00:6.0f}s] {m}", flush=True)

# ---------------------------------------------------------------- inputs and integrity checks
rows = np.load(rf"{EXP}\holdout_rows.npy")
assert len(rows) == 658_615
y = np.load(rf"{BASE}\y_repaired.npy")[rows].astype(int); assert int(y.sum()) == 128_844
assert np.load(rf"{BASE}\keep_repaired.npy")[rows].all()
S = {m: np.load(rf"{PV}\{m}_holdout_probs_CALIBRATED.npy").astype(float).ravel() for m in MODELS}
for m in MODELS: assert len(S[m]) == len(rows), m
fr = json.load(open(rf"{EXP}\FINAL_results.json"))
for m in MODELS:
    assert fr[m]["calibration"]["platt_a"] > 0
    assert abs(roc_auc_score(y, S[m]) - fr[m]["holdout"]["auroc"]) < 1e-6, f"{m}: predictions do not match FINAL_results"
S["news2_6"] = np.load(rf"{BASE}\news2_6_repaired_we.npy")[rows].astype(float)
S["news2_7"] = np.load(rf"{BASE}\news7_repaired.npy")[rows].astype(float)
mm = np.load(rf"{BASE}\news7_mismatch_mask.npy")[rows] | np.isnan(S["news2_7"])
W = pd.read_parquet(WINDOWS, columns=["window_id", "subject_id"]).sort_values("window_id")
assert len(W) == N_TOTAL
subj = W.subject_id.to_numpy().astype(np.int64); del W; gc.collect()
train_all = np.load(rf"{PHD}\temporal_holdout\train_s4_rows.npy"); hold_all = np.load(rf"{PHD}\temporal_holdout\holdout_s4_rows.npy")
overlap_subj = np.intersect1d(np.unique(subj[train_all]), np.unique(subj[hold_all]))
pid = subj[rows]; ov = np.isin(pid, overlap_subj)
counts = {"heldout_windows": int(len(rows)), "heldout_positives": int(y.sum()), "heldout_patients": int(len(np.unique(pid))),
          "news7_mismatch_excluded": int(mm.sum()),
          "overlap_subjects_total": int(len(overlap_subj)),
          "overlap_subjects_in_analysis_set": int(len(np.unique(pid[ov & ~mm]))),
          "overlap_heldout_windows_removed": int((ov & ~mm).sum()),
          "overlap_heldout_positives_removed": int(y[ov & ~mm].sum())}
log(f"counts: {counts}")

# ---------------------------------------------------------------- weighted AUROC / AUPRC for the patient bootstrap
def prep(s, yy):
    o = np.argsort(s, kind="mergesort"); ss = s[o]
    gid = np.r_[0, np.cumsum(ss[1:] != ss[:-1])]
    return o, gid, yy[o].astype(float)
def wauc(pre, wt):
    o, gid, yo = pre; wo = wt[o]; G = gid[-1] + 1
    pos = np.bincount(gid, weights=wo * yo, minlength=G); neg = np.bincount(gid, weights=wo * (1 - yo), minlength=G)
    den = pos.sum() * neg.sum()
    return np.nan if den == 0 else float(np.sum(pos * (np.cumsum(neg) - neg + 0.5 * neg)) / den)
def wap(pre_desc, wt):                                   # pre built on -score: groups in descending score order
    o, gid, yo = pre_desc; wo = wt[o]; G = gid[-1] + 1
    pos = np.bincount(gid, weights=wo * yo, minlength=G); neg = np.bincount(gid, weights=wo * (1 - yo), minlength=G)
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    return np.nan if tp[-1] == 0 else float(np.sum(pos * tp / np.maximum(tp + fp, 1e-300)) / tp[-1])
def logit(p): p = np.clip(p, 1e-7, 1 - 1e-7); return np.log(p / (1 - p))
def calib(x, yy, w):
    a = 0.0                                              # intercept with logit(p) as offset (calibration-in-the-large)
    for _ in range(8):
        mu = 1 / (1 + np.exp(-(a + x))); a += np.sum(w * (yy - mu)) / np.sum(w * mu * (1 - mu))
    b = np.array([0.0, 1.0])                             # logistic recalibration: y ~ b0 + b1 * logit(p)
    for _ in range(8):
        mu = 1 / (1 + np.exp(-(b[0] + b[1] * x))); v = w * mu * (1 - mu); r = w * (yy - mu)
        H = np.array([[v.sum(), (v * x).sum()], [(v * x).sum(), (v * x * x).sum()]])
        b = b + np.linalg.solve(H, np.array([r.sum(), (r * x).sum()]))
    return a, b[1]

# ---------------------------------------------------------------- analysis
def analyse(mask, tag, with_calibration):
    codes, _ = pd.factorize(pid[mask]); P = codes.max() + 1; yy = y[mask]
    sc = {k: v[mask] for k, v in S.items()}; s7 = np.rint(sc["news2_7"]).astype(int)
    groups = {"global": np.ones(len(yy), bool)} | {f"s{k}": s7 == k for k in range(5)}
    auc_items = [("global", m) for m in S] + [(f"s{k}", m) for k in range(5) for m in ["full", "no_both"]]
    ap_items = [("global", m) for m in S]
    pre_auc = {(g, m): prep(sc[m][groups[g]], yy[groups[g]]) for g, m in auc_items}
    pre_ap = {(g, m): prep(-sc[m][groups[g]], yy[groups[g]]) for g, m in ap_items}
    xf = logit(sc["full"])
    def stats(wt):
        r = {("auroc",) + it: wauc(pre_auc[it], wt[groups[it[0]]]) for it in auc_items}
        r.update({("auprc",) + it: wap(pre_ap[it], wt[groups[it[0]]]) for it in ap_items})
        if with_calibration:
            r[("brier", "global", "full")] = float(np.sum(wt * (sc["full"] - yy) ** 2) / wt.sum())
            a, b = calib(xf, yy, wt); r[("cal_intercept", "global", "full")] = a; r[("cal_slope", "global", "full")] = b
        return r
    point = stats(np.ones(len(yy)))
    assert abs(point[("auroc", "global", "full")] - roc_auc_score(yy, sc["full"])) < 1e-7
    assert abs(point[("auprc", "global", "full")] - average_precision_score(yy, sc["full"])) < 1e-7
    rng = np.random.default_rng(SEED); keys = list(point); boot = np.empty((B, len(keys)))
    for bi in range(B):
        wt = np.bincount(rng.integers(0, P, P), minlength=P).astype(float)[codes]
        r = stats(wt); boot[bi] = [r[k] for k in keys]
        if (bi + 1) % 250 == 0: log(f"{tag}: bootstrap {bi+1}/{B}")
    bd = {k: boot[:, i] for i, k in enumerate(keys)}; pt = point
    Q = []
    def add(name, group, p, b):
        Q.append({"analysis": tag, "quantity": name, "group": group, "point": p,
                  "lo95": np.nanpercentile(b, 2.5), "hi95": np.nanpercentile(b, 97.5),
                  "lo99": np.nanpercentile(b, 0.5), "hi99": np.nanpercentile(b, 99.5)})
    for k in keys: add(f"{k[0]} {k[2]}", k[1], pt[k], bd[k])
    A = lambda g, m: ("auroc", g, m)
    for met in ["auroc", "auprc"]:
        for a_, b_ in [("full", "no_both"), ("full", "news2_6"), ("full", "news2_7"), ("full_avail48", "no_both"), ("full_avail48", "full")]:
            ka, kb = (met, "global", a_), (met, "global", b_)
            add(f"{met} {a_} - {b_}", "global", pt[ka] - pt[kb], bd[ka] - bd[kb])
    dg_p, dg_b = pt[A("global", "full")] - pt[A("global", "no_both")], bd[A("global", "full")] - bd[A("global", "no_both")]
    for k in range(5):
        g = f"s{k}"; dp, db = pt[A(g, "full")] - pt[A(g, "no_both")], bd[A(g, "full")] - bd[A(g, "no_both")]
        add("auroc delta_s (full - no_both)", g, dp, db)
        add("auroc delta_s - delta_global", g, dp - dg_p, db - dg_b)
    d0 = (pt[A("s0", "full")] - pt[A("s0", "no_both")], bd[A("s0", "full")] - bd[A("s0", "no_both")])
    d4 = (pt[A("s4", "full")] - pt[A("s4", "no_both")], bd[A("s4", "full")] - bd[A("s4", "no_both")])
    add("G = delta_0 - delta_4 (descriptive)", "s0 vs s4", d0[0] - d4[0], d0[1] - d4[1])
    Qd = pd.DataFrame(Q)
    sizes = pd.DataFrame([{"analysis": tag, "group": g, "windows": int(m.sum()), "patients": int(len(np.unique(codes[m]))),
                           "positives": int(yy[m].sum()), "prevalence": float(yy[m].mean())} for g, m in groups.items()])
    return Qd, sizes, ((sc["full"], yy) if with_calibration else None)

Q1, Z1, calib_data = analyse(~mm, "primary", True)
Q2, Z2, _ = analyse(~mm & ~ov, "excluding overlap subjects", False)
Q = pd.concat([Q1, Q2], ignore_index=True); Z = pd.concat([Z1, Z2], ignore_index=True)
Q.to_csv(os.path.join(OUT, "P2_bootstrap_results.csv"), index=False)
Z.to_csv(os.path.join(OUT, "P2_analysis_set_sizes.csv"), index=False)

# ---------------------------------------------------------------- reliability (full model, primary set)
pf, yy = calib_data
dec = pd.qcut(pf, 10, labels=False, duplicates="drop")
rel = pd.DataFrame({"p": pf, "y": yy, "d": dec}).groupby("d").agg(n=("y", "size"), mean_predicted=("p", "mean"), observed=("y", "mean")).reset_index()
rel.to_csv(os.path.join(OUT, "P2_reliability_deciles.csv"), index=False)
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#999999")
ax.plot(rel.mean_predicted, rel.observed, marker="o", ms=3, lw=1, color="#0072B2")
ax.set_xlabel("Mean predicted risk (decile)"); ax.set_ylabel("Observed outcome rate")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); fig.tight_layout()
fig.savefig(os.path.join(OUT, "P2_reliability.png"), dpi=600); fig.savefig(os.path.join(OUT, "P2_reliability.pdf"))

# ---------------------------------------------------------------- pre-registered decisions
def decisions(Qa):
    st = Qa[Qa.quantity == "auroc delta_s (full - no_both)"]
    het = Qa[Qa.quantity == "auroc delta_s - delta_global"]
    dg = Qa[(Qa.quantity == "auroc full - no_both") & (Qa.group == "global")].iloc[0]
    n_pos = int((st.lo99 > 0).sum())
    return {"strata_with_delta_s_99_lower_above_0": n_pos,
            "within_score_claim_holds (>= 4 of 5)": bool(n_pos >= 4),
            "heterogeneity_claim_holds (any 99% excludes 0)": bool(((het.lo99 > 0) | (het.hi99 < 0)).any()),
            "global_delta": float(dg.point), "global_delta_95": [float(dg.lo95), float(dg.hi95)],
            "reported_global_gain (95% lower > 0 and point >= 0.002)": bool(dg.lo95 > 0 and dg.point >= 0.002)}
dec_out = {"plan": "Paper 2 analysis plan emailed 2026-10-28 14:08 (+03:00)", "B": B, "seed": SEED, "counts": counts,
           "primary": decisions(Q1), "excluding_overlap_subjects": decisions(Q2)}
json.dump(dec_out, open(os.path.join(OUT, "P2_decisions.json"), "w"), indent=2)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print("\n=== analysis set sizes ===\n", Z.to_string(index=False))
print("\n=== bootstrap results ===\n", Q.round(4).to_string(index=False))
print("\n=== reliability (deciles) ===\n", rel.round(4).to_string(index=False))
print("\n=== PRE-REGISTERED DECISIONS ===\n", json.dumps(dec_out, indent=2))
log("done")
