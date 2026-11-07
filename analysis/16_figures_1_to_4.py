# Source: paper2.ipynb, code cells [23] (outputs not included)

# %% ---- cell 23 ----
# ============================================================================
# Paper 2 — Figures 1-4 (corrected model), layout v3
#   Fig 1: panels placed with two grids (no label overlap), compact height, B title and legend close to panels
#   Fig 3: legend raised above the 0.5 reference line, 0.5 label below the line
#   Fig 4: B value labels above each interval
# Output: <PHD_ROOT>\Paper2\corrected\Figure1..4_*.png (600 dpi) / .pdf
# ============================================================================
import os, sys, importlib, ast, numpy as np, pandas as pd
sys.path.insert(0, r"<PHD_ROOT>")
import paths.shared, paths.paper2; importlib.reload(paths.shared); importlib.reload(paths.paper2); from paths.paper2 import *
from paths.registry import register
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, roc_auc_score

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.dpi": 600, "mathtext.default": "regular"})
BLUE, ORANGE, GREY, DARK, RED = "#0072B2", "#E69F00", "#999999", "#333333", "#F4CCC0"
A = pd.read_csv(P2_BOOTSTRAP); T = pd.read_csv(P2_TABLES_BOOTSTRAP); Bb = pd.read_csv(P2_BASELINE_BOOTSTRAP)
S7 = pd.read_csv(P2_SET_SIZES); S6 = pd.read_csv(P2_TABLES_SET_SIZES)
def get(df, q, g, an=None):
    s = df[(df.quantity == q) & (df.group == g)]
    if an is not None: s = s[s.analysis == an]
    assert len(s) == 1, (q, g, an); return s.iloc[0]
r3 = lambda x: f"{round(float(x), 3):.3f}"
f4 = lambda x: f"{round(float(x), 4):.4f}".replace("-", "−")
def save(fig, name, key):
    for ext in ("png", "pdf"): fig.savefig(os.path.join(P2_OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.03)
    register("paper2", key, os.path.join(P2_OUT, f"{name}.png")); plt.close(fig); print("saved", name)
prim = S7[S7.analysis == "primary"]
N_HO = int(prim[prim.group == "global"].windows.iloc[0]); NS = [int(prim[prim.group == f"s{k}"].windows.iloc[0]) for k in range(5)]
PV = [float(prim[prim.group == f"s{k}"].prevalence.iloc[0]) for k in range(5)]
rows = np.load(CORRECTED_HO_ROWS); y = np.load(Y_REPAIRED)[rows].astype(int)
pf = np.load(PRED_FULL_CORR).astype(float).ravel(); n7 = np.rint(np.load(NEWS7_REP)[rows]).astype(int)
assert len(y) == N_HO == 658_615 and abs(roc_auc_score(y, pf) - get(A, "auroc full", "global", "primary").point) < 1e-7

# ---------------------------------------------------------------- Figure 1
fig = plt.figure(figsize=(7.1, 2.25))
gA = fig.add_gridspec(1, 1, left=0.06, right=0.19, bottom=0.30, top=0.80)
gB = fig.add_gridspec(1, 5, left=0.29, right=0.995, bottom=0.30, top=0.80, wspace=0.18)
ax = fig.add_subplot(gA[0, 0]); fpr, tpr, _ = roc_curve(y, pf)
ax.plot(fpr, tpr, color="black", lw=1.2); ax.plot([0, 1], [0, 1], ls=":", color=GREY, lw=0.8)
ax.set_title(f"A  All held-out windows\nAUROC {r3(get(A, 'auroc full', 'global', 'primary').point)}", loc="left", fontsize=7, fontweight="bold")
ax.set_xlabel(f"False positive rate\nn = {N_HO:,}"); ax.set_ylabel("True positive rate")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
grid = np.linspace(0, 1, 400); rng = np.random.default_rng(42)
for k in range(5):
    axk = fig.add_subplot(gB[0, k]); m = n7 == k
    for lab, colr in [(0, BLUE), (1, ORANGE)]:
        v = pf[m & (y == lab)]
        if len(v) > 30000: v = rng.choice(v, 30000, replace=False)
        dens = gaussian_kde(v)(grid); dens /= dens.max()
        axk.fill_between(grid, dens, color=colr, alpha=0.30, lw=0); axk.plot(grid, dens, color=colr, lw=0.9)
    axk.set_title(f"NEWS-2 = {k}\nAUROC {r3(get(A, 'auroc full', f's{k}', 'primary').point)}", fontsize=7)
    axk.set_xlim(0, 1); axk.set_ylim(0, 1.05); axk.set_xticks([0, 0.5, 1]); axk.set_xticklabels(["0", "0.5", "1"])
    axk.set_xlabel(f"Predicted risk\nn = {NS[k]:,} · {100 * PV[k]:.1f}%")
    if k == 0: axk.set_ylabel("Density (peak-normalised)")
    else: axk.set_yticklabels([])
fig.text(0.29, 0.99, "B  Within each sub-threshold NEWS-2 value: identical score, different outcomes", fontweight="bold", fontsize=7, va="top")
fig.legend(handles=[plt.Line2D([], [], color=BLUE, lw=1.5, label="Did not deteriorate"),
                    plt.Line2D([], [], color=ORANGE, lw=1.5, label="Deteriorated (NEWS-2 ≥ 7 within 6 h)")],
           loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.64, 0.06))
save(fig, "Figure1_same_score_different_risk", "P2_FIG1")

# ---------------------------------------------------------------- Figure 2 (unchanged)
fig = plt.figure(figsize=(7.1, 5.6))
ax = fig.add_axes([0.07, 0.56, 0.9, 0.38])
ax.add_patch(Rectangle((0, 0), 5.5, 9.5, color="#E5E5E5", lw=0)); ax.add_patch(Rectangle((5.5, 0), 6, 9.5, color="#DCEBF7", lw=0))
ax.add_patch(Rectangle((11.5, 0), 6, 9.5, color="#FBE8CC", lw=0))
ax.text(2.75, 9.1, "Run-in (0–5 h)\nnot used", ha="center", va="top", color="#666666")
ax.text(8.5, 9.1, "Observation period (6 h)\nfeatures from first 5 h", ha="center", va="top", color="#666666")
ax.text(14.5, 9.1, "Prediction horizon (6 h)", ha="center", va="top", color="#666666")
h = np.arange(0, 25); s = np.array([2, 2, 3, 2, 3, 3, 2, 3, 3, 4, 3, 4, 4, 5, 6, 7, 8, 7, 6, 5, 4, 4, 3, 3, 3])
ax.step(h, s, where="post", color="black", lw=1.2)
ax.axhline(7, ls="--", color="#D55E00", lw=0.8); ax.text(24, 7.15, "7  outcome threshold", ha="right", color="#D55E00", fontsize=6)
ax.axhline(5, ls="--", color="#666666", lw=0.8); ax.text(24, 5.15, "5  generation threshold", ha="right", color="#666666", fontsize=6)
ax.plot(11.5, 4, "o", mfc="white", mec="black", ms=4); ax.plot(16, 8, "o", color="#D55E00", ms=4)
ax.annotate("Concurrent score < 5\n→ window generated", xy=(11.4, 4), xytext=(2.5, 6.2), fontsize=6, arrowprops=dict(arrowstyle="-", color="#666666", lw=0.6))
ax.annotate("NEWS-2 ≥ 7 in horizon\n→ positive label", xy=(16.1, 8), xytext=(19, 2.3), fontsize=6, arrowprops=dict(arrowstyle="-", color="#666666", lw=0.6))
ax.text(24, 0.3, "Windows slide forward hourly →", ha="right", fontsize=6, style="italic", color="#666666")
ax.set_xlim(-0.5, 24.5); ax.set_ylim(0, 9.5); ax.set_xticks(range(0, 25, 2)); ax.set_yticks(range(0, 10))
ax.set_xlabel("Hours since ICU admission (illustrative stay)"); ax.set_ylabel("NEWS-2")
ax.set_title("A  Two-threshold window generation", loc="left", fontweight="bold", fontsize=7)
bx = fig.add_axes([0.03, 0.02, 0.95, 0.46]); bx.set_xlim(0, 100); bx.set_ylim(0, 50); bx.axis("off")
bx.text(0, 49, "B  Within-score evaluation", fontweight="bold", fontsize=7, va="top")
def box(x, y0, w, hh, text, fc="#F2F2F2"):
    bx.add_patch(FancyBboxPatch((x, y0), w, hh, boxstyle="round,pad=0.4,rounding_size=0.8", fc=fc, ec="#666666", lw=0.6))
    bx.text(x + w / 2, y0 + hh / 2, text, ha="center", va="center", fontsize=6)
def arrow(x0, y0, x1, y1): bx.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="black", lw=0.7))
N_PRIM = sum(NS); N_EXC = N_HO - N_PRIM; assert N_PRIM == 380_728
box(1, 33, 24, 11, f"Held-out windows\nn = {N_HO:,}\n\nGenerated and labelled with\nsix-component NEWS-2\n(carry-forward limits, Section 3.4)")
box(31, 33, 24, 11, "Stratify by concurrent\nseven-component NEWS-2\n(+2 if supplemental O$_2$,\ncarried forward ≤ 4 h)")
box(61, 36, 22, 6, f"NEWS-2 ≥ 5: excluded\nn = {N_EXC:,} ({100 * N_EXC / N_HO:.1f}%)", fc=RED)
arrow(25.8, 38.5, 30.2, 38.5); arrow(55.8, 39, 60.2, 39); arrow(43, 32.2, 43, 26.5)
bx.text(43, 24.5, f"Primary analysis: {N_PRIM:,} windows", ha="center", fontweight="bold", fontsize=6)
for k, n in enumerate(NS): box(1 + k * 15.5, 10, 13, 7, f"NEWS-2 = {k}\nn = {n:,}")
arrow(78.5, 13.5, 81.2, 13.5)
box(82, 4, 17, 30, "Within each stratum\n\nAUROC, full model\n(223 features)\n\nΔAUROC after removing\n46 monitoring-process\nfeatures\n\nNEWS-2-inputs model\n(post hoc)\n\n99% CI: patient-level\nbootstrap (2,000)")
bx.text(1, 4, f"Sensitivity analysis: same evaluation with six-component strata\nacross all {N_HO:,} windows (Table 3).", fontsize=6, style="italic", color="#666666", va="top")
save(fig, "Figure2_study_design", "P2_FIG2")

# ---------------------------------------------------------------- Figure 3
x = np.arange(5)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.6), gridspec_kw={"wspace": 0.3})
p7 = [100 * v for v in PV]; p6 = [S6[S6.group == f"n6_s{k}"]["prevalence (AUPRC baseline)"].iloc[0] * 100 for k in range(5)]
a1.bar(x - 0.2, p7, 0.38, color=BLUE, label="Seven-component (primary)"); a1.bar(x + 0.2, p6, 0.38, color=BLUE, alpha=0.35, label="Six-component (S1)")
for i, (u, v) in enumerate(zip(p7, p6)):
    a1.text(i - 0.2, u + 0.6, f"{u:.1f}", ha="center", fontsize=6); a1.text(i + 0.2, v + 0.6, f"{v:.1f}", ha="center", fontsize=6)
a1.set_xticks(x); a1.set_xlabel("Concurrent NEWS-2"); a1.set_ylabel("Windows deteriorating within 6 h (%)")
a1.set_title("A  Outcome prevalence", loc="left", fontweight="bold", fontsize=7); a1.legend(frameon=False, loc="upper left")
for lbl, df, q, off, col, mk in [("Full model (223 features)", A, "auroc full", -0.12, BLUE, "o"),
                                 ("NEWS-2 inputs only (post hoc)", Bb, "auroc baseline", 0.12, ORANGE, "s")]:
    rr = [get(df, q, f"s{k}", "primary" if df is A else None) for k in range(5)]
    pt = np.array([r.point for r in rr]); lo = np.array([r.lo99 for r in rr]); hi = np.array([r.hi99 for r in rr])
    a2.errorbar(x + off, pt, yerr=[pt - lo, hi - pt], fmt=mk, color=col, ms=4, lw=0.9, capsize=2, label=lbl)
gl = get(A, "auroc full", "global", "primary").point
a2.axhline(gl, ls="--", color=DARK, lw=0.7); a2.text(4.45, gl + 0.004, f"Global AUROC {r3(gl)}", ha="right", fontsize=6)
a2.axhline(0.5, ls=":", color=GREY, lw=0.7); a2.text(4.45, 0.493, "NEWS-2 within any stratum (0.5)", ha="right", va="top", fontsize=6, color=GREY)
a2.set_xticks(x); a2.set_xlim(-0.5, 4.5); a2.set_ylim(0.45, 0.85); a2.set_xlabel("Concurrent NEWS-2 (seven-component)")
a2.set_ylabel("Within-score AUROC (99% CI)"); a2.set_title("B  Within-score discrimination", loc="left", fontweight="bold", fontsize=7)
a2.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.22))
save(fig, "Figure3_risk_and_within_score_discrimination", "P2_FIG3")

# ---------------------------------------------------------------- Figure 4
R = np.load(P2_TABLES_BOOT_REPLICATES); K = [ast.literal_eval(k) for k in pd.read_csv(P2_TABLES_BOOT_KEYS).iloc[:, 0]]
col = {k: i for i, k in enumerate(K)}
d = lambda g: R[:, col[("auroc", g, "full")]] - R[:, col[("auroc", g, "no_both")]]
gb6 = d("n6_s0") - d("n6_s4")
gp6 = (get(T, "auroc full", "n6_s0").point - get(T, "auroc no_both", "n6_s0").point) - (get(T, "auroc full", "n6_s4").point - get(T, "auroc no_both", "n6_s4").point)
g7 = get(A, "G = delta_0 - delta_4 (descriptive)", "s0 vs s4", "primary")
fig, (b1, b2) = plt.subplots(1, 2, figsize=(7.1, 2.6), gridspec_kw={"width_ratios": [2.2, 1.1], "wspace": 0.35})
for lbl, rr, off, mk, face in [("Seven-component (primary)", [get(A, "auroc delta_s (full - no_both)", f"s{k}", "primary") for k in range(5)], -0.12, "o", BLUE),
                               ("Six-component (S1)", [get(T, "auroc delta_s (full - no_both)", f"n6_s{k}") for k in range(5)], 0.12, "s", "white")]:
    pt = np.array([r.point for r in rr]); lo = np.array([r.lo99 for r in rr]); hi = np.array([r.hi99 for r in rr])
    b1.errorbar(x + off, pt, yerr=[pt - lo, hi - pt], fmt=mk, color=BLUE, mfc=face, ms=4, lw=0.9, capsize=2, label=lbl)
b1.axhline(0, color=DARK, lw=0.7); b1.set_xticks(x); b1.set_xlim(-0.5, 4.5); b1.set_xlabel("Concurrent NEWS-2")
b1.set_ylabel("ΔAUROC, full − without monitoring (99% CI)"); b1.legend(frameon=False, loc="upper right")
b1.set_title("A  Contribution of monitoring-process features", loc="left", fontweight="bold", fontsize=7)
for i, (p, lo, hi, mk, face) in enumerate([(g7.point, g7.lo95, g7.hi95, "o", BLUE),
                                          (gp6, np.percentile(gb6, 2.5), np.percentile(gb6, 97.5), "s", "white")]):
    b2.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt=mk, color=BLUE, mfc=face, ms=4, lw=0.9, capsize=2)
    b2.text(i, hi + 0.0003, f"{f4(p)}\n({f4(lo)} to {f4(hi)})", fontsize=6, ha="center", va="bottom")
b2.axhline(0, color=DARK, lw=0.7); b2.set_xticks([0, 1]); b2.set_xticklabels(["Seven-\ncomponent", "Six-\ncomponent"])
b2.set_xlim(-0.6, 1.6); b2.set_ylim(-0.0015, 0.0105)
b2.set_ylabel(r"G = $\Delta_0 - \Delta_4$ (95% CI)"); b2.set_title("B  Contrast between scores 0 and 4", loc="left", fontweight="bold", fontsize=7)
save(fig, "Figure4_monitoring_contribution", "P2_FIG4")
