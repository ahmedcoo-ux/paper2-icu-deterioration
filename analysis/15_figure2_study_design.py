# Source: paper2.ipynb, code cells [21] (outputs not included)

# %% ---- cell 21 ----
# ============================================================================
# Paper 2 — Figure 2 (study design), panel B counts from saved outputs
# Output: <PHD_ROOT>\Paper2\corrected\Figure2_study_design.png/.pdf
# ============================================================================
import os, sys, importlib, numpy as np, pandas as pd
sys.path.insert(0, r"<PHD_ROOT>")
import paths.shared, paths.paper2; importlib.reload(paths.shared); importlib.reload(paths.paper2); from paths.paper2 import *
from paths.registry import register
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "savefig.dpi": 600})
BLUE, ORANGE, GREY, RED = "#0072B2", "#D55E00", "#666666", "#F4CCC0"
S7 = pd.read_csv(P2_SET_SIZES); prim = S7[S7.analysis == "primary"]
N_HO = int(prim[prim.group == "global"].windows.iloc[0])
NS = [int(prim[prim.group == f"s{k}"].windows.iloc[0]) for k in range(5)]
N_PRIM = sum(NS); N_EXC = N_HO - N_PRIM
assert (N_HO, N_PRIM) == (658_615, 380_728)

fig = plt.figure(figsize=(7.1, 5.6))
# ---------------------------------------------------------------- panel A
ax = fig.add_axes([0.07, 0.56, 0.9, 0.38])
ax.add_patch(Rectangle((0, 0), 5.5, 9.5, color="#E5E5E5", lw=0))
ax.add_patch(Rectangle((5.5, 0), 6, 9.5, color="#DCEBF7", lw=0))
ax.add_patch(Rectangle((11.5, 0), 6, 9.5, color="#FBE8CC", lw=0))
ax.text(2.75, 9.1, "Run-in (0–5 h)\nnot used", ha="center", va="top", color=GREY)
ax.text(8.5, 9.1, "Observation period (6 h)\nfeatures from first 5 h", ha="center", va="top", color=GREY)
ax.text(14.5, 9.1, "Prediction horizon (6 h)", ha="center", va="top", color=GREY)
h = np.arange(0, 25); s = np.array([2, 2, 3, 2, 3, 3, 2, 3, 3, 4, 3, 4, 4, 5, 6, 7, 8, 7, 6, 5, 4, 4, 3, 3, 3])
ax.step(h, s, where="post", color="black", lw=1.2)
ax.axhline(7, ls="--", color=ORANGE, lw=0.8); ax.text(24, 7.15, "7  outcome threshold", ha="right", color=ORANGE, fontsize=6)
ax.axhline(5, ls="--", color=GREY, lw=0.8); ax.text(24, 5.15, "5  generation threshold", ha="right", color=GREY, fontsize=6)
ax.plot(11.5, 4, "o", mfc="white", mec="black", ms=4); ax.plot(16, 8, "o", color=ORANGE, ms=4)
ax.annotate("Concurrent score < 5\n→ window generated", xy=(11.4, 4), xytext=(2.5, 6.2), fontsize=6, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
ax.annotate("NEWS-2 ≥ 7 in horizon\n→ positive label", xy=(16.1, 8), xytext=(19, 2.3), fontsize=6, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
ax.text(24, 0.3, "Windows slide forward hourly →", ha="right", fontsize=6, style="italic", color=GREY)
ax.set_xlim(-0.5, 24.5); ax.set_ylim(0, 9.5); ax.set_xticks(range(0, 25, 2)); ax.set_yticks(range(0, 10))
ax.set_xlabel("Hours since ICU admission (illustrative stay)"); ax.set_ylabel("NEWS-2")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.set_title("A  Two-threshold window generation", loc="left", fontweight="bold", fontsize=7)
# ---------------------------------------------------------------- panel B
bx = fig.add_axes([0.03, 0.02, 0.95, 0.46]); bx.set_xlim(0, 100); bx.set_ylim(0, 50); bx.axis("off")
bx.text(0, 49, "B  Within-score evaluation", fontweight="bold", fontsize=7, va="top")
def box(x, y, w, hgt, text, fc="#F2F2F2"):
    bx.add_patch(FancyBboxPatch((x, y), w, hgt, boxstyle="round,pad=0.4,rounding_size=0.8", fc=fc, ec=GREY, lw=0.6))
    bx.text(x + w / 2, y + hgt / 2, text, ha="center", va="center", fontsize=6)
def arrow(x0, y0, x1, y1): bx.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="black", lw=0.7))
box(1, 33, 24, 11, f"Held-out windows\nn = {N_HO:,}\n\nGenerated and labelled with\nsix-component NEWS-2\n(carry-forward limits, Section 3.4)")
box(31, 33, 24, 11, "Stratify by concurrent\nseven-component NEWS-2\n(+2 if supplemental O₂,\ncarried forward ≤ 4 h)")
box(61, 36, 22, 6, f"NEWS-2 ≥ 5: excluded\nn = {N_EXC:,} ({100 * N_EXC / N_HO:.1f}%)", fc=RED)
arrow(25.8, 38.5, 30.2, 38.5); arrow(55.8, 39, 60.2, 39); arrow(43, 32.2, 43, 26.5)
bx.text(43, 24.5, f"Primary analysis: {N_PRIM:,} windows", ha="center", fontweight="bold", fontsize=6)
for k, n in enumerate(NS):
    box(1 + k * 15.5, 10, 13, 7, f"NEWS-2 = {k}\nn = {n:,}")
arrow(78.5, 13.5, 81.2, 13.5)
box(82, 4, 17, 30, "Within each stratum\n\nAUROC, full model\n(223 features)\n\nΔAUROC after removing\n46 monitoring-process\nfeatures\n\nNEWS-2-inputs model\n(post hoc)\n\n99% CI: patient-level\nbootstrap (2,000)")
bx.text(1, 4, f"Sensitivity analysis: same evaluation with six-component strata\nacross all {N_HO:,} windows (Table 3).", fontsize=6, style="italic", color=GREY, va="top")
for ext in ("png", "pdf"): fig.savefig(os.path.join(P2_OUT, f"Figure2_study_design.{ext}"), bbox_inches="tight")
register("paper2", "P2_FIG2", os.path.join(P2_OUT, "Figure2_study_design.png")); print("saved Figure2_study_design")
