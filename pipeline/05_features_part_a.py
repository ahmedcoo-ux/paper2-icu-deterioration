# Source: S5 Part a.ipynb, code cells [0, 1] (outputs not included)

# %% ---- cell 0 ----
"""
============================================================
S5 Part A — Feature Engineering (First Half of Stays)
============================================================
Run this first, then run s5b_features_partB.py.
After both complete, run s5_merge.py to combine outputs.

Memory strategy:
  - Loads signals for ONE stay at a time from disk
  - Never holds the full 8M-row signals table in RAM
  - Writes results in chunks of CHUNK_SIZE windows
  - Peak RAM usage: ~1.5 GB (safe for 8 GB machines)

This part processes stays in the FIRST HALF of the cohort
sorted by stay_id. Part B processes the second half.
============================================================
"""

import logging
from pathlib import Path

import numpy  as np
import pandas as pd
from scipy import stats as scipy_stats


# ─────────────────────────────────────────────────────────────
#  PATHS — edit to match your machine
# ─────────────────────────────────────────────────────────────
S1_DIR      = Path(r"<PHD_ROOT>\PHD Research\DATAset\s1")             # cohort.parquet
S3_DIR      = Path(r"<PHD_ROOT>\PHD Research\DATAset\s3")             # hourly_signals.parquet
S4_DIR    = Path(r"<PHD_ROOT>\PHD Research\DATAset\s4")
OUTPUT_DIR = Path(r"<PHD_ROOT>\PHD Research\DATAset\s5a")


# ─────────────────────────────────────────────────────────────
#  SIGNAL DEFINITIONS  (same as original S5)
# ─────────────────────────────────────────────────────────────
VITAL_SIGNALS = ["hr", "sbp", "dbp", "map", "rr", "spo2", "temp", "urine_output"]

LAB_SIGNALS = [
    "lactate", "ph", "pco2", "po2",
    "creatinine", "bun", "potassium", "sodium",
    "glucose", "wbc", "hemoglobin", "hematocrit",
    "platelets", "bilirubin", "bicarbonate", "albumin",
]

VITAL_THRESHOLDS = [
    ("spo2",  lambda x: x  < 94,           "spo2_pct_below94"),
    ("rr",    lambda x: (x > 25) | (x < 8),"rr_pct_abnormal"),
    ("sbp",   lambda x: x  < 100,          "sbp_pct_below100"),
    ("hr",    lambda x: (x > 130)|(x < 40),"hr_pct_abnormal"),
    ("temp",  lambda x: (x>39.1)|(x<35.0), "temp_pct_abnormal"),
]

LAB_THRESHOLDS = {
    "lactate"   : lambda x: (x >= 2.0).any(),
    "creatinine": lambda x: (x >= 1.5).any(),
    "potassium" : lambda x: ((x < 3.0) | (x > 5.5)).any(),
    "wbc"       : lambda x: ((x > 12)  | (x < 4)).any(),
    "glucose"   : lambda x: ((x < 70)  | (x > 180)).any(),
    "bilirubin" : lambda x: (x > 2.0).any(),
}

WINDOW_SIZE = 6

# How many windows to accumulate before writing to disk.
# Lower = less RAM. 10,000 is safe for 8 GB machines.
CHUNK_SIZE  = 10_000


# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "s5_partA.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("S5A")


# ─────────────────────────────────────────────────────────────
#  FEATURE HELPERS  (identical to original S5)
# ─────────────────────────────────────────────────────────────
def _slope_observed_only(values: np.ndarray, obs_mask: np.ndarray) -> float:
    hours = np.arange(len(values))
    mask  = obs_mask == 1
    if mask.sum() < 2:
        return 0.0
    x = hours[mask].astype(float)
    y = values[mask]
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return 0.0
    slope, _, _, _, _ = scipy_stats.linregress(x[valid], y[valid])
    return float(slope) if np.isfinite(slope) else 0.0


def compute_window_features(
    row:        pd.Series,
    stay_data:  pd.DataFrame,   # pre-loaded for this stay only
    age_lookup: dict,
) -> dict:
    stay_id      = int(row["stay_id"])
    window_start = int(row["window_start"])
    window_end   = int(row["window_end"])

    feats = {
        "window_id" : int(row["window_id"]),
        "stay_id"   : stay_id,
        "subject_id": int(row["subject_id"]),
        "label"     : int(row["label"]),
    }

    if stay_data is None or stay_data.empty:
        return feats

    # Slice input window
    window_data = stay_data[
        (stay_data["hour_offset"] >= window_start) &
        (stay_data["hour_offset"] <= window_end)
    ].set_index("hour_offset")

    # ── Vital statistics ──────────────────────────────────
    for sig in VITAL_SIGNALS:
        col     = sig
        obs_col = f"{sig}_obs"
        vals = window_data[col].values    if col     in window_data.columns else np.full(WINDOW_SIZE, np.nan)
        obs  = window_data[obs_col].values if obs_col in window_data.columns else np.zeros(WINDOW_SIZE)

        non_nan   = vals[~np.isnan(vals)]
        n_obs     = int((obs == 1).sum())
        n_total   = len(vals)

        feats[f"{sig}_mean"]      = float(np.nanmean(vals))   if n_obs > 0 else 0.0
        feats[f"{sig}_min"]       = float(np.nanmin(vals))    if n_obs > 0 else 0.0
        feats[f"{sig}_max"]       = float(np.nanmax(vals))    if n_obs > 0 else 0.0
        feats[f"{sig}_std"]       = float(np.nanstd(vals))    if n_obs > 1 else 0.0
        feats[f"{sig}_last"]      = float(non_nan[-1])        if n_obs > 0 else 0.0
        feats[f"{sig}_slope"]     = _slope_observed_only(vals, obs)
        feats[f"{sig}_delta"]     = float(non_nan[-1] - non_nan[0]) if n_obs >= 2 else 0.0
        feats[f"{sig}_miss_rate"] = float(1.0 - (n_obs / n_total))
        feats[f"{sig}_cross_pct"] = 0.0   # overwritten below for threshold signals

    # ── Vital threshold crossings ─────────────────────────
    for sig, cond_fn, feat_name in VITAL_THRESHOLDS:
        col  = sig
        vals = window_data[col].values if col in window_data.columns else np.array([])
        non_nan = vals[~np.isnan(vals)]
        feats[feat_name] = float(cond_fn(non_nan).mean()) if len(non_nan) > 0 else 0.0

    # ── Lab statistics ────────────────────────────────────
    for lab in LAB_SIGNALS:
        col      = lab
        obs_col  = f"{lab}_obs"
        hrs_col  = f"{lab}_hrs_since"

        vals = window_data[col].values     if col     in window_data.columns else np.full(WINDOW_SIZE, np.nan)
        obs  = window_data[obs_col].values if obs_col in window_data.columns else np.zeros(WINDOW_SIZE)
        hrs  = window_data[hrs_col].values if hrs_col in window_data.columns else np.full(WINDOW_SIZE, 48)

        non_nan = vals[~np.isnan(vals)]
        n_obs   = int((obs == 1).sum())
        n_total = len(vals)

        feats[f"{lab}_mean"]      = float(np.nanmean(vals))         if n_obs > 0 else 0.0
        feats[f"{lab}_last"]      = float(non_nan[-1])               if n_obs > 0 else 0.0
        feats[f"{lab}_slope"]     = _slope_observed_only(vals, obs)
        feats[f"{lab}_delta"]     = float(non_nan[-1] - non_nan[0]) if n_obs >= 2 else 0.0
        feats[f"{lab}_miss_rate"] = float(1.0 - (n_obs / n_total))
        feats[f"{lab}_hrs_since"] = float(hrs[-1])                   if len(hrs) > 0 else 48.0

    # ── Critical lab threshold flags ──────────────────────
    for lab, flag_fn in LAB_THRESHOLDS.items():
        vals    = window_data[lab].values if lab in window_data.columns else np.array([])
        non_nan = vals[~np.isnan(vals)]
        feats[f"{lab}_flag"] = int(flag_fn(non_nan)) if len(non_nan) > 0 else 0

    # ── GCS / AVPU ────────────────────────────────────────
    if "gcs_total" in window_data.columns:
        gcs     = window_data["gcs_total"].values
        non_nan = gcs[~np.isnan(gcs)]
        feats["gcs_mean"] = float(np.nanmean(gcs)) if len(non_nan) > 0 else 15.0
        feats["gcs_last"] = float(non_nan[-1])      if len(non_nan) > 0 else 15.0
    else:
        feats["gcs_mean"] = 15.0
        feats["gcs_last"] = 15.0

    feats["gcs_obs_rate"] = float(window_data["gcs_obs"].mean())   if "gcs_obs"    in window_data.columns else 0.0
    feats["avpu_sum"]     = float(window_data["avpu_score"].sum()) if "avpu_score" in window_data.columns else 0.0

    # ── Static ────────────────────────────────────────────
    feats["age"] = float(age_lookup.get(stay_id, 65.0))

    return feats


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def run_part_a():
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  S5 Part A — Feature Engineering (First Half)   ║")
    log.info("╚══════════════════════════════════════════════════╝")

    # ── Load index tables (small — safe to load fully) ───
    log.info("Loading cohort and windows …")
    cohort  = pd.read_parquet(S1_DIR / "cohort.parquet",
                              columns=["stay_id", "subject_id", "age"])
    cohort["stay_id"] = cohort["stay_id"].astype(int)
    age_lookup = dict(zip(cohort["stay_id"], cohort["age"]))

    windows = pd.read_parquet(S4_DIR / "windows.parquet")
    windows["stay_id"] = windows["stay_id"].astype(int)
    log.info("  Total windows: %d", len(windows))

    # ── Split stays into two halves ───────────────────────
    all_stays   = sorted(windows["stay_id"].unique())
    half        = len(all_stays) // 2
    stays_partA = set(all_stays[:half])

    windows_A = windows[windows["stay_id"].isin(stays_partA)].reset_index(drop=True)
    log.info("  Part A stays  : %d  |  Part A windows: %d",
             len(stays_partA), len(windows_A))

    # ── Save the stay split so Part B uses the same split ─
    split_path = OUTPUT_DIR / "partA_stay_ids.parquet"
    pd.DataFrame({"stay_id": list(stays_partA)}).to_parquet(split_path, index=False)
    log.info("  Stay split saved → %s", split_path)

    # ── Process stay by stay ──────────────────────────────
    # Load signals for ONE stay at a time — key memory fix
    log.info("Processing windows stay-by-stay (memory efficient) …")

    out_path    = OUTPUT_DIR / "window_features_partA.parquet"
    chunk_buf   = []
    n_done      = 0
    n_total     = len(windows_A)
    first_write = True
    stays_done  = 0
    n_stays     = len(stays_partA)

    # Group windows by stay_id for efficient iteration
    windows_grouped = windows_A.groupby("stay_id", sort=False)

    for stay_id, stay_windows in windows_grouped:
        stays_done += 1

        # Load signals for this stay ONLY
        # Use filters to avoid reading full parquet into RAM
        try:
            stay_signals = pd.read_parquet(
                S3_DIR / "hourly_signals_with_labs.parquet",
                filters=[("stay_id", "=", int(stay_id))],
            )
        except Exception:
            # Fallback if filter pushdown not supported
            stay_signals = pd.read_parquet(
                S3_DIR / "hourly_signals_with_labs.parquet"
            )
            stay_signals = stay_signals[stay_signals["stay_id"] == int(stay_id)]

        stay_signals["hour_offset"] = stay_signals["hour_offset"].astype(int)

        # Compute features for each window of this stay
        for _, row in stay_windows.iterrows():
            feats = compute_window_features(row, stay_signals, age_lookup)
            chunk_buf.append(feats)
            n_done += 1

        # Write chunk to disk when buffer is full
        if len(chunk_buf) >= CHUNK_SIZE:
            chunk_df = pd.DataFrame(chunk_buf)
            if first_write:
                chunk_df.to_parquet(out_path, index=False)
                first_write = False
            else:
                # Append by reading existing + concat + rewrite
                existing = pd.read_parquet(out_path)
                pd.concat([existing, chunk_df], ignore_index=True).to_parquet(
                    out_path, index=False
                )
            chunk_buf = []
            log.info("  Stays: %d/%d | Windows: %d/%d | Written to disk",
                     stays_done, n_stays, n_done, n_total)

        elif stays_done % 1000 == 0:
            log.info("  Stays: %d/%d | Windows: %d/%d",
                     stays_done, n_stays, n_done, n_total)

        del stay_signals

    # Write any remaining buffer
    if chunk_buf:
        chunk_df = pd.DataFrame(chunk_buf)
        if first_write:
            chunk_df.to_parquet(out_path, index=False)
        else:
            existing = pd.read_parquet(out_path)
            pd.concat([existing, chunk_df], ignore_index=True).to_parquet(
                out_path, index=False
            )
        log.info("  Final chunk written. Stays: %d/%d | Windows: %d/%d",
                 stays_done, n_stays, n_done, n_total)

    log.info("Part A DONE → %s  (%d windows)", out_path, n_done)
    log.info("Now run s5_partB.py")


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  S5 Part A — Feature Engineering (First Half)")
    print("=" * 60)

    ok = True
    for p, name in [
        (S1_DIR / "cohort.parquet",                      "cohort.parquet"),
        (S3_DIR / "hourly_signals_with_labs.parquet",    "hourly_signals_with_labs.parquet"),
        (S4_DIR / "windows.parquet",                     "windows.parquet"),
    ]:
        if not p.exists():
            print(f"  ✗ {name} not found at {p.resolve()}")
            ok = False
    if not ok:
        raise SystemExit(1)

    run_part_a()
    print()
    print("Part A complete. Now run s5_partB.py")

# %% ---- cell 1 ----

