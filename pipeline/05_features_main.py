# Source: S5.ipynb, code cells [0, 1, 2] (outputs not included)

# %% ---- cell 0 ----
"""
============================================================
S5 — Feature Engineering
============================================================
Pipeline : Rolling Window Pipeline for ICU Deterioration
           Prediction (NEWS-2 Onset Forecasting)
Author   : Ahmed Jassim Jabur
Dataset  : MIMIC-IV v3.1
Step     : S5 of S1 → S8  (runs after S4, before S5b)

Reads from:
    pipeline_outputs/s1/cohort.parquet
    pipeline_outputs/s3/hourly_signals_with_labs.parquet
    pipeline_outputs/s4/windows.parquet

Writes to OUTPUT_DIR (pipeline_outputs/s5/):
    window_features.parquet
        One row per window_id.
        Columns: window_id, stay_id, subject_id, label,
                 + ~194 feature columns

    s5.log
    s5_summary.txt

Features computed per window (input window hours only):
────────────────────────────────────────────────────────
For each of the 8 vital signals (hr, sbp, dbp, map,
rr, spo2, temp, urine_output):
    9 statistics:
        mean, min, max, std, last, slope,
        delta, miss_rate, threshold_pct

For each of the 16 lab signals:
    5 statistics:
        mean, last, slope, delta, miss_rate
    + hours_since_last_obs (already in S3, carried forward)
    + threshold_flag (binary, critical labs only)

GCS / AVPU features:
    gcs_mean, gcs_last, gcs_obs_rate, avpu_sum

Total feature count:
    Vitals (8 × 9)        =  72
    Vital thresholds      =   8
    Labs   (16 × 5)       =  80
    Lab hrs_since (16)    =  16  (last value in window)
    Lab threshold flags   =   6
    GCS/AVPU              =   4
    Static (age)          =   1
    ─────────────────────────────
    Total                 = 187

Slope computation (Fix C3 from gap analysis):
    Slope is computed using OLS on OBSERVED-ONLY hours
    (hours where _obs=1). This prevents imputed values
    from distorting the trend estimate.
    Requires >= 2 observed points; else slope = 0.

Threshold features (NEWS-2 aligned):
    Vital thresholds match NEWS-2 component boundaries:
        spo2    : % hours < 94
        rr      : % hours > 25 or < 8
        sbp     : % hours < 100
        hr      : % hours > 130 or < 40
        temp    : % hours > 39.1 or < 35.0 (Celsius)
    Critical lab thresholds:
        lactate    : any hour >= 2.0 mmol/L
        creatinine : any hour >= 1.5 mg/dL
        potassium  : any hour < 3.0 or > 5.5
        wbc        : any hour > 12 or < 4
        glucose    : any hour < 70 or > 180
        bilirubin  : any hour > 2.0

References:
    9-stat vital features  — Sadanandan (2026) Table 1
    Slope on observed only — Fix C3, Che et al. (2018) GRU-D
    Threshold features     — RCP NEWS-2 (2017)
    Lab features           — Sadanandan (2026) Table 2
    hrs_since feature      — Fix C4, Che et al. (2018)
============================================================
"""

import logging
from pathlib import Path

import numpy  as np
import pandas as pd
from scipy import stats as scipy_stats


# ─────────────────────────────────────────────────────────────
#  ❶  PATHS — edit to match your machine
# ─────────────────────────────────────────────────────────────
MIMIC_DIR   = Path(r"<PHD_ROOT>\PHD Research\DATAset\mimic-iv-3.1")                   # ← CHANGE
S1_DIR      = Path(r"<PHD_ROOT>\PHD Research\DATAset\s1")             # cohort.parquet
S3_DIR      = Path(r"<PHD_ROOT>\PHD Research\DATAset\s3")             # hourly_signals.parquet
S4_DIR    = Path(r"<PHD_ROOT>\PHD Research\DATAset\s4")
OUTPUT_DIR = Path(r"<PHD_ROOT>\PHD Research\DATAset\s5")


# ─────────────────────────────────────────────────────────────
#  ❷  SIGNAL DEFINITIONS
# ─────────────────────────────────────────────────────────────
VITAL_SIGNALS = ["hr", "sbp", "dbp", "map", "rr", "spo2", "temp", "urine_output"]

LAB_SIGNALS = [
    "lactate", "ph", "pco2", "po2",
    "creatinine", "bun", "potassium", "sodium",
    "glucose", "wbc", "hemoglobin", "hematocrit",
    "platelets", "bilirubin", "bicarbonate", "albumin",
]

# NEWS-2 aligned vital thresholds
# Each entry: (signal, condition_fn, feature_name)
VITAL_THRESHOLDS = [
    ("spo2",  lambda x: x  < 94,          "spo2_pct_below94"),
    ("rr",    lambda x: (x > 25) | (x<8), "rr_pct_abnormal"),
    ("sbp",   lambda x: x  < 100,         "sbp_pct_below100"),
    ("hr",    lambda x: (x>130)|(x<40),   "hr_pct_abnormal"),
    ("temp",  lambda x: (x>39.1)|(x<35.0),"temp_pct_abnormal"),
]

# Critical lab threshold flags (binary per window)
LAB_THRESHOLDS = {
    "lactate"   : lambda x: (x >= 2.0).any(),
    "creatinine": lambda x: (x >= 1.5).any(),
    "potassium" : lambda x: ((x < 3.0) | (x > 5.5)).any(),
    "wbc"       : lambda x: ((x > 12)  | (x < 4)).any(),
    "glucose"   : lambda x: ((x < 70)  | (x > 180)).any(),
    "bilirubin" : lambda x: (x > 2.0).any(),
}

# Window size (must match S4)
WINDOW_SIZE = 6

# Batch size for processing windows
BATCH_SIZE = 50_000


# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "s5.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("S5")


# ─────────────────────────────────────────────────────────────
#  STEP 1 — Load inputs
# ─────────────────────────────────────────────────────────────
def load_inputs() -> tuple:
    log.info("─── Loading inputs ───")

    cohort = pd.read_parquet(S1_DIR / "cohort.parquet",
                             columns=["stay_id", "subject_id", "age"])
    cohort["stay_id"] = cohort["stay_id"].astype(int)
    log.info("  Cohort : %d stays", len(cohort))

    signals = pd.read_parquet(S3_DIR / "hourly_signals_with_labs.parquet")
    signals["stay_id"]     = signals["stay_id"].astype(int)
    signals["hour_offset"] = signals["hour_offset"].astype(int)
    log.info("  Signals: %d rows × %d cols", *signals.shape)

    windows = pd.read_parquet(S4_DIR / "windows.parquet")
    windows["stay_id"]  = windows["stay_id"].astype(int)
    windows["window_id"]= windows["window_id"].astype(int)
    log.info("  Windows: %d rows", len(windows))

    return cohort, signals, windows


# ─────────────────────────────────────────────────────────────
#  STEP 2 — Build fast signal lookup
# ─────────────────────────────────────────────────────────────
def build_signal_index(signals: pd.DataFrame) -> dict:
    """
    Groups signals by stay_id into a dict of DataFrames
    for fast per-stay, per-window slicing.
    Returns: { stay_id (int) : DataFrame sorted by hour_offset }
    """
    log.info("─── Building signal index (grouping by stay_id) ───")
    index = {
        sid: grp.set_index("hour_offset").sort_index()
        for sid, grp in signals.groupby("stay_id", sort=False)
    }
    log.info("  Index built for %d stays.", len(index))
    return index


# ─────────────────────────────────────────────────────────────
#  FEATURE COMPUTATION HELPERS
# ─────────────────────────────────────────────────────────────

def _slope_observed_only(values: np.ndarray, obs_mask: np.ndarray) -> float:
    """
    OLS slope computed on observed-only hours (obs_mask == 1).
    Returns 0.0 if fewer than 2 observed points exist.
    Fix C3: prevents imputed values distorting trend estimates.
    Reference: Che et al. (2018) GRU-D.
    """
    hours = np.arange(len(values))
    mask  = obs_mask == 1
    if mask.sum() < 2:
        return 0.0
    x = hours[mask].astype(float)
    y = values[mask]
    # Remove NaN from y (safety)
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return 0.0
    slope, _, _, _, _ = scipy_stats.linregress(x[valid], y[valid])
    return float(slope) if np.isfinite(slope) else 0.0


def _compute_vital_stats(
    window_data: pd.DataFrame,
    signal: str,
) -> dict:
    """
    Computes 9 statistics for one vital signal over one window.
    Returns dict of {feature_name: value}.
    """
    col     = signal
    obs_col = f"{signal}_obs"

    vals = window_data[col].values    if col     in window_data.columns else np.full(WINDOW_SIZE, np.nan)
    obs  = window_data[obs_col].values if obs_col in window_data.columns else np.zeros(WINDOW_SIZE)

    observed_vals = vals[obs == 1]
    n_obs         = len(observed_vals)
    n_total       = len(vals)

    feats = {}
    feats[f"{signal}_mean"]     = float(np.nanmean(vals))     if n_obs > 0 else 0.0
    feats[f"{signal}_min"]      = float(np.nanmin(vals))      if n_obs > 0 else 0.0
    feats[f"{signal}_max"]      = float(np.nanmax(vals))      if n_obs > 0 else 0.0
    feats[f"{signal}_std"]      = float(np.nanstd(vals))      if n_obs > 1 else 0.0
    feats[f"{signal}_last"]     = float(vals[~np.isnan(vals)][-1]) if n_obs > 0 else 0.0
    feats[f"{signal}_slope"]    = _slope_observed_only(vals, obs)
    feats[f"{signal}_delta"]    = (
        float(vals[~np.isnan(vals)][-1] - vals[~np.isnan(vals)][0])
        if n_obs >= 2 else 0.0
    )
    feats[f"{signal}_miss_rate"]= float(1.0 - (n_obs / n_total))

    # Threshold crossing percentage
    # Computed on all non-NaN values (including LOCF-filled)
    non_nan = vals[~np.isnan(vals)]
    feats[f"{signal}_cross_pct"]= 0.0   # placeholder — overwritten below for vitals with thresholds

    return feats


def _compute_vital_threshold(
    window_data: pd.DataFrame,
    signal: str,
    condition_fn,
    feat_name: str,
) -> float:
    """
    Returns the fraction of non-NaN hours in the window
    where condition_fn(value) is True.
    """
    col  = signal
    vals = window_data[col].values if col in window_data.columns else np.array([])
    non_nan = vals[~np.isnan(vals)]
    if len(non_nan) == 0:
        return 0.0
    return float(condition_fn(non_nan).mean())


def _compute_lab_stats(
    window_data: pd.DataFrame,
    lab: str,
) -> dict:
    """
    Computes 5 statistics + hrs_since_last_obs for one lab signal.
    Returns dict of {feature_name: value}.
    """
    col      = lab
    obs_col  = f"{lab}_obs"
    hrs_col  = f"{lab}_hrs_since"

    vals = window_data[col].values     if col     in window_data.columns else np.full(WINDOW_SIZE, np.nan)
    obs  = window_data[obs_col].values if obs_col in window_data.columns else np.zeros(WINDOW_SIZE)
    hrs  = window_data[hrs_col].values if hrs_col in window_data.columns else np.full(WINDOW_SIZE, 48)

    observed_vals = vals[obs == 1]
    n_obs         = len(observed_vals)
    n_total       = len(vals)

    feats = {}
    feats[f"{lab}_mean"]      = float(np.nanmean(vals))          if n_obs > 0 else 0.0
    feats[f"{lab}_last"]      = float(vals[~np.isnan(vals)][-1]) if n_obs > 0 else 0.0
    feats[f"{lab}_slope"]     = _slope_observed_only(vals, obs)
    feats[f"{lab}_delta"]     = (
        float(vals[~np.isnan(vals)][-1] - vals[~np.isnan(vals)][0])
        if n_obs >= 2 else 0.0
    )
    feats[f"{lab}_miss_rate"] = float(1.0 - (n_obs / n_total))

    # hrs_since: take the LAST value in the window
    # (most recent missingness direction signal)
    feats[f"{lab}_hrs_since"] = float(hrs[-1]) if len(hrs) > 0 else 48.0

    return feats


def _compute_gcs_features(window_data: pd.DataFrame) -> dict:
    """
    Computes 4 GCS/AVPU features for the window.
    """
    feats = {}

    if "gcs_total" in window_data.columns:
        gcs  = window_data["gcs_total"].values
        feats["gcs_mean"] = float(np.nanmean(gcs)) if not np.all(np.isnan(gcs)) else 15.0
        non_nan = gcs[~np.isnan(gcs)]
        feats["gcs_last"] = float(non_nan[-1]) if len(non_nan) > 0 else 15.0
    else:
        feats["gcs_mean"] = 15.0
        feats["gcs_last"] = 15.0

    if "gcs_obs" in window_data.columns:
        feats["gcs_obs_rate"] = float(window_data["gcs_obs"].mean())
    else:
        feats["gcs_obs_rate"] = 0.0

    if "avpu_score" in window_data.columns:
        feats["avpu_sum"] = float(window_data["avpu_score"].sum())
    else:
        feats["avpu_sum"] = 0.0

    return feats


# ─────────────────────────────────────────────────────────────
#  STEP 3 — Compute features for one window
# ─────────────────────────────────────────────────────────────
def compute_window_features(
    row:          pd.Series,
    signal_index: dict,
    age_lookup:   dict,
) -> dict:
    """
    Computes all features for one window row.
    Returns a flat dict of {feature_name: value}.
    """
    stay_id      = int(row["stay_id"])
    window_start = int(row["window_start"])
    window_end   = int(row["window_end"])

    feats = {
        "window_id" : int(row["window_id"]),
        "stay_id"   : stay_id,
        "subject_id": int(row["subject_id"]),
        "label"     : int(row["label"]),
    }

    # Get the hourly data for this stay
    stay_data = signal_index.get(stay_id)
    if stay_data is None:
        # Stay not found in signals — return zero features
        return feats

    # Slice the input window [window_start … window_end]
    window_data = stay_data.loc[
        stay_data.index.isin(range(window_start, window_end + 1))
    ]

    # ── Vital statistics (9 per signal) ─────────────────
    for sig in VITAL_SIGNALS:
        sig_feats = _compute_vital_stats(window_data, sig)
        feats.update(sig_feats)

    # ── Vital threshold crossings ────────────────────────
    for sig, cond_fn, feat_name in VITAL_THRESHOLDS:
        feats[feat_name] = _compute_vital_threshold(
            window_data, sig, cond_fn, feat_name
        )

    # ── Lab statistics (5 + hrs_since per lab) ───────────
    for lab in LAB_SIGNALS:
        lab_feats = _compute_lab_stats(window_data, lab)
        feats.update(lab_feats)

    # ── Critical lab threshold flags ─────────────────────
    for lab, flag_fn in LAB_THRESHOLDS.items():
        col  = lab
        vals = window_data[col].values if col in window_data.columns else np.array([])
        non_nan = vals[~np.isnan(vals)]
        feats[f"{lab}_flag"] = int(flag_fn(non_nan)) if len(non_nan) > 0 else 0

    # ── GCS / AVPU ───────────────────────────────────────
    feats.update(_compute_gcs_features(window_data))

    # ── Static: age ──────────────────────────────────────
    feats["age"] = float(age_lookup.get(stay_id, 65.0))

    return feats


# ─────────────────────────────────────────────────────────────
#  STEP 4 — Process all windows in batches
# ─────────────────────────────────────────────────────────────
def compute_all_features(
    windows:      pd.DataFrame,
    signal_index: dict,
    age_lookup:   dict,
) -> pd.DataFrame:
    """
    Iterates over all windows in batches of BATCH_SIZE.
    Returns feature DataFrame with one row per window.
    """
    log.info("─── Computing features for %d windows ───", len(windows))
    log.info("  Batch size: %d", BATCH_SIZE)

    all_records = []
    n_total     = len(windows)
    n_batches   = (n_total + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, n_total)
        batch = windows.iloc[start:end]

        batch_records = []
        for _, row in batch.iterrows():
            batch_records.append(
                compute_window_features(row, signal_index, age_lookup)
            )

        all_records.extend(batch_records)

        if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
            log.info("  Batch %d/%d complete — %d/%d windows processed",
                     batch_idx + 1, n_batches, end, n_total)

    log.info("  All features computed.")
    return pd.DataFrame(all_records)


# ─────────────────────────────────────────────────────────────
#  STEP 5 — Sanity checks
# ─────────────────────────────────────────────────────────────
def sanity_checks(df: pd.DataFrame, windows: pd.DataFrame) -> None:
    log.info("─── S5 Sanity Checks ───")
    failures = []

    # Check 1: row count matches windows
    if len(df) != len(windows):
        failures.append(
            f"Check 1 FAIL: {len(df)} feature rows vs {len(windows)} windows"
        )
    else:
        log.info("  ✓ Check 1: Row count matches windows (%d)", len(df))

    # Check 2: no duplicate window_id
    n_dup = int(df["window_id"].duplicated().sum())
    if n_dup:
        failures.append(f"Check 2 FAIL: {n_dup} duplicate window_ids")
    else:
        log.info("  ✓ Check 2: No duplicate window_ids")

    # Check 3: label column preserved correctly
    label_match = (df["label"].values == windows["label"].values).all()
    if not label_match:
        failures.append("Check 3 FAIL: label column does not match windows")
    else:
        log.info("  ✓ Check 3: Label column matches windows")

    # Check 4: no all-NaN feature columns
    feat_cols = [c for c in df.columns
                 if c not in ["window_id","stay_id","subject_id","label"]]
    all_nan_cols = [c for c in feat_cols if df[c].isna().all()]
    if all_nan_cols:
        failures.append(f"Check 4 FAIL: all-NaN columns: {all_nan_cols[:5]}")
    else:
        log.info("  ✓ Check 4: No all-NaN feature columns (%d features)", len(feat_cols))

    # Check 5: slope features within physiological range
    slope_cols = [c for c in df.columns if c.endswith("_slope")]
    for col in slope_cols:
        bad = ((df[col].abs() > 1000) & df[col].notna()).sum()
        if bad > 0:
            failures.append(
                f"Check 5 FAIL: {col} has {bad} values with |slope| > 1000"
            )
    if not any("Check 5" in f for f in failures):
        log.info("  ✓ Check 5: All slope features within physiological range")

    # Check 6: miss_rate in [0, 1]
    miss_cols = [c for c in df.columns if c.endswith("_miss_rate")]
    for col in miss_cols:
        bad = ((df[col] < 0) | (df[col] > 1)).sum()
        if bad:
            failures.append(f"Check 6 FAIL: {col} has {bad} values outside [0,1]")
    if not any("Check 6" in f for f in failures):
        log.info("  ✓ Check 6: All miss_rate features in [0, 1]")

    # Check 7: threshold flag columns are binary
    flag_cols = [c for c in df.columns if c.endswith("_flag")]
    for col in flag_cols:
        bad = (~df[col].isin([0, 1])).sum()
        if bad:
            failures.append(f"Check 7 FAIL: {col} has {bad} non-binary values")
    if not any("Check 7" in f for f in failures):
        log.info("  ✓ Check 7: All threshold flag columns are binary {0,1}")

    if failures:
        msg = "S5 sanity checks FAILED:\n" + "\n".join(failures)
        log.error(msg)
        raise AssertionError(msg)

    log.info("  All 7 S5 sanity checks passed ✓")


# ─────────────────────────────────────────────────────────────
#  STEP 6 — Save output + summary
# ─────────────────────────────────────────────────────────────
def save_output(df: pd.DataFrame) -> None:
    out_path = OUTPUT_DIR / "window_features.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Saved: %s  (%d rows × %d cols)", out_path, *df.shape)

    feat_cols  = [c for c in df.columns
                  if c not in ["window_id","stay_id","subject_id","label"]]
    n_pos      = int(df["label"].sum())
    n_total    = len(df)
    pos_rate   = n_pos / n_total * 100
    n_nan_pct  = df[feat_cols].isna().mean().mean() * 100

    lines = [
        "",
        "=" * 65,
        "  S5 FEATURE ENGINEERING SUMMARY",
        "=" * 65,
        f"  Total windows        : {n_total:>10,}",
        f"  Total features       : {len(feat_cols):>10,}",
        f"  Label=1 (positive)   : {n_pos:>10,}  ({pos_rate:.2f}%)",
        f"  Label=0 (negative)   : {n_total-n_pos:>10,}  ({100-pos_rate:.2f}%)",
        f"  Avg NaN% across feats: {n_nan_pct:>10.2f}%",
        "",
        "  Feature groups:",
        f"    Vital stats  (8 signals × 9)  : {8*9:>4}",
        f"    Vital thresholds              : {len(VITAL_THRESHOLDS):>4}",
        f"    Lab stats    (16 × 5)         : {16*5:>4}",
        f"    Lab hrs_since (16)            : {16:>4}",
        f"    Lab threshold flags           : {len(LAB_THRESHOLDS):>4}",
        f"    GCS / AVPU                    :    4",
        f"    Static (age)                  :    1",
        f"    ─────────────────────────────────────",
        f"    Total                         : {len(feat_cols):>4}",
        "",
        "  Output: window_features.parquet → feed into S5b (text features)",
        "=" * 65,
        "",
    ]
    summary = "\n".join(lines)
    log.info(summary)
    (OUTPUT_DIR / "s5_summary.txt").write_text(summary, encoding="utf-8")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def run_s5() -> pd.DataFrame:
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  S5 — Feature Engineering                       ║")
    log.info("╚══════════════════════════════════════════════════╝")
    log.info("S1_DIR     = %s", S1_DIR.resolve())
    log.info("S3_DIR     = %s", S3_DIR.resolve())
    log.info("S4_DIR     = %s", S4_DIR.resolve())
    log.info("OUTPUT_DIR = %s", OUTPUT_DIR.resolve())

    # Load
    cohort, signals, windows = load_inputs()

    # Age lookup
    age_lookup = dict(zip(cohort["stay_id"], cohort["age"]))

    # Build signal index
    signal_index = build_signal_index(signals)

    # Compute features
    df = compute_all_features(windows, signal_index, age_lookup)

    # Sanity checks
    sanity_checks(df, windows)

    # Save
    save_output(df)

    log.info("S5 DONE — window_features.parquet written. %d features.", len(df.columns) - 4)
    return df


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print()
    print("=" * 65)
    print("  S5 — Feature Engineering")
    print("=" * 65)
    print(f"  S3_DIR     = {S3_DIR.resolve()}")
    print(f"  S4_DIR     = {S4_DIR.resolve()}")
    print(f"  OUTPUT_DIR = {OUTPUT_DIR.resolve()}")
    print()
    print("  NOTE: This step processes 3.5M windows.")
    print("  Expected runtime: 30–90 minutes depending on CPU.")
    print("  Progress logged every 5 batches.")
    print()

    ok = True
    if not (S1_DIR / "cohort.parquet").exists():
        print("  ✗ cohort.parquet not found — run S1 first")
        ok = False
    if not (S3_DIR / "hourly_signals_with_labs.parquet").exists():
        print("  ✗ hourly_signals_with_labs.parquet not found — run S3 first")
        ok = False
    if not (S4_DIR / "windows.parquet").exists():
        print("  ✗ windows.parquet not found — run S4 first")
        ok = False
    if not ok:
        raise SystemExit(1)

    df = run_s5()

    print()
    print("── Sample (first 3 rows, first 10 feature cols) ──")
    feat_cols = [c for c in df.columns
                 if c not in ["window_id","stay_id","subject_id","label"]]
    preview   = ["window_id","label"] + feat_cols[:8]
    print(df[preview].head(3).to_string())
    print()
    print(f"Shape   : {df.shape}")
    print(f"Features: {len(feat_cols)}")
    print(f"Saved → {(OUTPUT_DIR / 'window_features.parquet').resolve()}")

# %% ---- cell 1 ----
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import gc, os

s2b = pd.read_parquet(r'<PHD_ROOT>\PHD Research\DATAset\s2b\hourly_signals_with_gcs.parquet')
print("s2b columns:", s2b.columns.tolist())
print("s2b shape:  ", s2b.shape)

# %% ---- cell 2 ----

