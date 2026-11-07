# Source: s4.ipynb, code cells [0] (outputs not included)

# %% ---- cell 0 ----
"""
============================================================
S4 — Rolling Window Generation
============================================================
Pipeline : Rolling Window Pipeline for ICU Deterioration
           Prediction (NEWS-2 Onset Forecasting)
Author   : Ahmed Jassim Jabur
Dataset  : MIMIC-IV v3.1
Step     : S4 of S1 → S8  (runs after S3, before S5)

Reads from:
    pipeline_outputs/s1/cohort.parquet
    pipeline_outputs/s3/hourly_signals_with_labs.parquet

Writes to OUTPUT_DIR (pipeline_outputs/s4/):
    windows.parquet
        One row per (stay_id, window_id).
        Columns:
            stay_id         — ICU stay identifier
            subject_id      — patient identifier (for GroupKFold)
            window_id       — unique integer per window
            window_start    — first hour of input window (inclusive)
            window_end      — last  hour of input window (inclusive)
            label_start     — first hour of label horizon (= window_end + 1)
            label_end       — last  hour of label horizon
            [signal]_[stat] — feature columns (added in S5)

        At S4 stage: only index columns written.
        The raw hourly data slice per window is stored as a
        memory-mapped index so S5 can retrieve it efficiently.

    window_index.parquet
        Maps each window to its (stay_id, window_start, window_end)
        for fast slice lookup in S5.

    s4.log
    s4_summary.txt

Window design (from pipeline spec, validated by Fix C1/C2):
    Input window  W = 6  hours  [window_start … window_end]
    Prediction horizon H = 12 hours  [window_end+1 … window_end+12]
    Stride            = 1  hour
    First window start = hour 6  (ensures W hours of history exist)
    Last  window start = los_hours_int − W − H + 1
                         (ensures full H-hour label horizon fits)

    Minimum LOS for any window to exist:
        W + H + first_start = 6 + 12 + 6 = 24 hours → guaranteed
        by S1 (LOS >= 12h) combined with the window bounds.

    Total windows per stay with LOS = L hours:
        max(0, L − W − H − first_start + 2)

Leakage prevention (verified by Sanity Check 2):
    label_start = window_end + 1
    → zero overlap between input window and label horizon
    → a measurement at hour t cannot appear in both window and label

References:
    Window design  — Jiang et al. (2020) Scientific Reports
                     (6+12 config for AKI rolling window)
    Stride=1       — Harutyunyan et al. (2019) Scientific Data
    Leakage check  — McDermott et al. (2021) Science Trans. Med.
    Patient-level  — Harutyunyan et al. (2019)
============================================================
"""

import logging
from pathlib import Path

import numpy  as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
#  ❶  PATHS — edit to match your machine
# ─────────────────────────────────────────────────────────────


MIMIC_DIR   = Path(r"<PHD_ROOT>\PHD Research\DATAset\mimic-iv-3.1")                   # ← CHANGE
S1_DIR      = Path(r"<PHD_ROOT>\PHD Research\DATAset\s1")             # cohort.parquet
S3_DIR      = Path(r"<PHD_ROOT>\PHD Research\DATAset\s3")             # hourly_signals.parquet

OUTPUT_DIR  = Path(r"<PHD_ROOT>\PHD Research\DATAset\s4")

# ─────────────────────────────────────────────────────────────
#  ❷  WINDOW PARAMETERS
# ─────────────────────────────────────────────────────────────
WINDOW_SIZE     = 6    # W — input window length in hours
HORIZON_SIZE    = 6    # H — prediction horizon in hours (changed from 12 → 6)
STRIDE          = 1    # hours between consecutive windows
FIRST_WIN_START = 6    # earliest hour a window can start
                       # (ensures W hours of history from hour 0)

# NEWS-2 thresholds
NEWS2_EXCLUSION_THRESHOLD = 5  # exclude windows where NEWS-2 >= this at window_end
                                # (patient already deteriorating)
NEWS2_LABEL_THRESHOLD     = 7  # label=1 if NEWS-2 >= this in horizon
                                # (changed from 5 → 7: high urgency / severe deterioration)
                                # RCP NEWS-2: >= 7 = highest clinical risk category


# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "s4.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("S4")


# ─────────────────────────────────────────────────────────────
#  STEP 1 — Load inputs
# ─────────────────────────────────────────────────────────────
def load_inputs() -> tuple:
    """
    Returns (cohort, signals) DataFrames.
    cohort  : stay_id, subject_id, los_hours_int
    signals : full hourly grid from S3
    """
    log.info("─── Loading inputs ───")

    cohort = pd.read_parquet(S1_DIR / "cohort.parquet")
    cohort["stay_id"]       = cohort["stay_id"].astype(int)
    cohort["subject_id"]    = cohort["subject_id"].astype(int)
    cohort["los_hours_int"] = cohort["los_hours_int"].astype(int)
    log.info("  Cohort : %d stays", len(cohort))

    signals = pd.read_parquet(S3_DIR / "hourly_signals_with_labs.parquet")
    signals["stay_id"]      = signals["stay_id"].astype(int)
    signals["hour_offset"]  = signals["hour_offset"].astype(int)
    log.info("  Signals: %d rows × %d cols", *signals.shape)

    return cohort, signals


# ─────────────────────────────────────────────────────────────
#  STEP 2 — Compute NEWS-2 score per hour
#           (needed for the onset-only exclusion in S4)
# ─────────────────────────────────────────────────────────────
def compute_news2_hourly(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the hourly NEWS-2 score for every row in signals.
    This is used at window generation time to:
      - Identify windows where NEWS-2 >= 5 at window_end (excluded)
      - Identify windows where NEWS-2 >= 5 in label horizon (label=1)

    NEWS-2 scoring (Royal College of Physicians 2017):
      Respiratory rate (rr):
        <= 8       → 3
         9–11      → 1
        12–20      → 0
        21–24      → 2
        >= 25      → 3
      SpO2 Scale 1 (spo2, no supplemental O2 flag in MIMIC):
        <= 91      → 3
        92–93      → 2
        94–95      → 1
        >= 96      → 0
      Systolic BP (sbp):
        <= 90      → 3
        91–100     → 2
        101–110    → 1
        111–219    → 0
        >= 220     → 3
      Heart rate (hr):
        <= 40      → 3
        41–50      → 1
        51–90      → 0
        91–110     → 1
        111–130    → 2
        >= 131     → 3
      Temperature (temp, Celsius):
        <= 35.0    → 3
        35.1–36.0  → 1
        36.1–38.0  → 0
        38.1–39.0  → 1
        >= 39.1    → 2
      Consciousness (avpu_score):
        Already computed in S2b as 0 or 3

    NaN values in any component → that component contributes 0
    (conservative — avoids inflating score when data is missing)
    """
    log.info("─── Computing hourly NEWS-2 scores ───")

    df = signals[["stay_id", "hour_offset",
                  "rr", "spo2", "sbp", "hr", "temp", "avpu_score"]].copy()

    score = pd.Series(np.zeros(len(df), dtype=np.int8), index=df.index)

    # ── Respiratory rate ──────────────────────────────────
    rr = df["rr"]
    score += np.select(
        [rr.isna(), rr <= 8, rr <= 11, rr <= 20, rr <= 24, rr >= 25],
        [0,          3,       1,         0,         2,        3],
        default=0,
    )

    # ── SpO2 (Scale 1) ────────────────────────────────────
    sp = df["spo2"]
    score += np.select(
        [sp.isna(), sp <= 91, sp <= 93, sp <= 95, sp >= 96],
        [0,          3,        2,        1,         0],
        default=0,
    )

    # ── Systolic BP ───────────────────────────────────────
    sb = df["sbp"]
    score += np.select(
        [sb.isna(), sb <= 90, sb <= 100, sb <= 110, sb <= 219, sb >= 220],
        [0,          3,        2,         1,          0,         3],
        default=0,
    )

    # ── Heart rate ────────────────────────────────────────
    hr = df["hr"]
    score += np.select(
        [hr.isna(), hr <= 40, hr <= 50, hr <= 90, hr <= 110, hr <= 130, hr >= 131],
        [0,          3,        1,        0,         1,         2,         3],
        default=0,
    )

    # ── Temperature (Celsius) ─────────────────────────────
    tp = df["temp"]
    score += np.select(
        [tp.isna(), tp <= 35.0, tp <= 36.0, tp <= 38.0, tp <= 39.0, tp >= 39.1],
        [0,          3,          1,           0,           1,          2],
        default=0,
    )

    # ── Consciousness (AVPU) — from S2b ───────────────────
    score += df["avpu_score"].fillna(0).astype(int)

    df["news2"] = score.clip(0, 20).astype(np.int8)

    log.info("  NEWS-2 computed. Distribution:")
    for threshold in [0, 1, 3, 5, 7]:
        pct = (df["news2"] >= threshold).mean() * 100
        log.info("    NEWS-2 >= %2d : %.1f%% of hours", threshold, pct)

    return df[["stay_id", "hour_offset", "news2"]]


# ─────────────────────────────────────────────────────────────
#  STEP 3 — Generate window index
# ─────────────────────────────────────────────────────────────
def generate_window_index(
    cohort:   pd.DataFrame,
    news2_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each stay, generates all valid (window_start, window_end) pairs.

    A window is VALID if:
      1. window_start >= FIRST_WIN_START  (enough history)
      2. window_end   = window_start + WINDOW_SIZE - 1
      3. label_end    = window_end + HORIZON_SIZE  <= los_hours_int
         (full label horizon fits within the stay)
      4. NEWS-2 at window_end < 5
         (onset-only exclusion — Fix C1 from gap analysis)
         Windows where patient is ALREADY deteriorating are excluded.

    Returns DataFrame with columns:
        stay_id, subject_id, window_id,
        window_start, window_end,
        label_start, label_end
    """
    log.info("─── Generating window index ───")
    log.info("  W=%d h  H=%d h  stride=%d h  first_start=hour %d",
             WINDOW_SIZE, HORIZON_SIZE, STRIDE, FIRST_WIN_START)

    # Build fast NEWS-2 lookup: (stay_id, hour_offset) → news2
    news2_lookup = news2_df.set_index(["stay_id", "hour_offset"])["news2"]

    # Build subject_id lookup
    sid_to_subj = dict(zip(cohort["stay_id"], cohort["subject_id"]))
    sid_to_los  = dict(zip(cohort["stay_id"], cohort["los_hours_int"]))

    records          = []
    window_id        = 0
    n_excluded_news2 = 0
    n_excluded_los   = 0

    for _, row in cohort.iterrows():
        stay_id  = int(row["stay_id"])
        los      = int(row["los_hours_int"])
        subj_id  = int(row["subject_id"])

        # Window start range:
        #   earliest: FIRST_WIN_START
        #   latest  : such that label_end <= los
        #             window_end   = ws + WINDOW_SIZE - 1
        #             label_end    = window_end + HORIZON_SIZE
        #             label_end <= los  →  ws <= los - WINDOW_SIZE - HORIZON_SIZE + 1
        last_start = los - WINDOW_SIZE - HORIZON_SIZE + 1

        if last_start < FIRST_WIN_START:
            n_excluded_los += 1
            continue   # stay too short for even one window

        for ws in range(FIRST_WIN_START, last_start + 1, STRIDE):
            we          = ws + WINDOW_SIZE - 1      # window_end (inclusive)
            label_start = we + 1
            label_end   = we + HORIZON_SIZE

            # ── Onset-only exclusion ─────────────────────
            # Exclude windows where NEWS-2 >= NEWS2_EXCLUSION_THRESHOLD
            # at window_end. Patient is already deteriorating —
            # we cannot model the ONSET of deterioration.
            news2_at_end = news2_lookup.get((stay_id, we), 0)
            if news2_at_end >= NEWS2_EXCLUSION_THRESHOLD:
                n_excluded_news2 += 1
                continue

            records.append({
                "stay_id"     : stay_id,
                "subject_id"  : subj_id,
                "window_id"   : window_id,
                "window_start": ws,
                "window_end"  : we,
                "label_start" : label_start,
                "label_end"   : label_end,
            })
            window_id += 1

    log.info("  Windows generated    : %d", len(records))
    log.info("  Excluded (LOS short) : %d stays", n_excluded_los)
    log.info("  Excluded (NEWS-2>=5 at window_end): %d windows", n_excluded_news2)

    if not records:
        raise RuntimeError("No windows generated — check WINDOW_SIZE, HORIZON_SIZE, LOS thresholds")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
#  STEP 4 — Assign labels
# ─────────────────────────────────────────────────────────────
def assign_labels(
    windows:  pd.DataFrame,
    news2_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each window, label = 1 if NEWS-2 >= 5 occurs at ANY hour
    in [label_start … label_end].

    This is the NEWS-2 onset label:
      - Patient is currently stable (NEWS-2 < 5 at window_end — guaranteed by generation)
      - Label = 1 if they deteriorate (NEWS-2 >= 5) within the next 12 hours
      - Label = 0 if they remain stable throughout the horizon

    Also computes:
      news2_max_in_horizon  — maximum NEWS-2 score in label window
                              (useful for severity analysis)
    """
    log.info("─── Assigning NEWS-2 onset labels ───")

    # Build lookup: stay_id → Series indexed by hour_offset
    news2_by_stay = (
        news2_df
        .set_index(["stay_id", "hour_offset"])["news2"]
    )

    labels         = []
    news2_max_list = []

    for _, row in windows.iterrows():
        sid          = int(row["stay_id"])
        label_start  = int(row["label_start"])
        label_end    = int(row["label_end"])

        # Get all news2 values in label horizon for this stay
        try:
            horizon_scores = news2_by_stay.loc[sid].loc[label_start:label_end]
        except KeyError:
            horizon_scores = pd.Series(dtype=float)

        if len(horizon_scores) == 0:
            # Label horizon extends beyond last recorded hour
            # (patient discharged early) → label = 0 (no event observed)
            labels.append(0)
            news2_max_list.append(0)
        else:
            max_score = int(horizon_scores.max())
            labels.append(1 if max_score >= NEWS2_LABEL_THRESHOLD else 0)
            news2_max_list.append(max_score)

    windows = windows.copy()
    windows["label"]             = np.array(labels, dtype=np.int8)
    windows["news2_max_horizon"] = np.array(news2_max_list, dtype=np.int8)

    n_pos     = int(windows["label"].sum())
    n_total   = len(windows)
    pos_rate  = n_pos / n_total * 100

    log.info("  Total windows : %d", n_total)
    log.info("  Positive (label=1): %d  (%.2f%%)", n_pos, pos_rate)
    log.info("  Negative (label=0): %d  (%.2f%%)", n_total - n_pos, 100 - pos_rate)

    return windows


# ─────────────────────────────────────────────────────────────
#  STEP 5 — Sanity checks
# ─────────────────────────────────────────────────────────────
def sanity_checks(
    windows:  pd.DataFrame,
    news2_df: pd.DataFrame,
    cohort:   pd.DataFrame,
) -> None:
    log.info("─── S4 Sanity Checks ───")
    failures = []

    # Check 1: No news2 >= NEWS2_EXCLUSION_THRESHOLD at window_end
    # (onset-only exclusion must hold for every retained window)
    news2_lookup = news2_df.set_index(["stay_id", "hour_offset"])["news2"]

    sample = windows.sample(min(10_000, len(windows)), random_state=42)
    bad_news2 = 0
    for _, row in sample.iterrows():
        score = news2_lookup.get((int(row["stay_id"]), int(row["window_end"])), 0)
        if score >= NEWS2_EXCLUSION_THRESHOLD:
            bad_news2 += 1
    if bad_news2 > 0:
        failures.append(
            f"Check 1 FAIL: {bad_news2}/10000 sampled windows have "
            f"NEWS-2>={NEWS2_EXCLUSION_THRESHOLD} at window_end"
        )
    else:
        log.info("  ✓ Check 1: No NEWS-2>=%d at window_end (sampled 10,000 windows)",
                 NEWS2_EXCLUSION_THRESHOLD)

    # Check 2: Zero temporal overlap — label_start > window_end
    bad_overlap = (windows["label_start"] <= windows["window_end"]).sum()
    if bad_overlap:
        failures.append(f"Check 2 FAIL: {bad_overlap} windows have label_start <= window_end")
    else:
        log.info("  ✓ Check 2: Zero temporal overlap (label_start > window_end for all)")

    # Check 3: Prevalence in expected range (5–45%)
    # With NEWS-2>=7 label threshold + H=6h horizon,
    # expected prevalence is lower than the original 5–50% range.
    pos_rate = windows["label"].mean() * 100
    if not (5.0 <= pos_rate <= 45.0):
        failures.append(
            f"Check 3 FAIL: prevalence = {pos_rate:.2f}% (expected 5–45%)\n"
            f"    Label threshold : NEWS-2 >= {NEWS2_LABEL_THRESHOLD}\n"
            f"    Horizon         : {HORIZON_SIZE} hours\n"
            f"    If < 5%:  horizon too short or label threshold too high\n"
            f"    If > 45%: label threshold too low or horizon too long"
        )
    else:
        log.info("  ✓ Check 3: Prevalence = %.2f%% (within expected 5–45%%)", pos_rate)

    # Check 4: No patient overlap check
    # Every subject_id must appear in only one split later (enforced by GroupKFold).
    # Here we check that subject_id is correctly populated.
    n_null_subj = windows["subject_id"].isna().sum()
    if n_null_subj:
        failures.append(f"Check 4 FAIL: {n_null_subj} windows have null subject_id")
    else:
        n_unique_subj = windows["subject_id"].nunique()
        log.info("  ✓ Check 4: subject_id populated for all windows (%d unique patients)",
                 n_unique_subj)

    # Check 5: window_end = window_start + WINDOW_SIZE - 1 for all windows
    bad_we = (windows["window_end"] != windows["window_start"] + WINDOW_SIZE - 1).sum()
    if bad_we:
        failures.append(f"Check 5 FAIL: {bad_we} windows have wrong window_end")
    else:
        log.info("  ✓ Check 5: window_end = window_start + %d - 1 for all windows",
                 WINDOW_SIZE)

    # Check 6: label_end = window_end + HORIZON_SIZE for all windows
    bad_le = (windows["label_end"] != windows["window_end"] + HORIZON_SIZE).sum()
    if bad_le:
        failures.append(f"Check 6 FAIL: {bad_le} windows have wrong label_end")
    else:
        log.info("  ✓ Check 6: label_end = window_end + %d for all windows",
                 HORIZON_SIZE)

    # Check 7: label column is binary {0, 1}
    bad_label = (~windows["label"].isin([0, 1])).sum()
    if bad_label:
        failures.append(f"Check 7 FAIL: {bad_label} non-binary label values")
    else:
        log.info("  ✓ Check 7: Labels are binary {0, 1}")

    if failures:
        msg = "S4 sanity checks FAILED:\n" + "\n".join(failures)
        log.error(msg)
        raise AssertionError(msg)

    log.info("  All 7 S4 sanity checks passed ✓")


# ─────────────────────────────────────────────────────────────
#  STEP 6 — Save output + summary
# ─────────────────────────────────────────────────────────────
def save_output(windows: pd.DataFrame) -> None:

    out_path = OUTPUT_DIR / "windows.parquet"
    windows.to_parquet(out_path, index=False)
    log.info("Saved: %s  (%d rows × %d cols)", out_path, *windows.shape)

    n_total   = len(windows)
    n_pos     = int(windows["label"].sum())
    n_neg     = n_total - n_pos
    pos_rate  = n_pos / n_total * 100
    n_stays   = windows["stay_id"].nunique()
    n_patients= windows["subject_id"].nunique()
    wins_per_stay = n_total / n_stays

    lines = [
        "",
        "=" * 65,
        "  S4 ROLLING WINDOW SUMMARY",
        "=" * 65,
        f"  Window parameters:",
        f"    Input window  W = {WINDOW_SIZE} hours",
        f"    Horizon       H = {HORIZON_SIZE} hours",
        f"    Stride          = {STRIDE} hour",
        f"    First start     = hour {FIRST_WIN_START}",
        f"    Exclusion: NEWS-2 >= {NEWS2_EXCLUSION_THRESHOLD} at window_end → excluded",
        f"    Label    : NEWS-2 >= {NEWS2_LABEL_THRESHOLD} in horizon → label=1",
        "",
        f"  Windows generated   : {n_total:>10,}",
        f"  Unique stays        : {n_stays:>10,}",
        f"  Unique patients     : {n_patients:>10,}",
        f"  Avg windows/stay    : {wins_per_stay:>10.1f}",
        "",
        f"  Label distribution:",
        f"    Positive (label=1): {n_pos:>10,}  ({pos_rate:.2f}%)",
        f"    Negative (label=0): {n_neg:>10,}  ({100-pos_rate:.2f}%)",
        "",
        f"  Temporal structure (no overlap guaranteed):",
        f"    Input  : [window_start … window_end]   ({WINDOW_SIZE} hours)",
        f"    Gap    : 0 (label_start = window_end + 1)",
        f"    Label  : [label_start  … label_end]    ({HORIZON_SIZE} hours)",
        "",
        f"  Output: windows.parquet → feed into S5 (feature engineering)",
        "=" * 65,
        "",
    ]
    summary = "\n".join(lines)
    log.info(summary)
    (OUTPUT_DIR / "s4_summary.txt").write_text(summary, encoding="utf-8")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def run_s4() -> pd.DataFrame:
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  S4 — Rolling Window Generation                 ║")
    log.info("╚══════════════════════════════════════════════════╝")
    log.info("S1_DIR     = %s", S1_DIR.resolve())
    log.info("S3_DIR     = %s", S3_DIR.resolve())
    log.info("OUTPUT_DIR = %s", OUTPUT_DIR.resolve())

    # Load
    cohort, signals = load_inputs()

    # Compute hourly NEWS-2
    news2_df = compute_news2_hourly(signals)

    # Generate window index with onset-only exclusion
    windows = generate_window_index(cohort, news2_df)

    # Assign labels
    windows = assign_labels(windows, news2_df)

    # Sanity checks
    sanity_checks(windows, news2_df, cohort)

    # Save
    save_output(windows)

    log.info("S4 DONE — windows.parquet written. %d windows total.", len(windows))
    return windows


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print()
    print("=" * 65)
    print("  S4 — Rolling Window Generation")
    print("=" * 65)
    print(f"  S1_DIR     = {S1_DIR.resolve()}")
    print(f"  S3_DIR     = {S3_DIR.resolve()}")
    print(f"  OUTPUT_DIR = {OUTPUT_DIR.resolve()}")
    print(f"  W={WINDOW_SIZE}h  H={HORIZON_SIZE}h  stride={STRIDE}h  first_start=hour {FIRST_WIN_START}")
    print()

    ok = True
    if not (S1_DIR / "cohort.parquet").exists():
        print("  ✗ cohort.parquet not found — run S1 first")
        ok = False
    if not (S3_DIR / "hourly_signals_with_labs.parquet").exists():
        print("  ✗ hourly_signals_with_labs.parquet not found — run S3 first")
        ok = False
    if not ok:
        raise SystemExit(1)

    windows = run_s4()

    print()
    print("── Sample windows (first 5 rows) ──")
    print(windows.head().to_string())
    print()
    print(f"Shape   : {windows.shape}")
    print(f"Columns : {list(windows.columns)}")
    print(f"Saved → {(OUTPUT_DIR / 'windows.parquet').resolve()}")
