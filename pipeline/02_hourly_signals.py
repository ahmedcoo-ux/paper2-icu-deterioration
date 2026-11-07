# Source: S2.ipynb, code cells [0] (outputs not included)

# %% ---- cell 0 ----
"""
============================================================
S2 — Raw Signal Extraction & Hourly Resampling
============================================================
Pipeline : Rolling Window Pipeline for ICU Deterioration
           Prediction (NEWS-2 Onset Forecasting)
Author   : Ahmed Jassim Jabur
Dataset  : MIMIC-IV v3.1  (CSV / CSV.GZ files)
Step     : S2 of S1 → S8

Reads from:
    pipeline_outputs/s1/cohort.parquet   ← output of S1

Writes to OUTPUT_DIR:
    hourly_signals.parquet   ← one row per (stay_id, hour_offset)
                                columns: stay_id, hour_offset,
                                         hr, sbp, dbp, map,
                                         rr, spo2, temp,
                                         urine_output,
                                         + _obs flag per signal

    s2.log

What this step does:
  1. Reads chartevents in chunks for the 88,713 cohort stays
     → extracts 7 vital signs (HR, SBP, DBP, MAP, RR, SpO2, Temp)
  2. Reads outputevents in one pass
     → extracts urine output (hourly SUM, not mean)
  3. Resamples each signal to a complete hourly grid
     (hour 0 … los_hours_int)  for every stay
  4. Applies LOCF imputation with signal-specific caps:
       vitals  → 4-hour cap
       urine   → 4-hour cap  (zero-fill if never observed)
  5. Adds binary _obs columns (1 = directly observed, 0 = imputed/missing)
     These are used by S5 to compute slope on observed-only values
     and by the missingness-direction feature (hours_since_last_obs)
============================================================
"""

import gc
import logging
from pathlib import Path

import numpy  as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
#  ❶  PATHS  — edit to match your machine
# ─────────────────────────────────────────────────────────────
MIMIC_DIR   = Path(r"<PHD_ROOT>\PHD Research\DATAset\mimic-iv-3.1")                        # ← CHANGE
S1_DIR      = Path(r"<PHD_ROOT>\PHD Research\DATAset\s1")                  # S1 output folder
OUTPUT_DIR  = Path(r"<PHD_ROOT>\PHD Research\DATAset\s2")                  # S2 output folder

# ─────────────────────────────────────────────────────────────
#  ❷  SIGNAL DEFINITIONS
# ─────────────────────────────────────────────────────────────
#
#  Each vital sign maps to one or more MIMIC-IV v3.1 itemids.
#  When multiple itemids exist for the same signal (e.g. invasive
#  vs non-invasive BP), they are merged and the median is taken
#  per hour.  The LOCF_CAP is the maximum number of hours a value
#  is carried forward after the last observed measurement.
#
VITAL_SIGNALS = {
    #  name       : ([itemids],            locf_cap_hours)
    "hr"          : ([220045],             4),
    "sbp"         : ([220179, 220050],     4),   # non-invasive + arterial
    "dbp"         : ([220180, 220051],     4),
    "map"         : ([220181, 220052],     4),
    "rr"          : ([220210],             4),
    "spo2"        : ([220277],             4),
    "temp"        : ([223761, 223762],     4),   # °F and °C  (unified below)
}

# Urine output itemids — hourly SUM (not mean)
# MIMIC-IV v3.1 verified
URINE_ITEMIDS = {
    226559,   # Foley catheter
    226560,   # Void
    226561,   # Condom catheter
    226584,   # Ileoconduit
    226563,   # Suprapubic
    226564,   # Right nephrostomy
    226565,   # Left nephrostomy
    226567,   # Straight cath
    226557,   # R ureteral stent
    226558,   # L ureteral stent
    227488,   # GU irrigant volume in
    227489,   # GU irrigant out
}

URINE_LOCF_CAP = 4   # hours

# Chunk size for reading chartevents
CHARTEVENTS_CHUNKSIZE = 1_000_000


# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "s2.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("S2")


# ─────────────────────────────────────────────────────────────
#  UTILITY
# ─────────────────────────────────────────────────────────────
def mimic_path(subfolder: str, stem: str) -> Path:
    folder = MIMIC_DIR / subfolder
    for ext in (".csv.gz", ".csv"):
        p = folder / (stem + ext)
        if p.exists():
            return p
    raise FileNotFoundError(
        f"\n  Cannot find {stem}.csv.gz or {stem}.csv in {folder.resolve()}\n"
        f"  Set MIMIC_DIR at the top of s2_signal_extraction.py\n"
        f"  Current MIMIC_DIR = {MIMIC_DIR.resolve()}"
    )


def to_celsius(series: pd.Series) -> pd.Series:
    """
    Converts Fahrenheit values to Celsius in-place where value > 45
    (a safe threshold — no human body temperature in Celsius exceeds 45).
    Leaves values already in Celsius unchanged.
    """
    mask = series > 45
    series = series.copy()
    series.loc[mask] = (series.loc[mask] - 32) / 1.8
    return series


# ─────────────────────────────────────────────────────────────
#  STEP 1 — Load cohort from S1
# ─────────────────────────────────────────────────────────────
def load_cohort() -> pd.DataFrame:
    path = S1_DIR / "cohort.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"\n  cohort.parquet not found at {path.resolve()}\n"
            f"  Run s1_cohort_selection.py first.\n"
            f"  Set S1_DIR at the top of this script."
        )
    cohort = pd.read_parquet(path)
    log.info("Cohort loaded: %d stays", len(cohort))

    # Build fast lookup dicts used throughout S2
    cohort["stay_id"]       = cohort["stay_id"].astype(int)
    cohort["los_hours_int"] = cohort["los_hours_int"].astype(int)

    return cohort


# ─────────────────────────────────────────────────────────────
#  STEP 2 — Extract vital signs from chartevents (chunked)
# ─────────────────────────────────────────────────────────────
def extract_vitals(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Reads chartevents in chunks.
    Keeps only rows for cohort stays and vital sign itemids.
    Returns a DataFrame with columns:
        stay_id, hour_offset, signal_name, valuenum
    where hour_offset = floor((charttime - intime) / 1h)
    """
    log.info("─── Extracting vital signs from chartevents ───")

    # Build lookup and sets for fast filtering
    intime_lookup  = dict(zip(cohort["stay_id"], cohort["intime"]))
    los_lookup     = dict(zip(cohort["stay_id"], cohort["los_hours_int"]))
    valid_stay_ids = set(cohort["stay_id"].astype(int))

    # Reverse map: itemid → signal name
    itemid_to_signal = {}
    all_vital_itemids = set()
    for sig_name, (itemids, _) in VITAL_SIGNALS.items():
        for iid in itemids:
            itemid_to_signal[iid] = sig_name
            all_vital_itemids.add(iid)

    # Detect available columns (error col may not exist)
    import csv, gzip
    ce_path = mimic_path("icu", "chartevents")
    opener  = gzip.open if str(ce_path).endswith(".gz") else open
    with opener(ce_path, "rt") as fh:
        header_cols = next(csv.reader(fh))
    has_error = "error" in header_cols

    use_cols = ["stay_id", "itemid", "charttime", "valuenum"]
    if has_error:
        use_cols.append("error")

    log.info("  All vital itemids : %d", len(all_vital_itemids))
    log.info("  Chunk size        : %d rows", CHARTEVENTS_CHUNKSIZE)
    log.info("  error column      : %s", "present" if has_error else "absent")

    records    = []
    n_chunks   = 0
    n_rows_total = 0

    reader = pd.read_csv(
        ce_path,
        usecols=use_cols,
        parse_dates=["charttime"],
        chunksize=CHARTEVENTS_CHUNKSIZE,
        low_memory=False,
    )

    for chunk in reader:
        n_chunks     += 1
        n_rows_total += len(chunk)

        # Filter 1: cohort stays only
        chunk = chunk[chunk["stay_id"].isin(valid_stay_ids)].copy()
        if chunk.empty:
            continue

        # Filter 2: vital itemids only
        chunk = chunk[chunk["itemid"].isin(all_vital_itemids)]
        if chunk.empty:
            continue

        # Filter 3: valid numeric value
        chunk = chunk[chunk["valuenum"].notna() & (chunk["valuenum"] > 0)]
        if chunk.empty:
            continue

        # Filter 4: exclude measurement errors (if column exists)
        if has_error:
            chunk = chunk[chunk["error"].fillna(0) != 1]
        if chunk.empty:
            continue

        # Compute hour_offset from ICU admission
        chunk["intime"]      = chunk["stay_id"].map(intime_lookup)
        chunk["los_hrs_int"] = chunk["stay_id"].map(los_lookup)
        chunk = chunk.dropna(subset=["intime"])

        chunk["hour_offset"] = (
            (chunk["charttime"] - chunk["intime"])
            .dt.total_seconds() / 3600.0
        ).apply(np.floor).astype(int)

        # Keep only hours within the ICU stay (0 … los_hours_int)
        chunk = chunk[
            (chunk["hour_offset"] >= 0) &
            (chunk["hour_offset"] <= chunk["los_hrs_int"])
        ]
        if chunk.empty:
            continue

        # Map itemid → signal name
        chunk["signal"] = chunk["itemid"].map(itemid_to_signal)

        records.append(
            chunk[["stay_id", "hour_offset", "signal", "valuenum"]].copy()
        )

        if n_chunks % 10 == 0:
            log.info("  chunk %3d | rows: %9d | records kept: %d",
                     n_chunks, n_rows_total,
                     sum(len(r) for r in records))

        del chunk
        gc.collect()

    log.info("  chartevents scan complete: %d chunks, %d rows read",
             n_chunks, n_rows_total)

    if not records:
        raise RuntimeError("No vital sign records found — check MIMIC_DIR and cohort stay_ids.")

    vitals_raw = pd.concat(records, ignore_index=True)
    log.info("  Raw vital records : %d", len(vitals_raw))

    # Median per (stay_id, hour_offset, signal) — handles duplicate readings
    vitals_agg = (
        vitals_raw
        .groupby(["stay_id", "hour_offset", "signal"], sort=False)["valuenum"]
        .median()
        .reset_index()
        .rename(columns={"valuenum": "value"})
    )
    log.info("  After hourly aggregation: %d records", len(vitals_agg))

    # Unify temperature: convert °F → °C where value > 45
    mask_temp = vitals_agg["signal"] == "temp"
    vitals_agg.loc[mask_temp, "value"] = to_celsius(
        vitals_agg.loc[mask_temp, "value"]
    )

    return vitals_agg


# ─────────────────────────────────────────────────────────────
#  STEP 3 — Extract urine output from outputevents
# ─────────────────────────────────────────────────────────────
def extract_urine(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Reads outputevents (small enough to load fully).
    Returns hourly SUMS of urine output per stay.
    Columns: stay_id, hour_offset, signal ("urine_output"), value
    """
    log.info("─── Extracting urine output from outputevents ───")

    intime_lookup  = dict(zip(cohort["stay_id"], cohort["intime"]))
    los_lookup     = dict(zip(cohort["stay_id"], cohort["los_hours_int"]))
    valid_stay_ids = set(cohort["stay_id"].astype(int))

    oe_path = mimic_path("icu", "outputevents")

    oe = pd.read_csv(
        oe_path,
        usecols=["stay_id", "itemid", "charttime", "value"],
        parse_dates=["charttime"],
        low_memory=False,
    )
    log.info("  outputevents loaded: %d rows", len(oe))

    # Filter to cohort and urine itemids
    oe = oe[
        oe["stay_id"].isin(valid_stay_ids) &
        oe["itemid"].isin(URINE_ITEMIDS)
    ].copy()
    log.info("  After itemid + cohort filter: %d rows", len(oe))

    if oe.empty:
        log.warning("  No urine output records found — returning empty DataFrame")
        return pd.DataFrame(columns=["stay_id", "hour_offset", "signal", "value"])

    # Numeric value only
    oe["value"] = pd.to_numeric(oe["value"], errors="coerce")
    oe = oe[oe["value"].notna() & (oe["value"] >= 0)]

    # Hour offset
    oe["intime"]      = oe["stay_id"].map(intime_lookup)
    oe["los_hrs_int"] = oe["stay_id"].map(los_lookup)
    oe = oe.dropna(subset=["intime"])

    oe["hour_offset"] = (
        (oe["charttime"] - oe["intime"])
        .dt.total_seconds() / 3600.0
    ).apply(np.floor).astype(int)

    oe = oe[
        (oe["hour_offset"] >= 0) &
        (oe["hour_offset"] <= oe["los_hrs_int"])
    ]

    # Hourly SUM (urine is cumulative, unlike vitals which are point-in-time)
    urine_agg = (
        oe
        .groupby(["stay_id", "hour_offset"], sort=False)["value"]
        .sum()
        .reset_index()
    )
    urine_agg["signal"] = "urine_output"
    urine_agg = urine_agg.rename(columns={"value": "value"})

    log.info("  Urine hourly records: %d", len(urine_agg))
    return urine_agg[["stay_id", "hour_offset", "signal", "value"]]


# ─────────────────────────────────────────────────────────────
#  STEP 4 — Build complete hourly grid per stay
# ─────────────────────────────────────────────────────────────
def build_hourly_grid(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a complete hourly grid for every stay:
        (stay_id, hour_offset)  for  hour_offset in [0, los_hours_int]
    This ensures every hour exists as a row even if no signals
    were recorded — NaN will be filled by LOCF in Step 5.
    """
    log.info("─── Building complete hourly grid ───")

    rows = []
    for _, row in cohort.iterrows():
        sid = int(row["stay_id"])
        los = int(row["los_hours_int"])
        hours = np.arange(0, los + 1, dtype=np.int32)
        rows.append(pd.DataFrame({"stay_id": sid, "hour_offset": hours}))

    grid = pd.concat(rows, ignore_index=True)
    log.info("  Grid shape: %s  (%.1f M rows)",
             grid.shape, len(grid) / 1_000_000)
    return grid


# ─────────────────────────────────────────────────────────────
#  STEP 5 — Pivot signals onto the grid + LOCF imputation
# ─────────────────────────────────────────────────────────────
def pivot_and_impute(
    grid:       pd.DataFrame,
    vitals_agg: pd.DataFrame,
    urine_agg:  pd.DataFrame,
    cohort:     pd.DataFrame,
) -> pd.DataFrame:
    """
    1. Combines vitals and urine into one long DataFrame
    2. Pivots to wide format: one column per signal
    3. Left-joins onto the complete hourly grid
    4. Adds _obs binary columns (1 = observed, 0 = missing/imputed)
    5. Applies LOCF with signal-specific caps
    Returns the final wide DataFrame.
    """
    log.info("─── Pivoting and applying LOCF imputation ───")

    # Combine vitals + urine
    all_signals = pd.concat([vitals_agg, urine_agg], ignore_index=True)

    # Pivot to wide: one row per (stay_id, hour_offset), one col per signal
    wide = all_signals.pivot_table(
        index=["stay_id", "hour_offset"],
        columns="signal",
        values="value",
        aggfunc="median",   # safety: should already be aggregated
    ).reset_index()
    wide.columns.name = None

    log.info("  Wide pivot shape: %s", wide.shape)

    # Left-join onto complete hourly grid
    # NaN in a signal column = no measurement at that hour
    df = grid.merge(wide, on=["stay_id", "hour_offset"], how="left")
    log.info("  After grid merge : %s", df.shape)

    # Ensure all expected signal columns exist (some may be absent
    # if no records were found for that signal)
    all_signal_names = list(VITAL_SIGNALS.keys()) + ["urine_output"]
    for col in all_signal_names:
        if col not in df.columns:
            df[col] = np.nan
            log.warning("  Signal column '%s' missing — filled with NaN", col)

    # ── Add _obs columns BEFORE imputation ────────────────
    # _obs = 1 means the value at this hour was directly observed
    # _obs = 0 means it will be filled by LOCF or remains NaN
    for sig in all_signal_names:
        df[f"{sig}_obs"] = df[sig].notna().astype(np.int8)

    log.info("  _obs columns added for %d signals", len(all_signal_names))

    # ── LOCF imputation per stay ───────────────────────────
    # We group by stay_id and forward-fill within each stay only.
    # The limit= parameter enforces the cap.
    #
    # Signal-specific caps:
    LOCF_CAPS = {sig: cap for sig, (_, cap) in VITAL_SIGNALS.items()}
    LOCF_CAPS["urine_output"] = URINE_LOCF_CAP

    log.info("  Applying LOCF (grouped by stay_id) …")
    df = df.sort_values(["stay_id", "hour_offset"]).reset_index(drop=True)

    for sig, cap in LOCF_CAPS.items():
        df[sig] = (
            df.groupby("stay_id", sort=False)[sig]
            .transform(lambda s: s.ffill(limit=cap))
        )

    # Urine output: hours with no output and no LOCF fill → 0
    # (absence of urine recording means zero output, not missing)
    df["urine_output"] = df["urine_output"].fillna(0.0)

    log.info("  LOCF complete.")

    # ── Clip physiologically implausible values ────────────
    # These bounds are conservative — extreme outliers only.
    CLIP_BOUNDS = {
        "hr"          : (10,  300),
        "sbp"         : (40,  300),
        "dbp"         : (10,  200),
        "map"         : (20,  200),
        "rr"          : (4,   60),
        "spo2"        : (50,  100),
        "temp"        : (25,  45),    # Celsius
        "urine_output": (0,   5000),  # mL/hr
    }
    for sig, (lo, hi) in CLIP_BOUNDS.items():
        if sig in df.columns:
            df[sig] = df[sig].clip(lower=lo, upper=hi)

    log.info("  Physiological clipping applied.")
    return df


# ─────────────────────────────────────────────────────────────
#  STEP 6 — Sanity checks
# ─────────────────────────────────────────────────────────────
def sanity_checks(df: pd.DataFrame, cohort: pd.DataFrame) -> None:
    log.info("─── S2 Sanity Checks ───")
    failures = []

    # Check 1: row count matches sum of (los_hours_int + 1) per stay
    expected_rows = int((cohort["los_hours_int"] + 1).sum())
    actual_rows   = len(df)
    if actual_rows != expected_rows:
        failures.append(
            f"Check 1 FAIL: expected {expected_rows:,} rows, got {actual_rows:,}"
        )
    else:
        log.info("  ✓ Check 1: Row count correct (%d)", actual_rows)

    # Check 2: no duplicate (stay_id, hour_offset) pairs
    n_dup = int(df.duplicated(subset=["stay_id", "hour_offset"]).sum())
    if n_dup != 0:
        failures.append(f"Check 2 FAIL: {n_dup} duplicate (stay_id, hour_offset) pairs")
    else:
        log.info("  ✓ Check 2: No duplicate (stay_id, hour_offset) pairs")

    # Check 3: hour_offset always >= 0
    n_neg = int((df["hour_offset"] < 0).sum())
    if n_neg != 0:
        failures.append(f"Check 3 FAIL: {n_neg} rows with hour_offset < 0")
    else:
        log.info("  ✓ Check 3: All hour_offset >= 0")

    # Check 4: no stay_id in df that is not in cohort
    extra = set(df["stay_id"].unique()) - set(cohort["stay_id"].astype(int))
    if extra:
        failures.append(f"Check 4 FAIL: {len(extra)} stay_ids in df not in cohort")
    else:
        log.info("  ✓ Check 4: All stay_ids are from cohort")

    # Check 5: _obs columns are binary (0 or 1 only)
    obs_cols = [c for c in df.columns if c.endswith("_obs")]
    for col in obs_cols:
        bad = (~df[col].isin([0, 1])).sum()
        if bad:
            failures.append(f"Check 5 FAIL: {col} has {bad} non-binary values")
    if not any(f.startswith("Check 5") for f in failures):
        log.info("  ✓ Check 5: All _obs columns are binary")

    # Check 6: urine_output has no NaN (should be 0-filled)
    n_nan_urine = int(df["urine_output"].isna().sum())
    if n_nan_urine != 0:
        failures.append(f"Check 6 FAIL: {n_nan_urine} NaN values in urine_output")
    else:
        log.info("  ✓ Check 6: urine_output has no NaN values")

    if failures:
        msg = "S2 sanity checks FAILED:\n" + "\n".join(failures)
        log.error(msg)
        raise AssertionError(msg)

    log.info("  All 6 S2 sanity checks passed ✓")


# ─────────────────────────────────────────────────────────────
#  STEP 7 — Save output + summary
# ─────────────────────────────────────────────────────────────
def save_output(df: pd.DataFrame) -> None:
    out_path = OUTPUT_DIR / "hourly_signals.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Saved: %s  (%d rows × %d cols)", out_path, len(df), len(df.columns))

    # ── Summary ──────────────────────────────────────────
    all_signal_names = list(VITAL_SIGNALS.keys()) + ["urine_output"]
    lines = [
        "",
        "=" * 62,
        "  S2 SIGNAL EXTRACTION SUMMARY",
        "=" * 62,
        f"  Total hourly rows    : {len(df):>10,}",
        f"  Unique stays         : {df['stay_id'].nunique():>10,}",
        f"  Columns              : {len(df.columns):>10,}",
        "",
        "  Signal coverage (% of hours with observed value):",
    ]
    for sig in all_signal_names:
        obs_col  = f"{sig}_obs"
        pct_obs  = df[obs_col].mean() * 100 if obs_col in df.columns else float("nan")
        pct_nan  = df[sig].isna().mean() * 100
        lines.append(
            f"    {sig:<15}: {pct_obs:5.1f}% observed  |  {pct_nan:5.1f}% still NaN after LOCF"
        )

    lines += [
        "",
        "  Output: hourly_signals.parquet → feed into S2b (GCS) then S3",
        "=" * 62,
        "",
    ]
    summary = "\n".join(lines)
    log.info(summary)
    (OUTPUT_DIR / "s2_summary.txt").write_text(summary, encoding="utf-8")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def run_s2() -> pd.DataFrame:
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  S2 — Signal Extraction & Hourly Resampling     ║")
    log.info("╚══════════════════════════════════════════════════╝")
    log.info("MIMIC_DIR  = %s", MIMIC_DIR.resolve())
    log.info("S1_DIR     = %s", S1_DIR.resolve())
    log.info("OUTPUT_DIR = %s", OUTPUT_DIR.resolve())

    cohort = load_cohort()

    # Extract signals
    vitals_agg = extract_vitals(cohort)
    urine_agg  = extract_urine(cohort)

    # Build complete grid and pivot
    grid = build_hourly_grid(cohort)
    df   = pivot_and_impute(grid, vitals_agg, urine_agg, cohort)

    # Sanity checks
    sanity_checks(df, cohort)

    # Save
    save_output(df)

    log.info("S2 DONE — hourly_signals.parquet written.")
    return df


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print()
    print("=" * 62)
    print("  S2 — Signal Extraction & Hourly Resampling")
    print("=" * 62)
    print(f"  MIMIC_DIR  = {MIMIC_DIR.resolve()}")
    print(f"  S1_DIR     = {S1_DIR.resolve()}")
    print(f"  OUTPUT_DIR = {OUTPUT_DIR.resolve()}")
    print()

    if not MIMIC_DIR.exists():
        print("  ✗  ERROR: MIMIC_DIR not found.")
        print(f"     Set MIMIC_DIR at the top of s2_signal_extraction.py")
        raise SystemExit(1)

    if not (S1_DIR / "cohort.parquet").exists():
        print("  ✗  ERROR: cohort.parquet not found in S1_DIR.")
        print(f"     Run s1_cohort_selection.py first.")
        print(f"     Then set S1_DIR = Path to where cohort.parquet was saved.")
        raise SystemExit(1)

    df = run_s2()

    print()
    print("── Sample (first 5 rows) ──")
    print(df.head().to_string())
    print()
    print(f"Shape : {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Saved → {(OUTPUT_DIR / 'hourly_signals.parquet').resolve()}")
