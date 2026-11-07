# Source: S1.ipynb, code cells [0] (outputs not included)

# %% ---- cell 0 ----
"""
============================================================
S1 — Cohort Selection
============================================================
Pipeline : Rolling Window Pipeline for ICU Deterioration
           Prediction (NEWS-2 Onset Forecasting)
Author   : Ahmed Jassim Jabur
Dataset  : MIMIC-IV v3.1  (CSV / CSV.GZ files)
Step     : S1 of S1 → S8

Expected folder structure:
    mimic-iv-3.1/
        hosp/
            patients.csv.gz       (or .csv)
            admissions.csv.gz
        icu/
            icustays.csv.gz
            chartevents.csv.gz    ← large file, read in chunks

Output (written to OUTPUT_DIR):
    cohort.parquet                ← fed into S2 onwards
    cohort_summary.txt            ← human-readable report
    s1.log                        ← full run log

Inclusion criteria (applied in this order):
  1. Valid intime and outtime timestamps
  2. Age >= 18 years at ICU admission
  3. ICU length of stay >= 12 hours
  4. intime strictly before outtime
  5. Does NOT die within first 6 hours of ICU admission
  6. At least one vital sign recorded within first 6 hours
============================================================
"""

import os
import gc
import logging
from pathlib import Path

import pandas as pd
import numpy  as np


# ─────────────────────────────────────────────────────────────
#  ❶  SET THESE TWO PATHS FOR YOUR MACHINE  ← only thing to edit
# ─────────────────────────────────────────────────────────────
#
#  MIMIC_DIR  = the folder that contains  hosp/  and  icu/
#
#  Windows examples:
#      MIMIC_DIR = Path(r"C:\data\mimic-iv-3.1")
#      MIMIC_DIR = Path(r"D:\PhysioNet\mimic-iv-3.1")
#
#  Linux / Mac examples:
#      MIMIC_DIR = Path("/home/ahmed/data/mimic-iv-3.1")
#      MIMIC_DIR = Path("mimic-iv-3.1")   # relative to script location
#
MIMIC_DIR  = Path(r"<PHD_ROOT>\PHD Research\DATAset\mimic-iv-3.1")          # ← CHANGE THIS LINE
OUTPUT_DIR = Path(r"<PHD_ROOT>\PHD Research\DATAset\s1")   # ← change if you want a different output folder


# ─────────────────────────────────────────────────────────────
#  ❷  COHORT THRESHOLDS
# ─────────────────────────────────────────────────────────────
MIN_LOS_HOURS        = 12   # minimum ICU stay length (hours)
MIN_AGE              = 18   # minimum patient age
EARLY_VITAL_WINDOW_H = 6    # vital sign must exist within first N hours of admission
EARLY_DEATH_EXCL_H   = 6    # exclude if patient dies within first N hours


# ─────────────────────────────────────────────────────────────
#  ❸  VITAL SIGN ITEMIDS  (MIMIC-IV v3.1 verified)
# ─────────────────────────────────────────────────────────────
VITAL_ITEMIDS = {
    220045,   # Heart Rate
    220179,   # Systolic BP  (non-invasive)
    220050,   # Systolic BP  (arterial)
    220180,   # Diastolic BP (non-invasive)
    220051,   # Diastolic BP (arterial)
    220052,   # Mean Arterial Pressure (arterial)
    220181,   # Mean Arterial Pressure (non-invasive)
    220210,   # Respiratory Rate
    220277,   # SpO2 / Pulse Oximetry
    223761,   # Temperature (°F)
    223762,   # Temperature (°C)
}

# Rows per chunk when reading chartevents (large file)
# Reduce to 500_000 if you have < 8 GB RAM
CHARTEVENTS_CHUNKSIZE = 1_000_000


# ─────────────────────────────────────────────────────────────
#  LOGGING  — console + file
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_log_handlers = [
    logging.FileHandler(OUTPUT_DIR / "s1.log", mode="w", encoding="utf-8"),
    logging.StreamHandler(),
]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=_log_handlers,
)
log = logging.getLogger("S1")


# ════════════════════════════════════════════════════════════
#  UTILITY — find a MIMIC file (.csv.gz or .csv)
# ════════════════════════════════════════════════════════════
def mimic_path(subfolder: str, stem: str) -> Path:
    """
    Returns the Path to mimic-iv-3.1/<subfolder>/<stem>.csv.gz
    or mimic-iv-3.1/<subfolder>/<stem>.csv, whichever exists.

    Raises FileNotFoundError with a diagnostic message if neither exists.
    """
    folder = MIMIC_DIR / subfolder
    for ext in (".csv.gz", ".csv"):
        p = folder / (stem + ext)
        if p.exists():
            return p

    # Build a helpful error message
    found_files = sorted(folder.glob("*"))[:20] if folder.exists() else []
    found_str   = "\n".join(f"    {f.name}" for f in found_files) or "    (folder not found)"

    raise FileNotFoundError(
        f"\n\n"
        f"  Cannot find: {stem}.csv.gz  or  {stem}.csv\n"
        f"  Looked in  : {folder.resolve()}\n"
        f"\n"
        f"  Files in that folder:\n"
        f"{found_str}\n"
        f"\n"
        f"  ➜  Fix:  Edit MIMIC_DIR at the top of s1_cohort_selection.py\n"
        f"           It must point to the folder that CONTAINS hosp/ and icu/\n"
        f"           Current value: MIMIC_DIR = {MIMIC_DIR.resolve()}\n"
    )


# ════════════════════════════════════════════════════════════
#  STEP 1 — Verify folder structure
# ════════════════════════════════════════════════════════════
def verify_folder_structure() -> None:
    """
    Confirms the four required files exist before any heavy work starts.
    Fails fast with a precise error if anything is missing.
    """
    log.info("─── Verifying folder structure ───")
    log.info("MIMIC_DIR = %s", MIMIC_DIR.resolve())

    for subfolder, stem in [
        ("hosp", "patients"),
        ("hosp", "admissions"),
        ("icu",  "icustays"),
        ("icu",  "chartevents"),
    ]:
        p = mimic_path(subfolder, stem)
        size_mb = p.stat().st_size / 1_048_576
        log.info("  ✓  %-15s  %s  (%.0f MB)", stem, p.name, size_mb)


# ════════════════════════════════════════════════════════════
#  STEP 2 — Load the three small tables
# ════════════════════════════════════════════════════════════
def load_small_tables() -> tuple:
    """
    Loads icustays, patients, admissions into memory.
    These are small (< 50 MB each) and fit entirely in RAM.
    Returns: (icustays_df, patients_df, admissions_df)
    """
    log.info("─── Loading small tables ───")

    icustays = pd.read_csv(
        mimic_path("icu", "icustays"),
        usecols=["stay_id", "subject_id", "hadm_id",
                 "intime", "outtime", "first_careunit"],
        parse_dates=["intime", "outtime"],
    )
    log.info("  icustays   : %d rows", len(icustays))

    patients = pd.read_csv(
        mimic_path("hosp", "patients"),
        usecols=["subject_id", "anchor_age", "dod"],
        parse_dates=["dod"],
    )
    log.info("  patients   : %d rows", len(patients))

    admissions = pd.read_csv(
        mimic_path("hosp", "admissions"),
        usecols=["hadm_id", "hospital_expire_flag"],
    )
    log.info("  admissions : %d rows", len(admissions))

    return icustays, patients, admissions


# ════════════════════════════════════════════════════════════
#  STEP 3 — Merge into one DataFrame
# ════════════════════════════════════════════════════════════
def build_base_cohort(icustays, patients, admissions) -> pd.DataFrame:
    """
    Left-joins patients and admissions onto icustays.
    Computes los_hours from intime / outtime.
    """
    log.info("─── Merging tables ───")

    df = (
        icustays
        .merge(patients,   on="subject_id", how="left")
        .merge(admissions, on="hadm_id",    how="left")
    )
    df = df.rename(columns={"anchor_age": "age"})

    df["los_hours"] = (
        (df["outtime"] - df["intime"])
        .dt.total_seconds() / 3600.0
    )

    log.info("  Base cohort (unfiltered): %d rows", len(df))
    return df


# ════════════════════════════════════════════════════════════
#  STEP 4 — Apply criteria 1–5
# ════════════════════════════════════════════════════════════
def apply_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies five criteria in sequence.
    Logs how many rows are removed at each step.
    """
    log.info("─── Applying inclusion / exclusion criteria ───")

    def _log(label, df_before, df_after):
        removed = len(df_before) - len(df_after)
        log.info("  %-38s → %7d stays  (removed %d)",
                 label, len(df_after), removed)
        return df_after

    n0 = len(df)
    log.info("  %-38s → %7d stays", "Start", n0)

    # 1. Valid timestamps
    df = df[df["intime"].notna() & df["outtime"].notna()].copy()
    df = _log("Valid intime + outtime", pd.DataFrame(index=range(n0)), df)

    # 2. Age >= 18
    df = _log("Age >= 18", df, df[df["age"] >= MIN_AGE].copy())

    # 3. LOS >= 12 hours
    df = _log("LOS >= 12 h", df, df[df["los_hours"] >= MIN_LOS_HOURS].copy())

    # 4. intime < outtime  (catches zero-length and reversed stays)
    df = _log("intime < outtime", df, df[df["outtime"] > df["intime"]].copy())

    # 5. Exclude deaths within first 6 hours
    #    Method A: dod column (date of death from patients table)
    #    Method B: hospital_expire_flag=1 AND los_hours <= 6
    #              (catches patients where dod is NULL)
    df["death_within_6h"] = False

    has_dod = df["dod"].notna()
    if has_dod.any():
        dod_ts     = pd.to_datetime(df.loc[has_dod, "dod"])
        h_to_death = (dod_ts - df.loc[has_dod, "intime"]).dt.total_seconds() / 3600.0
        df.loc[has_dod, "death_within_6h"] = (
            h_to_death.between(0, EARLY_DEATH_EXCL_H, inclusive="both")
        )

    died_short = (
        (df["hospital_expire_flag"] == 1) &
        (df["los_hours"] <= EARLY_DEATH_EXCL_H)
    )
    df.loc[died_short, "death_within_6h"] = True

    df = _log(
        f"Exclude deaths <= {EARLY_DEATH_EXCL_H} h",
        df,
        df[~df["death_within_6h"]].copy()
    )

    return df


# ════════════════════════════════════════════════════════════
#  STEP 5 — Criterion 6: vital sign within first 6 hours
#  chartevents is ~3 GB compressed — read in chunks
# ════════════════════════════════════════════════════════════
def find_early_vital_stays(intime_lookup: dict) -> set:
    """
    Reads chartevents chunk by chunk.
    Returns the set of stay_ids that have at least one vital sign
    recorded within the first EARLY_VITAL_WINDOW_H hours of ICU admission.

    intime_lookup : dict  { stay_id (int) : intime (Timestamp) }
    """
    log.info("─── Scanning chartevents for early vital signs ───")
    log.info("  Vital itemids checked : %d", len(VITAL_ITEMIDS))
    log.info("  Window                : first %d hours", EARLY_VITAL_WINDOW_H)
    log.info("  Chunk size            : %d rows", CHARTEVENTS_CHUNKSIZE)
    log.info("  (This step takes 5–20 min depending on disk speed)")

    valid_stay_ids = set(intime_lookup.keys())
    qualifying     = set()
    n_chunks       = 0
    n_rows_read    = 0

    # Detect available columns in chartevents before reading chunks
    # (the 'error' column exists in some MIMIC-IV versions but not all)
    import csv
    ce_path = mimic_path("icu", "chartevents")
    with (
        open(ce_path, "rb") if str(ce_path).endswith(".csv")
        else __import__("gzip").open(ce_path, "rt")
    ) as fh:
        header_cols = next(csv.reader(fh))

    use_cols = ["stay_id", "itemid", "charttime"]
    has_error_col = "error" in header_cols
    if has_error_col:
        use_cols.append("error")
        log.info("  'error' column found — will exclude flagged rows")
    else:
        log.info("  'error' column not present in chartevents — skipping error filter")

    reader = pd.read_csv(
        ce_path,
        usecols=use_cols,
        parse_dates=["charttime"],
        chunksize=CHARTEVENTS_CHUNKSIZE,
        low_memory=False,
    )

    for chunk in reader:
        n_chunks    += 1
        n_rows_read += len(chunk)

        # ── keep only rows relevant to our candidate stays ──
        chunk = chunk[chunk["stay_id"].isin(valid_stay_ids)].copy()
        if chunk.empty:
            continue

        # ── keep only vital sign itemids ────────────────────
        chunk = chunk[chunk["itemid"].isin(VITAL_ITEMIDS)]
        if chunk.empty:
            continue

        # ── exclude flagged measurement errors (if column exists) ──
        if has_error_col:
            chunk = chunk[chunk["error"].fillna(0) != 1]
        if chunk.empty:
            continue

        # ── keep only readings within first 6 hours ─────────
        chunk["intime"] = chunk["stay_id"].map(intime_lookup)
        chunk = chunk.dropna(subset=["intime"])

        hours_offset = (
            (chunk["charttime"] - chunk["intime"])
            .dt.total_seconds() / 3600.0
        )
        in_window = hours_offset.between(0, EARLY_VITAL_WINDOW_H, inclusive="both")
        qualifying.update(chunk.loc[in_window, "stay_id"].unique())

        # ── progress log every 10 chunks ────────────────────
        if n_chunks % 10 == 0:
            log.info(
                "  chunk %3d | rows read: %9d | qualifying stays: %d",
                n_chunks, n_rows_read, len(qualifying),
            )

        del chunk
        gc.collect()

    log.info("  Done. Total rows read: %d  |  stays with early vital: %d",
             n_rows_read, len(qualifying))
    return qualifying


# ════════════════════════════════════════════════════════════
#  STEP 6 — Finalise columns
# ════════════════════════════════════════════════════════════
def finalise_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects and adds the columns needed by S2–S8.
    los_hours_int is used by S4 to set the window generation range.
    """
    df = df.copy()
    df["los_hours_int"]       = df["los_hours"].apply(int)
    df["icu_type"]            = df["first_careunit"].str.strip()
    df["expired_in_hospital"] = (df["hospital_expire_flag"] == 1)

    keep = [
        "stay_id",
        "subject_id",
        "hadm_id",
        "intime",
        "outtime",
        "los_hours",
        "los_hours_int",
        "age",
        "icu_type",
        "expired_in_hospital",
        "death_within_6h",     # all False here — kept for audit trail
    ]
    return df[keep].reset_index(drop=True)


# ════════════════════════════════════════════════════════════
#  STEP 7 — Sanity checks
# ════════════════════════════════════════════════════════════
def sanity_checks(df: pd.DataFrame) -> None:
    """
    Five assertions that must ALL pass before cohort.parquet is saved.
    Raises AssertionError (with details) if any check fails.
    Never silently proceed to S2 with a corrupt cohort.
    """
    log.info("─── S1 Sanity Checks ───")
    failures = []

    # 1. No early deaths remain
    n = int(df["death_within_6h"].sum())
    if n != 0:
        failures.append(f"Check 1 FAIL: {n} stays with death_within_6h=True still present")
    else:
        log.info("  ✓ Check 1: No early deaths in cohort")

    # 2. All ages >= 18
    bad_age = int((df["age"] < MIN_AGE).sum())
    if bad_age != 0:
        failures.append(f"Check 2 FAIL: {bad_age} stays with age < {MIN_AGE}")
    else:
        log.info("  ✓ Check 2: All ages >= %d  (min=%d)", MIN_AGE, int(df["age"].min()))

    # 3. All LOS >= 12 h
    bad_los = int((df["los_hours"] < MIN_LOS_HOURS).sum())
    if bad_los != 0:
        failures.append(f"Check 3 FAIL: {bad_los} stays with LOS < {MIN_LOS_HOURS} h")
    else:
        log.info("  ✓ Check 3: All LOS >= %d h  (min=%.2f)", MIN_LOS_HOURS, df["los_hours"].min())

    # 4. No duplicate stay_ids
    n_dup = int(df["stay_id"].duplicated().sum())
    if n_dup != 0:
        failures.append(f"Check 4 FAIL: {n_dup} duplicate stay_ids")
    else:
        log.info("  ✓ Check 4: No duplicate stay_ids")

    # 5. All intime < outtime
    n_bad = int((df["outtime"] <= df["intime"]).sum())
    if n_bad != 0:
        failures.append(f"Check 5 FAIL: {n_bad} stays with outtime <= intime")
    else:
        log.info("  ✓ Check 5: All intime < outtime")

    if failures:
        msg = "S1 sanity checks FAILED — do not proceed to S2:\n" + "\n".join(failures)
        log.error(msg)
        raise AssertionError(msg)

    log.info("  All 5 checks passed ✓")


# ════════════════════════════════════════════════════════════
#  STEP 8 — Save outputs
# ════════════════════════════════════════════════════════════
def save_output(df: pd.DataFrame) -> None:

    # parquet
    parquet_path = OUTPUT_DIR / "cohort.parquet"
    df.to_parquet(parquet_path, index=False)
    log.info("Saved: %s  (%d rows)", parquet_path, len(df))

    # summary text
    n_stays    = len(df)
    n_pts      = df["subject_id"].nunique()
    age_med    = df["age"].median()
    age_q1     = df["age"].quantile(0.25)
    age_q3     = df["age"].quantile(0.75)
    los_med    = df["los_hours"].median()
    los_q1     = df["los_hours"].quantile(0.25)
    los_q3     = df["los_hours"].quantile(0.75)
    n_expired  = int(df["expired_in_hospital"].sum())
    n_lt24     = int((df["los_hours"] < 24).sum())

    lines = [
        "",
        "=" * 60,
        "  S1 COHORT SUMMARY — MIMIC-IV v3.1",
        "=" * 60,
        f"  ICU stays            : {n_stays:>8,}",
        f"  Unique patients      : {n_pts:>8,}",
        f"  Multi-stay patients  : {n_stays - n_pts:>8,}",
        "",
        f"  Age  — median (IQR)  : {age_med:.0f}  ({age_q1:.0f}–{age_q3:.0f})  yrs",
        f"  LOS  — median (IQR)  : {los_med:.1f}  ({los_q1:.1f}–{los_q3:.1f})  hrs",
        "",
        f"  LOS 12–24 h stays    : {n_lt24:>8,}  ({n_lt24/n_stays*100:.1f}%)",
        f"    ^ These are RETAINED (our threshold = 12 h)",
        f"      Sadanandan (2026) uses 24 h and would exclude them",
        "",
        "  ICU type breakdown:",
    ]
    for icu, cnt in df["icu_type"].value_counts().items():
        lines.append(f"    {icu:<42}: {cnt:>6,}  ({cnt/n_stays*100:.1f}%)")
    lines += [
        "",
        f"  Hospital mortality   : {n_expired:>8,}  ({n_expired/n_stays*100:.1f}%)",
        "=" * 60,
        "",
        "  Output: cohort.parquet — feed this file into S2.",
        "",
    ]
    summary = "\n".join(lines)
    log.info(summary)

    txt_path = OUTPUT_DIR / "cohort_summary.txt"
    txt_path.write_text(summary, encoding="utf-8")
    log.info("Saved: %s", txt_path)


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
def run_s1() -> pd.DataFrame:
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  S1 — Cohort Selection  (MIMIC-IV v3.1 CSV/GZ)  ║")
    log.info("╚══════════════════════════════════════════════════╝")

    verify_folder_structure()

    icustays, patients, admissions = load_small_tables()
    df = build_base_cohort(icustays, patients, admissions)
    df = apply_criteria(df)

    # Build intime lookup AFTER criteria filtering
    # (smaller set = faster chartevents scan)
    intime_lookup = dict(zip(df["stay_id"].astype(int), df["intime"]))

    qualifying = find_early_vital_stays(intime_lookup)

    n_before = len(df)
    df = df[df["stay_id"].isin(qualifying)].copy()
    log.info(
        "─── After early vital criterion: %d stays  (removed %d)",
        len(df), n_before - len(df),
    )

    df = finalise_cohort(df)
    sanity_checks(df)
    save_output(df)

    log.info("S1 DONE — %d qualifying ICU stays → cohort.parquet", len(df))
    return df


# ════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print()
    print("=" * 60)
    print("  S1 — Cohort Selection  (MIMIC-IV v3.1)")
    print("=" * 60)
    print(f"  MIMIC_DIR  = {MIMIC_DIR.resolve()}")
    print(f"  OUTPUT_DIR = {OUTPUT_DIR.resolve()}")
    print()

    # Fast path check — give a clear message before touching any data
    if not MIMIC_DIR.exists():
        print("  ✗  ERROR: MIMIC_DIR folder not found.")
        print()
        print("  ➜  Open s1_cohort_selection.py and edit line 54:")
        print()
        print("       MIMIC_DIR = Path(r'mimic-iv-3.1')  ← current value")
        print()
        print("  Replace with the full path to the folder containing hosp/ and icu/")
        print()
        print("  Windows example:  MIMIC_DIR = Path(r'C:\\data\\mimic-iv-3.1')")
        print("  Linux example  :  MIMIC_DIR = Path('/home/ahmed/mimic-iv-3.1')")
        print()
        raise SystemExit(1)

    cohort = run_s1()

    print()
    print("── First 5 rows ──")
    print(cohort.head().to_string())
    print()
    print(f"Shape: {cohort.shape}  →  {cohort.shape[0]:,} stays × {cohort.shape[1]} columns")
    print(f"Saved to: {(OUTPUT_DIR / 'cohort.parquet').resolve()}")
