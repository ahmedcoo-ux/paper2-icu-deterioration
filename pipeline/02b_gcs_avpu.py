# Source: S2b.ipynb, code cells [0] (outputs not included)

# %% ---- cell 0 ----


"""
============================================================
S2b — GCS Extraction & AVPU Proxy
============================================================
Pipeline : Rolling Window Pipeline for ICU Deterioration
           Prediction (NEWS-2 Onset Forecasting)
Author   : Ahmed Jassim Jabur
Dataset  : MIMIC-IV v3.1  (CSV / CSV.GZ files)
Step     : S2b of S1 → S8  (runs after S2, before S3)

Reads from:
    pipeline_outputs/s1/cohort.parquet        ← S1 output
    pipeline_outputs/s2/hourly_signals.parquet ← S2 output

Writes to OUTPUT_DIR (pipeline_outputs/s2b/):
    hourly_signals_with_gcs.parquet
        = S2 output + 3 new columns:
            gcs_total   (int, 3–15, NaN if not recorded)
            avpu_score  (int, 0 or 3, NEWS-2 consciousness component)
            gcs_obs     (int8, 1=observed, 0=imputed/missing)
    s2b.log
    s2b_summary.txt

What this step does:
  1. Extracts GCS Motor, Verbal, Eye from chartevents
     (itemids 223901, 223900, 220739 — v3.1 verified)
  2. Computes GCS total = Motor + Verbal + Eye per hour
  3. Applies LOCF with a 4-hour cap (same as vitals)
  4. Maps GCS total → AVPU NEWS-2 score:
       GCS 15        → 0  (Alert)
       GCS 3–14      → 3  (any impairment = +3 in NEWS-2)
  5. Merges onto the hourly_signals DataFrame from S2
  6. Runs sanity checks
  7. Saves hourly_signals_with_gcs.parquet

Why AVPU matters:
  The original S2b document fixes the pipeline gap where
  AVPU was always set to 0 (Alert) for every patient.
  ~35% of ICU stays involve sedation or altered consciousness.
  Incorrect AVPU keeps those patients' NEWS-2 scores
  artificially low, causing some already-deteriorating
  windows to NOT be excluded in S6 — introducing label noise.

GCS → AVPU mapping (Royal College of Physicians NEWS-2):
  GCS 15   → Alert (A)         → 0 points
  GCS < 15 → Any impairment    → 3 points
  Note: New Confusion also scores 3 in NEWS-2; we use
  GCS < 15 as a conservative proxy since RASS/CAM-ICU
  is not reliably available at hourly resolution in MIMIC-IV.
============================================================
"""

import gc
import csv
import gzip
import logging
from pathlib import Path

import numpy  as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
#  ❶  PATHS — edit to match your machine
# ─────────────────────────────────────────────────────────────
MIMIC_DIR   = Path(r"<PHD_ROOT>\PHD Research\DATAset\mimic-iv-3.1")                   # ← CHANGE
S1_DIR      = Path(r"<PHD_ROOT>\PHD Research\DATAset\s1")             # cohort.parquet
S2_DIR      = Path(r"<PHD_ROOT>\PHD Research\DATAset\s2")             # hourly_signals.parquet
OUTPUT_DIR  = Path(r"<PHD_ROOT>\PHD Research\DATAset\s2b")            # output folder

# ─────────────────────────────────────────────────────────────
#  ❷  GCS ITEMIDS  (MIMIC-IV v3.1 verified)
# ─────────────────────────────────────────────────────────────
GCS_ITEMS = {
    223901: "gcs_motor",    # range 1–6
    223900: "gcs_verbal",   # range 1–5
    220739: "gcs_eye",      # range 1–4
}
# GCS total range: 3 (min) – 15 (max)
GCS_MOTOR_RANGE  = (1, 6)
GCS_VERBAL_RANGE = (1, 5)
GCS_EYE_RANGE    = (1, 4)

GCS_LOCF_CAP = 4   # hours — same cap as vital signs

# Chunk size for chartevents re-scan
CHARTEVENTS_CHUNKSIZE = 1_000_000


# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "s2b.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("S2b")


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
        f"  Set MIMIC_DIR at the top of s2b_gcs_avpu.py\n"
        f"  Current MIMIC_DIR = {MIMIC_DIR.resolve()}"
    )


# ─────────────────────────────────────────────────────────────
#  STEP 1 — Load cohort and S2 hourly signals
# ─────────────────────────────────────────────────────────────
def load_inputs() -> tuple:
    """
    Returns (cohort, hourly_signals) DataFrames.
    """
    log.info("─── Loading S1 cohort ───")
    cohort_path = S1_DIR / "cohort.parquet"
    if not cohort_path.exists():
        raise FileNotFoundError(
            f"cohort.parquet not found at {cohort_path.resolve()}\n"
            f"Run S1 first and set S1_DIR correctly."
        )
    cohort = pd.read_parquet(cohort_path)
    cohort["stay_id"] = cohort["stay_id"].astype(int)
    log.info("  Cohort: %d stays", len(cohort))

    log.info("─── Loading S2 hourly signals ───")
    sig_path = S2_DIR / "hourly_signals.parquet"
    if not sig_path.exists():
        raise FileNotFoundError(
            f"hourly_signals.parquet not found at {sig_path.resolve()}\n"
            f"Run S2 first and set S2_DIR correctly."
        )
    signals = pd.read_parquet(sig_path)
    signals["stay_id"] = signals["stay_id"].astype(int)
    log.info("  Hourly signals: %d rows × %d cols", *signals.shape)

    return cohort, signals


# ─────────────────────────────────────────────────────────────
#  STEP 2 — Extract GCS from chartevents (chunked)
# ─────────────────────────────────────────────────────────────
def extract_gcs(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Reads chartevents in chunks.
    Extracts GCS Motor, Verbal, Eye for all cohort stays.
    Returns a wide DataFrame:
        stay_id | hour_offset | gcs_motor | gcs_verbal | gcs_eye | gcs_total
    with one row per (stay_id, hour_offset).
    """
    log.info("─── Extracting GCS from chartevents ───")

    intime_lookup  = dict(zip(cohort["stay_id"], cohort["intime"]))
    los_lookup     = dict(zip(cohort["stay_id"], cohort["los_hours_int"]))
    valid_stay_ids = set(cohort["stay_id"].astype(int))
    all_gcs_ids    = set(GCS_ITEMS.keys())

    # Detect whether 'error' column exists in chartevents
    ce_path = mimic_path("icu", "chartevents")
    opener  = gzip.open if str(ce_path).endswith(".gz") else open
    with opener(ce_path, "rt") as fh:
        header_cols = next(csv.reader(fh))
    has_error = "error" in header_cols

    use_cols = ["stay_id", "itemid", "charttime", "valuenum"]
    if has_error:
        use_cols.append("error")
    log.info("  GCS itemids     : %s", list(GCS_ITEMS.keys()))
    log.info("  error column    : %s", "present" if has_error else "absent")

    records  = []
    n_chunks = 0
    n_rows   = 0

    reader = pd.read_csv(
        ce_path,
        usecols=use_cols,
        parse_dates=["charttime"],
        chunksize=CHARTEVENTS_CHUNKSIZE,
        low_memory=False,
    )

    for chunk in reader:
        n_chunks += 1
        n_rows   += len(chunk)

        # Filter 1: cohort stays only
        chunk = chunk[chunk["stay_id"].isin(valid_stay_ids)].copy()
        if chunk.empty:
            continue

        # Filter 2: GCS itemids only
        chunk = chunk[chunk["itemid"].isin(all_gcs_ids)]
        if chunk.empty:
            continue

        # Filter 3: valid numeric values
        chunk = chunk[chunk["valuenum"].notna() & (chunk["valuenum"] > 0)]
        if chunk.empty:
            continue

        # Filter 4: exclude measurement errors
        if has_error:
            chunk = chunk[chunk["error"].fillna(0) != 1]
        if chunk.empty:
            continue

        # Hour offset from ICU admission
        chunk["intime"]      = chunk["stay_id"].map(intime_lookup)
        chunk["los_hrs_int"] = chunk["stay_id"].map(los_lookup)
        chunk = chunk.dropna(subset=["intime"])

        chunk["hour_offset"] = (
            (chunk["charttime"] - chunk["intime"])
            .dt.total_seconds() / 3600.0
        ).apply(np.floor).astype(int)

        # Keep only hours within stay
        chunk = chunk[
            (chunk["hour_offset"] >= 0) &
            (chunk["hour_offset"] <= chunk["los_hrs_int"])
        ]
        if chunk.empty:
            continue

        # Map itemid → component name
        chunk["component"] = chunk["itemid"].map(GCS_ITEMS)

        records.append(
            chunk[["stay_id", "hour_offset", "component", "valuenum"]].copy()
        )

        if n_chunks % 10 == 0:
            log.info("  chunk %3d | rows read: %9d | records kept: %d",
                     n_chunks, n_rows,
                     sum(len(r) for r in records))

        del chunk
        gc.collect()

    log.info("  Scan complete: %d chunks, %d rows read", n_chunks, n_rows)

    if not records:
        log.warning("  No GCS records found — gcs_total will be all NaN")
        return pd.DataFrame(
            columns=["stay_id", "hour_offset",
                     "gcs_motor", "gcs_verbal", "gcs_eye", "gcs_total"]
        )

    raw = pd.concat(records, ignore_index=True)
    log.info("  Raw GCS records: %d", len(raw))

    # Median per (stay_id, hour_offset, component)
    agg = (
        raw
        .groupby(["stay_id", "hour_offset", "component"], sort=False)["valuenum"]
        .median()
        .reset_index()
    )

    # Pivot to wide: one column per GCS component
    wide = agg.pivot_table(
        index=["stay_id", "hour_offset"],
        columns="component",
        values="valuenum",
        aggfunc="median",
    ).reset_index()
    wide.columns.name = None

    # Ensure all three components present
    for col in ["gcs_motor", "gcs_verbal", "gcs_eye"]:
        if col not in wide.columns:
            wide[col] = np.nan
            log.warning("  GCS component '%s' not found in data", col)

    # Clip each component to its valid range
    wide["gcs_motor"]  = wide["gcs_motor"].clip(*GCS_MOTOR_RANGE)
    wide["gcs_verbal"] = wide["gcs_verbal"].clip(*GCS_VERBAL_RANGE)
    wide["gcs_eye"]    = wide["gcs_eye"].clip(*GCS_EYE_RANGE)

    # GCS total: sum of all three components
    # Require all three to be present for a valid total
    wide["gcs_total"] = wide[["gcs_motor", "gcs_verbal", "gcs_eye"]].sum(
        axis=1, skipna=False   # NaN if any component missing
    )

    # Clip total to valid range (3–15)
    wide["gcs_total"] = wide["gcs_total"].clip(3, 15)

    log.info("  GCS wide table: %d rows", len(wide))
    log.info("  GCS total range: %.0f – %.0f",
             wide["gcs_total"].min(), wide["gcs_total"].max())

    return wide[["stay_id", "hour_offset", "gcs_motor",
                 "gcs_verbal", "gcs_eye", "gcs_total"]]


# ─────────────────────────────────────────────────────────────
#  STEP 3 — Merge GCS onto hourly grid + LOCF + AVPU mapping
# ─────────────────────────────────────────────────────────────
def merge_and_impute_gcs(
    signals: pd.DataFrame,
    gcs_wide: pd.DataFrame,
) -> pd.DataFrame:
    """
    1. Left-joins gcs_wide onto the existing hourly signals
       (preserving ALL rows in signals — the complete hourly grid)
    2. Adds gcs_obs column BEFORE imputation
    3. Applies LOCF with GCS_LOCF_CAP
    4. Maps gcs_total → avpu_score
    5. Returns the merged DataFrame
    """
    log.info("─── Merging GCS onto hourly signals ───")

    df = signals.merge(
        gcs_wide[["stay_id", "hour_offset", "gcs_total"]],
        on=["stay_id", "hour_offset"],
        how="left",
    )
    log.info("  After merge: %d rows × %d cols", *df.shape)

    # ── gcs_obs: 1 if directly observed, 0 if not ───────
    df["gcs_obs"] = df["gcs_total"].notna().astype(np.int8)
    pct_obs = df["gcs_obs"].mean() * 100
    log.info("  GCS observed at %.1f%% of all hours", pct_obs)

    # ── LOCF with 4-hour cap (per stay) ──────────────────
    log.info("  Applying LOCF (cap = %d hours) …", GCS_LOCF_CAP)
    df = df.sort_values(["stay_id", "hour_offset"]).reset_index(drop=True)
    df["gcs_total"] = (
        df.groupby("stay_id", sort=False)["gcs_total"]
        .transform(lambda s: s.ffill(limit=GCS_LOCF_CAP))
    )

    # ── AVPU mapping ─────────────────────────────────────
    #
    #  NEWS-2 consciousness score:
    #    GCS 15   → Alert (A)    → 0 points
    #    GCS < 15 → Impaired     → 3 points
    #    NaN      → 0 points     → assume Alert when unknown
    #              (conservative; avoids penalising patients
    #               with no GCS documentation)
    #
    df["avpu_score"] = np.where(
        df["gcs_total"].isna(),  0,               # unknown → Alert
        np.where(
            df["gcs_total"] >= 15, 0,             # GCS 15  → Alert
            3                                     # GCS <15 → impaired
        )
    ).astype(np.int8)

    n_impaired = int((df["avpu_score"] == 3).sum())
    n_alert    = int((df["avpu_score"] == 0).sum())
    log.info("  AVPU=0 (Alert)   : %d hours (%.1f%%)",
             n_alert, n_alert / len(df) * 100)
    log.info("  AVPU=3 (Impaired): %d hours (%.1f%%)",
             n_impaired, n_impaired / len(df) * 100)

    return df


# ─────────────────────────────────────────────────────────────
#  STEP 4 — Sanity checks
# ─────────────────────────────────────────────────────────────
def sanity_checks(df: pd.DataFrame) -> None:
    log.info("─── S2b Sanity Checks ───")
    failures = []

    # Check 1: gcs_total range (3–15 or NaN)
    valid_gcs = df["gcs_total"].dropna()
    if len(valid_gcs) > 0:
        bad = ((valid_gcs < 3) | (valid_gcs > 15)).sum()
        if bad:
            failures.append(f"Check 1 FAIL: {bad} gcs_total values outside [3, 15]")
        else:
            log.info("  ✓ Check 1: gcs_total in [3, 15] for all non-NaN rows")
    else:
        log.warning("  Check 1: No non-NaN gcs_total values found")

    # Check 2: avpu_score is only 0 or 3
    bad_avpu = (~df["avpu_score"].isin([0, 3])).sum()
    if bad_avpu:
        failures.append(f"Check 2 FAIL: {bad_avpu} avpu_score values not in {{0, 3}}")
    else:
        log.info("  ✓ Check 2: avpu_score is always 0 or 3")

    # Check 3: avpu_score == 0 when gcs_total == 15
    mask_gcs15 = df["gcs_total"] == 15
    bad_alert  = (df.loc[mask_gcs15, "avpu_score"] != 0).sum()
    if bad_alert:
        failures.append(f"Check 3 FAIL: {bad_alert} rows where gcs=15 but avpu≠0")
    else:
        log.info("  ✓ Check 3: avpu_score=0 whenever gcs_total=15")

    # Check 4: avpu_score == 3 when gcs_total < 15 (non-NaN)
    mask_gcs_imp = df["gcs_total"].notna() & (df["gcs_total"] < 15)
    bad_imp      = (df.loc[mask_gcs_imp, "avpu_score"] != 3).sum()
    if bad_imp:
        failures.append(f"Check 4 FAIL: {bad_imp} rows where gcs<15 but avpu≠3")
    else:
        log.info("  ✓ Check 4: avpu_score=3 whenever gcs_total < 15")

    # Check 5: gcs_obs is binary
    bad_obs = (~df["gcs_obs"].isin([0, 1])).sum()
    if bad_obs:
        failures.append(f"Check 5 FAIL: {bad_obs} non-binary values in gcs_obs")
    else:
        log.info("  ✓ Check 5: gcs_obs is binary (0 or 1)")

    # Check 6: row count unchanged from S2
    # (merge must not add or drop rows)
    # We can't check absolute count here, but we check no NaN in stay_id
    bad_sid = df["stay_id"].isna().sum()
    if bad_sid:
        failures.append(f"Check 6 FAIL: {bad_sid} NaN stay_id rows after merge")
    else:
        log.info("  ✓ Check 6: No NaN stay_id rows (merge preserved all rows)")

    if failures:
        msg = "S2b sanity checks FAILED:\n" + "\n".join(failures)
        log.error(msg)
        raise AssertionError(msg)

    log.info("  All 6 S2b sanity checks passed ✓")


# ─────────────────────────────────────────────────────────────
#  STEP 5 — Save output + summary
# ─────────────────────────────────────────────────────────────
def save_output(df: pd.DataFrame) -> None:
    out_path = OUTPUT_DIR / "hourly_signals_with_gcs.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Saved: %s  (%d rows × %d cols)", out_path, *df.shape)

    pct_gcs_obs   = df["gcs_obs"].mean() * 100
    pct_gcs_nan   = df["gcs_total"].isna().mean() * 100
    pct_impaired  = (df["avpu_score"] == 3).mean() * 100
    pct_stays_gcs = (
        df.groupby("stay_id")["gcs_obs"].max() == 1
    ).mean() * 100

    lines = [
        "",
        "=" * 62,
        "  S2b GCS / AVPU SUMMARY",
        "=" * 62,
        f"  Total hourly rows        : {len(df):>10,}",
        f"  Unique stays             : {df['stay_id'].nunique():>10,}",
        f"  Total columns            : {len(df.columns):>10,}",
        "",
        "  GCS coverage:",
        f"    Stays with any GCS     : {pct_stays_gcs:>8.1f}%",
        f"    Hours observed (gcs_obs=1): {pct_gcs_obs:>6.1f}%",
        f"    Hours still NaN after LOCF: {pct_gcs_nan:>6.1f}%",
        f"      (NaN → AVPU=0, assumed Alert)",
        "",
        "  AVPU breakdown:",
        f"    AVPU=0 Alert   (GCS=15 or unknown): {100-pct_impaired:>6.1f}%",
        f"    AVPU=3 Impaired (GCS<15)           : {pct_impaired:>6.1f}%",
        "",
        "  Impact vs original pipeline (AVPU always=0):",
        f"    Hours now correctly scored AVPU=3 : {pct_impaired:>6.1f}%",
        "    These affect NEWS-2 labels in S6.",
        "",
        "  New columns added to hourly signals:",
        "    gcs_total   — GCS sum (3–15, NaN if not recorded/LOCF)",
        "    gcs_obs     — 1 if directly observed, 0 otherwise",
        "    avpu_score  — 0 (Alert) or 3 (Impaired), NEWS-2 component",
        "",
        "  Output: hourly_signals_with_gcs.parquet → feed into S3",
        "=" * 62,
        "",
    ]
    summary = "\n".join(lines)
    log.info(summary)
    (OUTPUT_DIR / "s2b_summary.txt").write_text(summary, encoding="utf-8")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def run_s2b() -> pd.DataFrame:
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  S2b — GCS Extraction & AVPU Proxy              ║")
    log.info("╚══════════════════════════════════════════════════╝")
    log.info("MIMIC_DIR  = %s", MIMIC_DIR.resolve())
    log.info("S1_DIR     = %s", S1_DIR.resolve())
    log.info("S2_DIR     = %s", S2_DIR.resolve())
    log.info("OUTPUT_DIR = %s", OUTPUT_DIR.resolve())

    # Load inputs
    cohort, signals = load_inputs()

    # Extract GCS from chartevents
    gcs_wide = extract_gcs(cohort)

    # Merge onto hourly signals + impute + map AVPU
    df = merge_and_impute_gcs(signals, gcs_wide)

    # Sanity checks
    sanity_checks(df)

    # Save
    save_output(df)

    log.info("S2b DONE — hourly_signals_with_gcs.parquet written.")
    return df


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print()
    print("=" * 62)
    print("  S2b — GCS Extraction & AVPU Proxy")
    print("=" * 62)
    print(f"  MIMIC_DIR  = {MIMIC_DIR.resolve()}")
    print(f"  S2_DIR     = {S2_DIR.resolve()}")
    print(f"  OUTPUT_DIR = {OUTPUT_DIR.resolve()}")
    print()

    # Pre-flight checks
    ok = True
    if not MIMIC_DIR.exists():
        print("  ✗ MIMIC_DIR not found — set it at the top of the file")
        ok = False
    if not (S1_DIR / "cohort.parquet").exists():
        print("  ✗ cohort.parquet not found — run S1 first")
        ok = False
    if not (S2_DIR / "hourly_signals.parquet").exists():
        print("  ✗ hourly_signals.parquet not found — run S2 first")
        ok = False
    if not ok:
        raise SystemExit(1)

    df = run_s2b()

    print()
    print("── Sample (first 3 rows, GCS columns only) ──")
    gcs_cols = ["stay_id", "hour_offset", "gcs_total", "gcs_obs", "avpu_score"]
    print(df[gcs_cols].head(3).to_string())
    print()
    print(f"Shape    : {df.shape}")
    print(f"Columns  : {list(df.columns)}")
    print(f"Saved  → {(OUTPUT_DIR / 'hourly_signals_with_gcs.parquet').resolve()}")
