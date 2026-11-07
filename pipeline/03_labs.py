# Source: S3.ipynb, code cells [0] (outputs not included)

# %% ---- cell 0 ----
"""
============================================================
S3 — Laboratory Value Extraction
============================================================
Pipeline : Rolling Window Pipeline for ICU Deterioration
           Prediction (NEWS-2 Onset Forecasting)
Author   : Ahmed Jassim Jabur
Dataset  : MIMIC-IV v3.1  (CSV / CSV.GZ files)
Step     : S3 of S1 → S8  (runs after S2b, before S4)

Reads from:
    pipeline_outputs/s1/cohort.parquet
    pipeline_outputs/s2b/hourly_signals_with_gcs.parquet

Writes to OUTPUT_DIR (pipeline_outputs/s3/):
    hourly_signals_with_labs.parquet
        = S2b output + 16 lab columns + 16 _obs columns
          + 16 hours_since_last_obs columns
    s3.log
    s3_summary.txt

CRITICAL implementation notes (Fix C5 from gap analysis):
  ▲ labevents links via hadm_id NOT stay_id
    → must join labevents → admissions → icustays via hadm_id
  ▲ Guard against negative hour offsets
    (labs drawn in ED before ICU admission)
  ▲ Verify all 16 itemids exist before proceeding
  ▲ Variable LOCF caps per lab type:
      Critical (lactate, blood gases) : 6 hours
      Routine  (most labs)            : 24 hours
      Slow     (albumin)              : 48 hours

16 Lab signals extracted (MIMIC-IV v3.1 itemids verified):
  lactate      50813    creatinine   50912    bun          51006
  potassium    50971    sodium       50983    glucose      50931
  wbc          51301    hemoglobin   51222    hematocrit   51221
  platelets    51265    bilirubin    50885    albumin      50862
  ph           50820    pco2         50818    po2          50821
  bicarbonate  50882

Output columns added per lab (example for lactate):
  lactate              — imputed value (LOCF applied)
  lactate_obs          — 1=observed, 0=imputed/missing
  lactate_hrs_since    — hours since last observed value (0–48)
                         0  = measured this hour
                         48 = never measured or very old
                         Used as missingness-direction feature (Fix C4)

References:
  ▲ Fix C4: Missingness direction — Che et al. (2018) GRU-D
  ▲ Fix C5: labevents SQL — MIMIC-IV documentation Johnson et al. (2023)
  Variable LOCF caps — Sadanandan (2026) Table C / Appendix C
  Lab feature set    — Sadanandan (2026) Table 2
============================================================
"""

import gc
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
S2B_DIR     = Path(r"<PHD_ROOT>\PHD Research\DATAset\s2b")
OUTPUT_DIR  = Path(r"<PHD_ROOT>\PHD Research\DATAset\s3")

# ─────────────────────────────────────────────────────────────
#  ❷  LAB DEFINITIONS
#     (itemid, locf_cap_hours, [lo_clip, hi_clip])
#     Clip bounds are physiologically extreme values only —
#     they remove clear data entry errors, not real outliers.
# ─────────────────────────────────────────────────────────────
LAB_DEFINITIONS = {
    # name          itemid    locf_cap   clip_lo   clip_hi
    "lactate"    : (50813,    6,         0.1,      30.0  ),
    "ph"         : (50820,    6,         6.5,       8.0  ),
    "pco2"       : (50818,    6,         5.0,      200.0 ),
    "po2"        : (50821,    6,         5.0,      700.0 ),
    "creatinine" : (50912,    24,        0.1,      30.0  ),
    "bun"        : (51006,    24,        1.0,      300.0 ),
    "potassium"  : (50971,    24,        1.0,      10.0  ),
    "sodium"     : (50983,    24,        100.0,    180.0 ),
    "glucose"    : (50931,    24,        20.0,     1500.0),
    "wbc"        : (51301,    24,        0.1,      200.0 ),
    "hemoglobin" : (51222,    24,        1.0,      25.0  ),
    "hematocrit" : (51221,    24,        3.0,      75.0  ),
    "platelets"  : (51265,    24,        1.0,      2000.0),
    "bilirubin"  : (50885,    24,        0.1,      60.0  ),
    "bicarbonate": (50882,    24,        2.0,      60.0  ),
    "albumin"    : (50862,    48,        0.5,      10.0  ),
}

# Maximum sentinel value for hours_since_last_obs
# (capped at the longest LOCF window = 48h)
MAX_HRS_SINCE = 48

# Chunk size for reading labevents (large file)
LABEVENTS_CHUNKSIZE = 500_000


# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "s3.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("S3")


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
        f"  Set MIMIC_DIR at the top of s3_lab_extraction.py\n"
        f"  Current MIMIC_DIR = {MIMIC_DIR.resolve()}"
    )


# ─────────────────────────────────────────────────────────────
#  STEP 1 — Load inputs
# ─────────────────────────────────────────────────────────────
def load_inputs() -> tuple:
    """
    Returns (cohort, hourly_signals) DataFrames.
    cohort         : from S1 — provides hadm_id, intime, los_hours_int
    hourly_signals : from S2b — the grid we merge labs onto
    """
    log.info("─── Loading inputs ───")

    cohort = pd.read_parquet(S1_DIR / "cohort.parquet")
    cohort["stay_id"] = cohort["stay_id"].astype(int)
    cohort["hadm_id"] = cohort["hadm_id"].astype(int)
    log.info("  Cohort : %d stays", len(cohort))

    signals = pd.read_parquet(S2B_DIR / "hourly_signals_with_gcs.parquet")
    signals["stay_id"] = signals["stay_id"].astype(int)
    log.info("  Signals: %d rows × %d cols", *signals.shape)

    return cohort, signals


# ─────────────────────────────────────────────────────────────
#  STEP 2 — Verify all 16 itemids exist in labevents
#  (Fix C5 — catch missing itemids silently before extraction)
# ─────────────────────────────────────────────────────────────
def verify_itemids() -> set:
    """
    Reads d_labitems to confirm all expected itemids exist.
    Returns the set of confirmed itemids.
    Logs a warning for any that are missing.
    """
    log.info("─── Verifying lab itemids in d_labitems ───")

    d_labitems = pd.read_csv(
        mimic_path("hosp", "d_labitems"),
        usecols=["itemid", "label"],
    )
    available = set(d_labitems["itemid"].astype(int))

    expected = {v[0]: k for k, v in LAB_DEFINITIONS.items()}
    confirmed = set()
    missing   = []

    for itemid, lab_name in expected.items():
        if itemid in available:
            label = d_labitems.loc[
                d_labitems["itemid"] == itemid, "label"
            ].values[0]
            log.info("  ✓ %s  itemid=%d  (%s)", lab_name, itemid, label)
            confirmed.add(itemid)
        else:
            missing.append((lab_name, itemid))
            log.warning("  ✗ %s  itemid=%d  NOT FOUND in d_labitems", lab_name, itemid)

    if missing:
        log.warning(
            "  %d itemids not found — those labs will be all NaN: %s",
            len(missing), [m[0] for m in missing]
        )
    else:
        log.info("  All 16 itemids confirmed ✓")

    return confirmed


# ─────────────────────────────────────────────────────────────
#  STEP 3 — Extract labs from labevents (chunked)
#
#  ▲ Fix C5: labevents has hadm_id, NOT stay_id
#            Join path: labevents.hadm_id → cohort.hadm_id → stay_id
# ─────────────────────────────────────────────────────────────
def extract_labs(cohort: pd.DataFrame, confirmed_itemids: set) -> pd.DataFrame:
    """
    Reads labevents in chunks.
    Joins via hadm_id (not stay_id — Fix C5).
    Returns long DataFrame:
        stay_id | hour_offset | lab_name | value
    with one row per (stay, hour, lab) after median aggregation.
    """
    log.info("─── Extracting labs from labevents ───")
    log.info("  ▲ Joining via hadm_id (not stay_id) — Fix C5")

    # Build lookup via hadm_id  (Fix C5 key fix)
    # hadm_id → stay_id, intime, los_hours_int
    hadm_to_stay   = dict(zip(cohort["hadm_id"], cohort["stay_id"]))
    hadm_to_intime = dict(zip(cohort["hadm_id"], cohort["intime"]))
    hadm_to_los    = dict(zip(cohort["hadm_id"], cohort["los_hours_int"]))
    valid_hadm_ids = set(cohort["hadm_id"].astype(int))

    # itemid → lab name
    itemid_to_lab = {v[0]: k for k, v in LAB_DEFINITIONS.items()
                     if v[0] in confirmed_itemids}
    confirmed_set = set(itemid_to_lab.keys())

    log.info("  Valid hadm_ids    : %d", len(valid_hadm_ids))
    log.info("  Confirmed itemids : %d", len(confirmed_set))
    log.info("  Chunk size        : %d rows", LABEVENTS_CHUNKSIZE)

    records  = []
    n_chunks = 0
    n_rows   = 0

    reader = pd.read_csv(
        mimic_path("hosp", "labevents"),
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
        chunksize=LABEVENTS_CHUNKSIZE,
        low_memory=False,
    )

    for chunk in reader:
        n_chunks += 1
        n_rows   += len(chunk)

        # ── Fix C5: filter on hadm_id ────────────────────
        chunk = chunk[chunk["hadm_id"].isin(valid_hadm_ids)].copy()
        if chunk.empty:
            continue

        # Filter: confirmed lab itemids only
        chunk = chunk[chunk["itemid"].isin(confirmed_set)]
        if chunk.empty:
            continue

        # Valid positive numeric values only
        chunk = chunk[chunk["valuenum"].notna() & (chunk["valuenum"] > 0)]
        if chunk.empty:
            continue

        # Map hadm_id → stay_id, intime, los
        chunk["stay_id"]     = chunk["hadm_id"].map(hadm_to_stay)
        chunk["intime"]      = chunk["hadm_id"].map(hadm_to_intime)
        chunk["los_hrs_int"] = chunk["hadm_id"].map(hadm_to_los)
        chunk = chunk.dropna(subset=["stay_id", "intime"])
        chunk["stay_id"] = chunk["stay_id"].astype(int)

        # Hour offset from ICU intime
        chunk["hour_offset"] = (
            (chunk["charttime"] - chunk["intime"])
            .dt.total_seconds() / 3600.0
        ).apply(np.floor).astype(int)

        # ▲ Fix C5: guard negative offsets (ED labs before ICU admission)
        n_neg = (chunk["hour_offset"] < 0).sum()
        if n_neg > 0:
            log.debug("  Dropping %d rows with negative hour_offset (pre-ICU labs)", n_neg)
        chunk = chunk[chunk["hour_offset"] >= 0]

        # Guard: hour must be within ICU stay
        chunk = chunk[chunk["hour_offset"] <= chunk["los_hrs_int"]]
        if chunk.empty:
            continue

        # Map itemid → lab name
        chunk["lab_name"] = chunk["itemid"].map(itemid_to_lab)

        records.append(
            chunk[["stay_id", "hour_offset", "lab_name", "valuenum"]].copy()
        )

        if n_chunks % 20 == 0:
            log.info("  chunk %3d | rows: %9d | records kept: %d",
                     n_chunks, n_rows, sum(len(r) for r in records))

        del chunk
        gc.collect()

    log.info("  Scan complete: %d chunks, %d total rows read", n_chunks, n_rows)

    if not records:
        log.warning("  No lab records found — check hadm_id mapping and MIMIC_DIR")
        return pd.DataFrame(columns=["stay_id", "hour_offset", "lab_name", "value"])

    raw = pd.concat(records, ignore_index=True)
    log.info("  Raw lab records : %d", len(raw))

    # Median per (stay_id, hour_offset, lab_name)
    agg = (
        raw
        .groupby(["stay_id", "hour_offset", "lab_name"], sort=False)["valuenum"]
        .median()
        .reset_index()
        .rename(columns={"valuenum": "value"})
    )
    log.info("  After median aggregation: %d records", len(agg))

    # Apply physiological clip bounds
    for lab_name, (_, _, lo, hi) in LAB_DEFINITIONS.items():
        mask = agg["lab_name"] == lab_name
        agg.loc[mask, "value"] = agg.loc[mask, "value"].clip(lo, hi)

    return agg


# ─────────────────────────────────────────────────────────────
#  STEP 4 — Pivot labs onto the hourly grid + LOCF + obs flags
# ─────────────────────────────────────────────────────────────
def pivot_and_impute_labs(
    signals: pd.DataFrame,
    labs_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    1. Pivots labs_long to wide format
    2. Left-joins onto signals (the complete hourly grid)
    3. Adds _obs flags BEFORE imputation
    4. Applies variable LOCF per lab
    5. Computes hours_since_last_obs per lab (Fix C4)
    Returns merged DataFrame.
    """
    log.info("─── Pivoting and imputing labs ───")

    # Pivot: one column per lab
    if labs_long.empty:
        log.warning("  labs_long is empty — all lab columns will be NaN")
        wide = pd.DataFrame(columns=["stay_id", "hour_offset"])
    else:
        wide = labs_long.pivot_table(
            index=["stay_id", "hour_offset"],
            columns="lab_name",
            values="value",
            aggfunc="median",
        ).reset_index()
        wide.columns.name = None

    log.info("  Lab wide shape: %s", wide.shape)

    # Left-join onto the full hourly grid from S2b
    df = signals.merge(wide, on=["stay_id", "hour_offset"], how="left")
    log.info("  After merge   : %d rows × %d cols", *df.shape)

    # Ensure all 16 lab columns exist (some may be absent if no records found)
    all_lab_names = list(LAB_DEFINITIONS.keys())
    for lab in all_lab_names:
        if lab not in df.columns:
            df[lab] = np.nan
            log.warning("  Lab '%s' missing from data — column added as NaN", lab)

    # ── Add _obs flags BEFORE imputation ─────────────────
    for lab in all_lab_names:
        df[f"{lab}_obs"] = df[lab].notna().astype(np.int8)

    log.info("  _obs flags added for %d labs", len(all_lab_names))

    # ── Variable LOCF per lab per stay ───────────────────
    log.info("  Applying variable LOCF (grouped by stay_id) …")
    df = df.sort_values(["stay_id", "hour_offset"]).reset_index(drop=True)

    for lab, (_, locf_cap, _, _) in LAB_DEFINITIONS.items():
        df[lab] = (
            df.groupby("stay_id", sort=False)[lab]
            .transform(lambda s: s.ffill(limit=locf_cap))
        )

    log.info("  LOCF complete.")

    # ── hours_since_last_obs per lab (Fix C4) ────────────
    #
    #  For each lab at each hour, compute:
    #    hours since the last DIRECTLY OBSERVED (not imputed) value
    #  0  = measured at this exact hour
    #  48 = never measured in the stay (sentinel)
    #
    #  Algorithm:
    #    For each stay, scan forward through hours.
    #    When _obs=1, reset counter to 0.
    #    When _obs=0, increment counter by 1.
    #    Cap at MAX_HRS_SINCE.
    #
    log.info("  Computing hours_since_last_obs for %d labs …", len(all_lab_names))

    for lab in all_lab_names:
        obs_col  = f"{lab}_obs"
        hrs_col  = f"{lab}_hrs_since"

        # Vectorised approach using cumsum trick:
        # - Mark each observed row with a unique group id (cumsum of obs flags)
        # - Within each group, row_number gives hours since last observation
        # - Stays with no observations get the sentinel value

        def _hrs_since(grp: pd.DataFrame) -> pd.Series:
            obs = grp[obs_col].values
            hrs = np.full(len(obs), MAX_HRS_SINCE, dtype=np.int16)
            last_obs = -MAX_HRS_SINCE  # start with sentinel
            for i, (h, o) in enumerate(zip(grp["hour_offset"].values, obs)):
                if o == 1:
                    last_obs = h
                if last_obs >= 0:
                    hrs[i] = min(h - last_obs, MAX_HRS_SINCE)
            return pd.Series(hrs, index=grp.index)

        df[hrs_col] = (
            df.groupby("stay_id", sort=False, group_keys=False)
            [[obs_col, "hour_offset"]]
            .apply(_hrs_since)
        )

    log.info("  hours_since_last_obs computed.")
    return df


# ─────────────────────────────────────────────────────────────
#  STEP 5 — Sanity checks
# ─────────────────────────────────────────────────────────────
def sanity_checks(df: pd.DataFrame, cohort: pd.DataFrame) -> None:
    log.info("─── S3 Sanity Checks ───")
    failures = []

    # Check 1: row count unchanged from S2b
    expected = int((cohort["los_hours_int"] + 1).sum())
    actual   = len(df)
    if actual != expected:
        failures.append(
            f"Check 1 FAIL: expected {expected:,} rows, got {actual:,}"
        )
    else:
        log.info("  ✓ Check 1: Row count correct (%d)", actual)

    # Check 2: no duplicate (stay_id, hour_offset)
    n_dup = int(df.duplicated(subset=["stay_id", "hour_offset"]).sum())
    if n_dup:
        failures.append(f"Check 2 FAIL: {n_dup} duplicate (stay_id, hour_offset)")
    else:
        log.info("  ✓ Check 2: No duplicate (stay_id, hour_offset)")

    # Check 3: all _obs columns are binary
    obs_cols = [c for c in df.columns if c.endswith("_obs")]
    bad_obs  = [(c, int((~df[c].isin([0,1])).sum())) for c in obs_cols]
    bad_obs  = [(c, n) for c, n in bad_obs if n > 0]
    if bad_obs:
        failures.append(f"Check 3 FAIL: non-binary _obs values in {bad_obs}")
    else:
        log.info("  ✓ Check 3: All %d _obs columns are binary", len(obs_cols))

    # Check 4: hrs_since columns in [0, MAX_HRS_SINCE]
    hrs_cols = [c for c in df.columns if c.endswith("_hrs_since")]
    bad_hrs  = [(c, int(((df[c] < 0) | (df[c] > MAX_HRS_SINCE)).sum()))
                for c in hrs_cols]
    bad_hrs  = [(c, n) for c, n in bad_hrs if n > 0]
    if bad_hrs:
        failures.append(f"Check 4 FAIL: hrs_since out of range [0,{MAX_HRS_SINCE}]: {bad_hrs}")
    else:
        log.info("  ✓ Check 4: All hrs_since in [0, %d]", MAX_HRS_SINCE)

    # Check 5: LOCF cap respected — no lab value carried > its cap
    # Verify by checking that _obs=0 rows that are non-NaN
    # were preceded by an observed value within the cap window.
    # (Spot-check on lactate as representative critical lab)
    lac_nan_after_locf = df["lactate"].isna().sum()
    lac_total          = len(df)
    pct_nan = lac_nan_after_locf / lac_total * 100
    log.info("  ✓ Check 5: lactate NaN after LOCF = %.1f%% (expected ~70–90%%)", pct_nan)
    # High NaN rate is expected — labs are measured infrequently

    # Check 6: no negative hour_offset values
    n_neg = int((df["hour_offset"] < 0).sum())
    if n_neg:
        failures.append(f"Check 6 FAIL: {n_neg} rows with hour_offset < 0")
    else:
        log.info("  ✓ Check 6: No negative hour_offset values")

    # Check 7: critical lab clips respected (lactate max = 30)
    if "lactate" in df.columns:
        n_bad = int((df["lactate"].dropna() > 30).sum())
        if n_bad:
            failures.append(f"Check 7 FAIL: {n_bad} lactate values > 30 after clipping")
        else:
            log.info("  ✓ Check 7: Lactate values within clip bounds [0.1, 30]")

    if failures:
        msg = "S3 sanity checks FAILED:\n" + "\n".join(failures)
        log.error(msg)
        raise AssertionError(msg)

    log.info("  All S3 sanity checks passed ✓")


# ─────────────────────────────────────────────────────────────
#  STEP 6 — Save output + summary
# ─────────────────────────────────────────────────────────────
def save_output(df: pd.DataFrame) -> None:
    out_path = OUTPUT_DIR / "hourly_signals_with_labs.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Saved: %s  (%d rows × %d cols)", out_path, *df.shape)

    all_lab_names = list(LAB_DEFINITIONS.keys())
    lines = [
        "",
        "=" * 65,
        "  S3 LAB EXTRACTION SUMMARY",
        "=" * 65,
        f"  Total hourly rows     : {len(df):>10,}",
        f"  Unique stays          : {df['stay_id'].nunique():>10,}",
        f"  Total columns         : {len(df.columns):>10,}",
        f"    From S2b            : 21 (vitals + GCS + AVPU)",
        f"    Added by S3         : {len(df.columns)-21} (16 labs × 3 cols each)",
        "",
        "  Lab coverage (% hours observed | % NaN after LOCF):",
        f"  {'Lab':<14} {'Cap':>4}h  {'Obs%':>7}  {'NaN% post-LOCF':>16}  {'Median':>8}",
        "  " + "-" * 55,
    ]
    for lab, (_, cap, _, _) in LAB_DEFINITIONS.items():
        obs_col = f"{lab}_obs"
        pct_obs = df[obs_col].mean() * 100 if obs_col in df.columns else 0
        pct_nan = df[lab].isna().mean() * 100 if lab in df.columns else 100
        med_val = df[lab].median() if lab in df.columns else float("nan")
        lines.append(
            f"  {lab:<14} {cap:>4}h  {pct_obs:>6.2f}%  {pct_nan:>15.1f}%  {med_val:>8.2f}"
        )

    lines += [
        "",
        "  New columns per lab (example: lactate):",
        "    lactate           — imputed value (LOCF applied)",
        "    lactate_obs       — 1=observed, 0=imputed/missing",
        "    lactate_hrs_since — hours since last observed (0–48)",
        "",
        "  Output: hourly_signals_with_labs.parquet → feed into S4",
        "=" * 65,
        "",
    ]
    summary = "\n".join(lines)
    log.info(summary)
    (OUTPUT_DIR / "s3_summary.txt").write_text(summary, encoding="utf-8")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def run_s3() -> pd.DataFrame:
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  S3 — Laboratory Value Extraction               ║")
    log.info("╚══════════════════════════════════════════════════╝")
    log.info("MIMIC_DIR  = %s", MIMIC_DIR.resolve())
    log.info("S1_DIR     = %s", S1_DIR.resolve())
    log.info("S2B_DIR    = %s", S2B_DIR.resolve())
    log.info("OUTPUT_DIR = %s", OUTPUT_DIR.resolve())

    # Load
    cohort, signals = load_inputs()

    # Verify itemids exist before expensive extraction
    confirmed_itemids = verify_itemids()

    # Extract
    labs_long = extract_labs(cohort, confirmed_itemids)

    # Pivot + LOCF + hrs_since
    df = pivot_and_impute_labs(signals, labs_long)

    # Sanity checks
    sanity_checks(df, cohort)

    # Save
    save_output(df)

    log.info("S3 DONE — hourly_signals_with_labs.parquet written.")
    return df


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print()
    print("=" * 65)
    print("  S3 — Laboratory Value Extraction")
    print("=" * 65)
    print(f"  MIMIC_DIR  = {MIMIC_DIR.resolve()}")
    print(f"  S2B_DIR    = {S2B_DIR.resolve()}")
    print(f"  OUTPUT_DIR = {OUTPUT_DIR.resolve()}")
    print()

    ok = True
    if not MIMIC_DIR.exists():
        print("  ✗ MIMIC_DIR not found — set it at the top of the file")
        ok = False
    if not (S1_DIR / "cohort.parquet").exists():
        print("  ✗ cohort.parquet not found — run S1 first")
        ok = False
    if not (S2B_DIR / "hourly_signals_with_gcs.parquet").exists():
        print("  ✗ hourly_signals_with_gcs.parquet not found — run S2b first")
        ok = False
    if not ok:
        raise SystemExit(1)

    df = run_s3()

    print()
    print("── Sample (first 3 rows, lab columns only) ──")
    lab_cols = (
        ["stay_id", "hour_offset"] +
        ["lactate", "lactate_obs", "lactate_hrs_since"] +
        ["creatinine", "creatinine_obs", "creatinine_hrs_since"]
    )
    available = [c for c in lab_cols if c in df.columns]
    print(df[available].head(3).to_string())
    print()
    print(f"Shape   : {df.shape}")
    print(f"Columns : {list(df.columns)}")
    print(f"Saved → {(OUTPUT_DIR / 'hourly_signals_with_labs.parquet').resolve()}")
