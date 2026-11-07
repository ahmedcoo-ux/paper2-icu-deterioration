# Source: s5 part b.ipynb, code cells [0, 1, 2] (outputs not included)

# %% ---- cell 0 ----
print ('done')

# %% ---- cell 1 ----
"""
============================================================
S5 Part B — Feature Engineering (Second Half of Stays)
============================================================
REQUIRES: s5_presplit.py and s5_partA.py completed first.
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
S4_DIR      = Path(r"<PHD_ROOT>\PHD Research\DATAset\s4")
SIGNALS_DIR = Path(r"<PHD_ROOT>\PHD Research\DATAset\s5_signals")   # per-stay files
OUTPUT_DIR  = Path(r"<PHD_ROOT>\PHD Research\DATAset\s5")



VITAL_SIGNALS = ["hr","sbp","dbp","map","rr","spo2","temp","urine_output"]
LAB_SIGNALS   = [
    "lactate","ph","pco2","po2","creatinine","bun","potassium","sodium",
    "glucose","wbc","hemoglobin","hematocrit","platelets","bilirubin",
    "bicarbonate","albumin",
]
VITAL_THRESHOLDS = [
    ("spo2", lambda x: x < 94,            "spo2_pct_below94"),
    ("rr",   lambda x: (x>25)|(x<8),      "rr_pct_abnormal"),
    ("sbp",  lambda x: x < 100,           "sbp_pct_below100"),
    ("hr",   lambda x: (x>130)|(x<40),    "hr_pct_abnormal"),
    ("temp", lambda x: (x>39.1)|(x<35.0), "temp_pct_abnormal"),
]
LAB_THRESHOLDS = {
    "lactate"   : lambda x: (x >= 2.0).any(),
    "creatinine": lambda x: (x >= 1.5).any(),
    "potassium" : lambda x: ((x<3.0)|(x>5.5)).any(),
    "wbc"       : lambda x: ((x>12)|(x<4)).any(),
    "glucose"   : lambda x: ((x<70)|(x>180)).any(),
    "bilirubin" : lambda x: (x > 2.0).any(),
}
WINDOW_SIZE = 6
CHUNK_SIZE  = 10_000

# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR/"s5_partB.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("S5B")


def _slope(values, obs_mask):
    hours = np.arange(len(values))
    mask  = obs_mask == 1
    if mask.sum() < 2:
        return 0.0
    x, y  = hours[mask].astype(float), values[mask]
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return 0.0
    s, *_ = scipy_stats.linregress(x[valid], y[valid])
    return float(s) if np.isfinite(s) else 0.0


def window_feats(row, stay_df, age_lookup):
    sid   = int(row["stay_id"])
    ws    = int(row["window_start"])
    we    = int(row["window_end"])
    feats = {
        "window_id" : int(row["window_id"]),
        "stay_id"   : sid,
        "subject_id": int(row["subject_id"]),
        "label"     : int(row["label"]),
    }
    if stay_df is None or stay_df.empty:
        return feats

    wd = stay_df[
        (stay_df["hour_offset"] >= ws) &
        (stay_df["hour_offset"] <= we)
    ].set_index("hour_offset")

    for sig in VITAL_SIGNALS:
        v  = wd[sig].values              if sig           in wd.columns else np.full(WINDOW_SIZE, np.nan)
        ob = wd[f"{sig}_obs"].values     if f"{sig}_obs"  in wd.columns else np.zeros(WINDOW_SIZE)
        nn = v[~np.isnan(v)]; n = int((ob==1).sum()); nt = len(v)
        feats[f"{sig}_mean"]      = float(np.nanmean(v))  if n>0 else 0.0
        feats[f"{sig}_min"]       = float(np.nanmin(v))   if n>0 else 0.0
        feats[f"{sig}_max"]       = float(np.nanmax(v))   if n>0 else 0.0
        feats[f"{sig}_std"]       = float(np.nanstd(v))   if n>1 else 0.0
        feats[f"{sig}_last"]      = float(nn[-1])          if n>0 else 0.0
        feats[f"{sig}_slope"]     = _slope(v, ob)
        feats[f"{sig}_delta"]     = float(nn[-1]-nn[0])    if n>=2 else 0.0
        feats[f"{sig}_miss_rate"] = float(1.0-(n/nt))
        feats[f"{sig}_cross_pct"] = 0.0

    for sig, fn, fname in VITAL_THRESHOLDS:
        v  = wd[sig].values if sig in wd.columns else np.array([])
        nn = v[~np.isnan(v)]
        feats[fname] = float(fn(nn).mean()) if len(nn)>0 else 0.0

    for lab in LAB_SIGNALS:
        v   = wd[lab].values                if lab               in wd.columns else np.full(WINDOW_SIZE, np.nan)
        ob  = wd[f"{lab}_obs"].values       if f"{lab}_obs"      in wd.columns else np.zeros(WINDOW_SIZE)
        hrs = wd[f"{lab}_hrs_since"].values if f"{lab}_hrs_since" in wd.columns else np.full(WINDOW_SIZE, 48)
        nn  = v[~np.isnan(v)]; n = int((ob==1).sum()); nt = len(v)
        feats[f"{lab}_mean"]      = float(np.nanmean(v))  if n>0 else 0.0
        feats[f"{lab}_last"]      = float(nn[-1])          if n>0 else 0.0
        feats[f"{lab}_slope"]     = _slope(v, ob)
        feats[f"{lab}_delta"]     = float(nn[-1]-nn[0])    if n>=2 else 0.0
        feats[f"{lab}_miss_rate"] = float(1.0-(n/nt))
        feats[f"{lab}_hrs_since"] = float(hrs[-1])         if len(hrs)>0 else 48.0

    for lab, fn in LAB_THRESHOLDS.items():
        v  = wd[lab].values if lab in wd.columns else np.array([])
        nn = v[~np.isnan(v)]
        feats[f"{lab}_flag"] = int(fn(nn)) if len(nn)>0 else 0

    if "gcs_total" in wd.columns:
        g  = wd["gcs_total"].values; nn = g[~np.isnan(g)]
        feats["gcs_mean"] = float(np.nanmean(g)) if len(nn)>0 else 15.0
        feats["gcs_last"] = float(nn[-1])         if len(nn)>0 else 15.0
    else:
        feats["gcs_mean"] = feats["gcs_last"] = 15.0
    feats["gcs_obs_rate"] = float(wd["gcs_obs"].mean())   if "gcs_obs"    in wd.columns else 0.0
    feats["avpu_sum"]     = float(wd["avpu_score"].sum()) if "avpu_score" in wd.columns else 0.0
    feats["age"]          = float(age_lookup.get(sid, 65.0))
    return feats


def run_part_b():
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  S5 Part B — Feature Engineering (Second Half)  ║")
    log.info("╚══════════════════════════════════════════════════╝")

    cohort = pd.read_parquet(S1_DIR/"cohort.parquet", columns=["stay_id","subject_id","age"])
    cohort["stay_id"] = cohort["stay_id"].astype(int)
    age_lookup = dict(zip(cohort["stay_id"], cohort["age"]))

    windows = pd.read_parquet(S4_DIR/"windows.parquet")
    windows["stay_id"] = windows["stay_id"].astype(int)

    stays_A = set(
        pd.read_parquet(OUTPUT_DIR/"partA_stay_ids.parquet")["stay_id"].astype(int).tolist()
    )
    stays_B = set(windows["stay_id"].unique()) - stays_A

    windows_B = windows[windows["stay_id"].isin(stays_B)].reset_index(drop=True)
    log.info("Part B: %d stays, %d windows", len(stays_B), len(windows_B))

    out_path    = OUTPUT_DIR/"window_features_partB.parquet"
    chunk_buf   = []
    first_write = True
    n_done      = 0
    stays_done  = 0
    n_stays     = len(stays_B)
    n_total     = len(windows_B)

    for stay_id, stay_windows in windows_B.groupby("stay_id", sort=False):
        stays_done += 1

        sig_file = SIGNALS_DIR / f"{int(stay_id)}.parquet"
        stay_df  = pd.read_parquet(sig_file) if sig_file.exists() else None
        if stay_df is not None:
            stay_df["hour_offset"] = stay_df["hour_offset"].astype(int)

        for _, row in stay_windows.iterrows():
            chunk_buf.append(window_feats(row, stay_df, age_lookup))
            n_done += 1

        if len(chunk_buf) >= CHUNK_SIZE:
            df_chunk = pd.DataFrame(chunk_buf)
            if first_write:
                df_chunk.to_parquet(out_path, index=False)
                first_write = False
            else:
                pd.concat([pd.read_parquet(out_path), df_chunk],
                          ignore_index=True).to_parquet(out_path, index=False)
            chunk_buf = []
            log.info("  Stays %d/%d | Windows %d/%d | flushed",
                     stays_done, n_stays, n_done, n_total)

        elif stays_done % 2000 == 0:
            log.info("  Stays %d/%d | Windows %d/%d",
                     stays_done, n_stays, n_done, n_total)

        del stay_df

    if chunk_buf:
        df_chunk = pd.DataFrame(chunk_buf)
        if first_write:
            df_chunk.to_parquet(out_path, index=False)
        else:
            pd.concat([pd.read_parquet(out_path), df_chunk],
                      ignore_index=True).to_parquet(out_path, index=False)
        log.info("  Final flush. Stays %d/%d | Windows %d/%d",
                 stays_done, n_stays, n_done, n_total)

    log.info("Part B DONE — %d windows → %s", n_done, out_path)
    log.info("Now run s5_merge.py")


if __name__ == "__main__":
    ok = True
    if not any(SIGNALS_DIR.glob("*.parquet")):
        print("  ✗ No per-stay signal files — run s5_presplit.py first")
        ok = False
    if not (OUTPUT_DIR/"partA_stay_ids.parquet").exists():
        print("  ✗ partA_stay_ids.parquet missing — run s5_partA.py first")
        ok = False
    if not ok:
        raise SystemExit(1)
    run_part_b()
    print("Part B complete — now run s5_merge.py")

# %% ---- cell 2 ----

