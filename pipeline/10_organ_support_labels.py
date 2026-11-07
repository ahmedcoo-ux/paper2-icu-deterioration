# Source: SAL.ipynb, code cells [1] (outputs not included)

# %% ---- cell 1 ----
# ============================================================================
# SAL — Step 1: organ-support outcome feasibility audit (NO model training)
#   Endpoints: new vasopressor initiation, new invasive ventilation, composite
#   Incident  = an episode START within the window's 6-h horizon
#   Eligible  = endpoint not active at any time in the observation window
#               (primary); also reported: never received earlier in the stay (strict)
#   Output    = counts by split x NEWS-2 value  +  per-window label file for the
#               precision simulation (organ_support_window_labels.npz)
# ============================================================================
import os, json, time
import numpy as np
import pandas as pd

D = r"<PHD_ROOT>\PHD Research\DATAset"
CFG = {
    "MIMIC_ICU": os.path.join(D, r"mimic-iv-3.1\icu"),
    "WINDOWS":   os.path.join(D, r"s4\windows.parquet"),
    "HOURLY":    os.path.join(D, r"s3\hourly_signals_with_labs.parquet"),
    "ROWS_DEV":  r"<PHD_ROOT>\temporal_holdout\train_s4_rows.npy",
    "ROWS_HO":   r"<PHD_ROOT>\temporal_holdout\holdout_s4_rows.npy",
    "NEWS6_HO":  r"<PHD_ROOT>\ewp_results\news2_DEFINITIVE_holdout.npy",
    "NEWS7_HO":  r"<PHD_ROOT>\ewp_results\news2_7comp_concurrent_holdout.npy",
    "O2":        r"<PHD_ROOT>\ewp_results\mimic_suppl_o2_v1.parquet",
    "OUT_DIR":   r"<PHD_ROOT>\ewp_results\sal_feasibility",
    "VASO_ITEMS": {221906: "norepinephrine", 221289: "epinephrine", 221749: "phenylephrine",
                   222315: "vasopressin", 221662: "dopamine"},
    "VENT_INTERVAL_ITEMS": {225792: "invasive ventilation"},
    "VENT_POINT_ITEMS":    {224385: "intubation"},
    "EPISODE_GAP_H": 1.0,
    "O2_TOL": 4,
    "MIN_PER_CLASS": 100,
}
os.makedirs(CFG["OUT_DIR"], exist_ok=True)
t0 = time.time()
def log(m): print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)
def stop(m): raise SystemExit("\n*** STOPPED: " + m + "\n*** Send me everything printed above.")
def mimic_path(name):
    for ext in (".csv.gz", ".csv"):
        p = os.path.join(CFG["MIMIC_ICU"], name + ext)
        if os.path.exists(p):
            return p
    stop(f"{name}.csv(.gz) not found in {CFG['MIMIC_ICU']}")

# ----------------------------------------------------------------------------
# 1. Verify item IDs against d_items
# ----------------------------------------------------------------------------
d_items = pd.read_csv(mimic_path("d_items"), usecols=["itemid", "label", "linksto", "category"])
want = {**{k: ("inputevents", v) for k, v in CFG["VASO_ITEMS"].items()},
        **{k: ("procedureevents", v) for k, v in CFG["VENT_INTERVAL_ITEMS"].items()},
        **{k: ("procedureevents", v) for k, v in CFG["VENT_POINT_ITEMS"].items()}}
chk = d_items[d_items.itemid.isin(want)].set_index("itemid")
print("\nITEM ID VERIFICATION")
for iid, (table, intended) in want.items():
    if iid not in chk.index:
        stop(f"itemid {iid} ({intended}) not found in d_items")
    r = chk.loc[iid]
    ok = r.linksto == table
    print(f"  {iid}: '{r.label}' | table={r.linksto} | category={r.category} | intended={intended} "
          f"{'OK' if ok else '*** UNEXPECTED TABLE ***'}")
    if not ok:
        stop(f"itemid {iid} links to {r.linksto}, expected {table}")

# ----------------------------------------------------------------------------
# 2. Windows, splits, ICU admission times
# ----------------------------------------------------------------------------
win = pd.read_parquet(CFG["WINDOWS"], columns=["subject_id", "stay_id", "window_start",
                                               "window_end", "label_start", "label_end"])
rows_dev = np.load(CFG["ROWS_DEV"]).ravel(); rows_ho = np.load(CFG["ROWS_HO"]).ravel()
split = np.full(len(win), "", dtype=object); split[rows_dev] = "development"; split[rows_ho] = "held-out"
win["split"] = split
win["file_row"] = np.arange(len(win))
win = win[win.split != ""].reset_index(drop=True)
log(f"windows: development {np.sum(win.split=='development'):,} | held-out {np.sum(win.split=='held-out'):,}")

rel = pd.DataFrame({"label_start - window_end": win.label_start - win.window_end,
                    "label_end - label_start": win.label_end - win.label_start})
print("\nWINDOW TIMING (must be constant):")
for c in rel: print(f"  {c}: {sorted(rel[c].unique())[:5]}")
if rel.nunique().max() > 1:
    stop("window timing is not constant")
win["t_pred"] = win.window_end.astype(float) + 1.0
win["h_start"] = win.label_start.astype(float)
win["h_end"] = win.label_end.astype(float) + 1.0
horizon_h = float(win.h_end.iloc[0] - win.h_start.iloc[0])
lead_h = float(win.h_start.iloc[0] - win.t_pred.iloc[0])
print(f"  prediction time = window_end + 1 h; horizon = [label_start, label_end + 1) = {horizon_h:.0f} h, "
      f"starting {lead_h:.0f} h after prediction")

icu = pd.read_csv(mimic_path("icustays"), usecols=["stay_id", "intime", "outtime"], parse_dates=["intime", "outtime"])
icu = icu[icu.stay_id.isin(win.stay_id.unique())].set_index("stay_id")
los_h = (icu.outtime - icu.intime).dt.total_seconds() / 3600
bad = np.mean(win.h_end.to_numpy() > los_h.reindex(win.stay_id).to_numpy() + 1e-6)
log(f"windows whose horizon ends after ICU discharge (final partial hour): {100*bad:.3f}%")

# ----------------------------------------------------------------------------
# 3. Seven-component NEWS-2 for every window
# ----------------------------------------------------------------------------
hcols = ["stay_id", "hour_offset", "rr", "spo2", "sbp", "hr", "temp", "gcs_total"]
hs = pd.read_parquet(CFG["HOURLY"], columns=hcols).rename(columns={"hour_offset": "hour"})
hs["hour"] = np.floor(hs["hour"].astype(float)).astype(np.int64)
hs = hs.groupby(["stay_id", "hour"], as_index=False).last()
if hs.temp.median() > 50: hs["temp"] = (hs.temp - 32) * 5 / 9
rr, sp, sb, hr, te, gc = (hs[c].to_numpy(float) for c in ["rr", "spo2", "sbp", "hr", "temp", "gcs_total"])
n6h = (np.select([rr <= 8, rr <= 11, rr <= 20, rr <= 24, rr > 24], [3, 1, 0, 2, 3], 0)
       + np.select([sp <= 91, sp <= 93, sp <= 95, sp > 95], [3, 2, 1, 0], 0)
       + np.select([sb <= 90, sb <= 100, sb <= 110, sb <= 219, sb > 219], [3, 2, 1, 0, 3], 0)
       + np.select([hr <= 40, hr <= 50, hr <= 90, hr <= 110, hr <= 130, hr > 130], [3, 1, 0, 1, 2, 3], 0)
       + np.where(gc < 15, 3, 0)
       + np.select([te <= 35.0, te <= 36.0, te <= 38.0, te <= 39.0, te > 39.0], [3, 1, 0, 1, 2], 0))
tot6 = pd.Series(n6h, index=pd.MultiIndex.from_arrays([hs.stay_id.astype(np.int64), hs.hour]))
del hs
news6 = tot6.reindex(pd.MultiIndex.from_arrays([win.stay_id.astype(np.int64),
                                                win.window_end.astype(np.int64)])).to_numpy()

o2 = pd.read_parquet(CFG["O2"], columns=["stay_id", "hour_offset", "suppl_o2"])
o2 = o2[o2.suppl_o2 == 1][["stay_id", "hour_offset"]].drop_duplicates().astype(np.int64).sort_values("hour_offset")
q = pd.DataFrame({"i": np.arange(len(win)), "stay_id": win.stay_id.astype(np.int64),
                  "hour_offset": win.window_end.astype(np.int64)}).sort_values("hour_offset")
m = pd.merge_asof(q, o2.assign(o2=1), on="hour_offset", by="stay_id", direction="backward", tolerance=CFG["O2_TOL"])
on_o2 = np.zeros(len(win)); on_o2[m.i.to_numpy()] = m.o2.fillna(0).to_numpy()
news7 = news6 + 2 * on_o2

ho = (win.split == "held-out").to_numpy()
n7_saved = np.load(CFG["NEWS7_HO"]).ravel()
# stored held-out arrays are in holdout_s4_rows order; map them to windows-file order
ho_pos = np.searchsorted(np.sort(rows_ho), win.file_row.to_numpy()[ho])
n7_saved_fileorder = n7_saved[np.argsort(rows_ho)][ho_pos]
match = np.mean(news7[ho] == n7_saved_fileorder)
log(f"rebuilt seven-component score vs stored (held-out): {100*match:.3f}% identical")
if match < 0.995: stop("seven-component rebuild does not reproduce the stored held-out score")
news7[ho] = n7_saved_fileorder
win["news7"] = news7

# ----------------------------------------------------------------------------
# 4. Treatment episodes (hours from ICU admission), merged across short gaps
# ----------------------------------------------------------------------------
def to_hours(df, col):
    return (df[col] - icu.intime.reindex(df.stay_id).to_numpy()).dt.total_seconds().to_numpy() / 3600

def merge_episodes(df):
    out = []
    for sid, g in df.sort_values(["stay_id", "s"]).groupby("stay_id", sort=False):
        s, e = g.s.to_numpy(), g.e.to_numpy()
        cs, ce = s[0], e[0]
        for a, b in zip(s[1:], e[1:]):
            if a <= ce + CFG["EPISODE_GAP_H"]:
                ce = max(ce, b)
            else:
                out.append((sid, cs, ce)); cs, ce = a, b
        out.append((sid, cs, ce))
    return pd.DataFrame(out, columns=["stay_id", "s", "e"])

stays = set(win.stay_id.unique())
vparts = []
for ch in pd.read_csv(mimic_path("inputevents"), chunksize=2_000_000,
                      usecols=["stay_id", "itemid", "starttime", "endtime", "rate", "amount", "statusdescription"]):
    ch = ch[ch.itemid.isin(CFG["VASO_ITEMS"]) & ch.stay_id.isin(stays)]
    ch = ch[(ch.statusdescription.fillna("") != "Rewritten") & ((ch.rate.fillna(0) > 0) | (ch.amount.fillna(0) > 0))]
    vparts.append(ch[["stay_id", "itemid", "starttime", "endtime"]])
vaso = pd.concat(vparts)
vaso["starttime"] = pd.to_datetime(vaso.starttime); vaso["endtime"] = pd.to_datetime(vaso.endtime)
vaso["s"] = to_hours(vaso, "starttime"); vaso["e"] = to_hours(vaso, "endtime")
vaso.loc[vaso.e < vaso.s, "e"] = vaso.s
log(f"vasopressor rows kept: {len(vaso):,}  by drug: " +
    ", ".join(f"{CFG['VASO_ITEMS'][k]} {v:,}" for k, v in vaso.itemid.value_counts().items()))
vaso_ep = merge_episodes(vaso[["stay_id", "s", "e"]])

pe = pd.read_csv(mimic_path("procedureevents"), usecols=["stay_id", "itemid", "starttime", "endtime"],
                 parse_dates=["starttime", "endtime"])
pe = pe[pe.stay_id.isin(stays)]
vint = pe[pe.itemid.isin(CFG["VENT_INTERVAL_ITEMS"])].copy()
vpt = pe[pe.itemid.isin(CFG["VENT_POINT_ITEMS"])].copy()
vint["s"] = to_hours(vint, "starttime"); vint["e"] = to_hours(vint, "endtime")
vint["e"] = np.where(np.isnan(vint.e) | (vint.e < vint.s), vint.s, vint.e)
vpt["s"] = to_hours(vpt, "starttime"); vpt["e"] = vpt.s
log(f"ventilation: {len(vint):,} interval records, {len(vpt):,} intubation records")

vi_first = vint.sort_values("s").groupby("stay_id").s.first()
in_first = vpt.sort_values("s").groupby("stay_id").s.first()
both = vi_first.index.intersection(in_first.index)
agree = np.mean(np.abs(vi_first[both] - in_first[both]) <= 2)
print(f"\nVENTILATION TIMING CHECK: stays with both records: {len(both):,}; first ventilation start within ±2 h "
      f"of first intubation in {100*agree:.1f}%; ventilation without any intubation record: "
      f"{len(vi_first.index.difference(in_first.index)):,} stays")
vent_ep = merge_episodes(pd.concat([vint[["stay_id", "s", "e"]], vpt[["stay_id", "s", "e"]]]))

# ----------------------------------------------------------------------------
# 5. Per-window status for each endpoint
# ----------------------------------------------------------------------------
def window_status(ep):
    n = len(win)
    active = np.zeros(n, bool); prior = np.zeros(n, bool); incident = np.zeros(n, bool)
    ep_id = np.full(n, -1, np.int64)
    eps = {sid: (g.s.to_numpy(), g.e.to_numpy(), g.index.to_numpy()) for sid, g in ep.groupby("stay_id")}
    ws_all = win.window_start.to_numpy().astype(float)
    tp_all, hs_all, he_all = win.t_pred.to_numpy(), win.h_start.to_numpy(), win.h_end.to_numpy()
    for sid, idx in win.groupby("stay_id").indices.items():
        if sid not in eps: continue
        s, e, eid = eps[sid]
        ws, tp, hs_, he = ws_all[idx], tp_all[idx], hs_all[idx], he_all[idx]
        k = np.searchsorted(s, tp, "left") - 1
        has = k >= 0
        active[idx] = has & (e[np.clip(k, 0, None)] > ws)
        prior[idx] = has
        a = np.searchsorted(s, hs_, "left"); b = np.searchsorted(s, he, "left")
        incident[idx] = b > a
        ep_id[idx] = np.where(b > a, eid[np.clip(a, 0, len(eid) - 1)], -1)
    return active, prior, incident, ep_id

st = {}
for name, ep in [("vasopressor", vaso_ep), ("ventilation", vent_ep)]:
    a, p, inc, eid = window_status(ep)
    st[name] = {"active": a, "prior": p, "incident": inc, "ep": eid}
    log(f"{name}: episodes {len(ep):,}")
st["composite"] = {
    "active": st["vasopressor"]["active"] | st["ventilation"]["active"],
    "prior": st["vasopressor"]["prior"] | st["ventilation"]["prior"],
    "incident": st["vasopressor"]["incident"] | st["ventilation"]["incident"],
    "ep": None}

# ----------------------------------------------------------------------------
# 6. Count tables
# ----------------------------------------------------------------------------
rows = []
subj = win.subject_id.to_numpy()
for endpoint, S in st.items():
    for elig_name, elig in [("primary (not active in observation window)", ~S["active"]),
                            ("strict (never received before prediction)", ~S["prior"])]:
        for sp_name in ["development", "held-out"]:
            in_split = (win.split == sp_name).to_numpy()
            for stratum in ["0", "1", "2", "3", "4", "0-4", "5-6"]:
                if stratum == "0-4":   sm = win.news7.between(0, 4).to_numpy()
                elif stratum == "5-6": sm = win.news7.between(5, 6).to_numpy()
                else:                  sm = (win.news7 == int(stratum)).to_numpy()
                base = in_split & sm
                e_ = base & elig
                pos = e_ & S["incident"]
                if S["ep"] is not None:
                    n_ep = len(np.unique(S["ep"][pos]))
                else:
                    n_ep = len(np.unique(st["vasopressor"]["ep"][pos & st["vasopressor"]["incident"]])) + \
                           len(np.unique(st["ventilation"]["ep"][pos & st["ventilation"]["incident"]]))
                rows.append({"endpoint": endpoint, "eligibility": elig_name, "split": sp_name,
                             "news7": stratum, "windows_total": int(base.sum()),
                             "windows_eligible": int(e_.sum()),
                             "pct_eligible": 100 * e_.sum() / max(base.sum(), 1),
                             "incident_windows": int(pos.sum()), "negative_windows": int((e_ & ~S["incident"]).sum()),
                             "prevalence_pct": 100 * pos.sum() / max(e_.sum(), 1),
                             "patients_eligible": int(len(np.unique(subj[e_]))),
                             "patients_with_event": int(len(np.unique(subj[pos]))),
                             "distinct_incident_episodes": int(n_ep)})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(CFG["OUT_DIR"], "organ_support_feasibility.csv"), index=False)
json.dump({"item_check": {int(k): str(chk.loc[k, "label"]) for k in want},
           "ventilation_timing_agreement_pct": float(100 * agree),
           "episode_gap_h": CFG["EPISODE_GAP_H"], "news7_rebuild_match_heldout_pct": float(100 * match)},
          open(os.path.join(CFG["OUT_DIR"], "organ_support_feasibility_meta.json"), "w"), indent=1)

# development-only summary printed (held-out counts are saved to the CSV but not displayed)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 300)
show = ["news7", "windows_total", "windows_eligible", "pct_eligible", "incident_windows",
        "prevalence_pct", "patients_eligible", "patients_with_event", "distinct_incident_episodes"]
for endpoint in st:
    for elig_name in res.eligibility.unique():
        print("\n" + "=" * 120 + f"\n{endpoint.upper()} — {elig_name} — DEVELOPMENT ONLY\n" + "=" * 120)
        sub = res[(res.endpoint == endpoint) & (res.eligibility == elig_name) & (res.split == "development")][show]
        print(sub.round(2).to_string(index=False))
log("saved organ_support_feasibility.csv")

# ----------------------------------------------------------------------------
# 7. Save per-window arrays for the precision simulation (windows-file order + file row index)
# ----------------------------------------------------------------------------
np.savez_compressed(os.path.join(CFG["OUT_DIR"], "organ_support_window_labels.npz"),
                    file_row=win.file_row.to_numpy(), split=win.split.to_numpy().astype(str),
                    subject_id=win.subject_id.to_numpy(), news7=win.news7.to_numpy(),
                    **{f"{k}_{f}": st[k][f] for k in st for f in ("active", "incident")})
log("saved organ_support_window_labels.npz")
