# Shared data and splits (used by every paper)
PHD            = r"<PHD_ROOT>"
WINDOWS        = r"<PHD_ROOT>\PHD Research\DATAset\s4\windows.parquet"
HOURLY         = r"<PHD_ROOT>\PHD Research\DATAset\s3\hourly_signals_with_labs.parquet"
CE_SUBSET      = r"<PHD_ROOT>\PHD Research\DATAset\s3\chartevents_subset.parquet"
ICUSTAYS       = r"<PHD_ROOT>\PHD Research\DATAset\mimic-iv-3.1\icu\icustays.csv"
EICU           = r"<PHD_ROOT>\PHD Research\DATAset\eicu-collaborative-research-database-2.0"
LABELS_NPZ     = r"<PHD_ROOT>\SAL\01_feasibility\organ_support_window_labels.npz"
TRAIN_ROWS     = r"<PHD_ROOT>\temporal_holdout\train_s4_rows.npy"
HOLDOUT_ROWS   = r"<PHD_ROOT>\temporal_holdout\holdout_s4_rows.npy"
FEATURE_COLS   = r"<PHD_ROOT>\shap\feature_cols.csv"
# Feature matrices
X_FULL_ORIGINAL    = r"<PHD_ROOT>\s7_rebuilt\X_full.npy"              # original (stay-length leak in 40 features)
X_FULL_V3          = r"<PHD_ROOT>\s7_rebuilt\X_full_v3.npy"           # leak-free, original 4 h consciousness
X_FULL_V3_REPAIRED = r"<PHD_ROOT>\s7_rebuilt\X_full_v3_repaired.npy"  # leak-free + 48 h consciousness (Paper 2 primary)
# Corrected labels / gate / strata (Script 1 v6.2)
CORRECTED_INPUTS = r"<PHD_ROOT>\ewp_results_v3"
Y_REPAIRED       = r"<PHD_ROOT>\ewp_results_v3\y_repaired.npy"
KEEP_REPAIRED    = r"<PHD_ROOT>\ewp_results_v3\keep_repaired.npy"
NEWS2_6_ORIG     = r"<PHD_ROOT>\ewp_results_v3\news2_6_original_we.npy"
NEWS2_6_REP      = r"<PHD_ROOT>\ewp_results_v3\news2_6_repaired_we.npy"
NEWS7_ORIG       = r"<PHD_ROOT>\ewp_results_v3\news7_original.npy"
NEWS7_REP        = r"<PHD_ROOT>\ewp_results_v3\news7_repaired.npy"
NEWS7_MISMATCH   = r"<PHD_ROOT>\ewp_results_v3\news7_mismatch_mask.npy"
GCS_OBS_AVAIL48  = r"<PHD_ROOT>\ewp_results_v3\gcs_obs_rate_avail48.npy"
INPUTS_SUMMARY   = r"<PHD_ROOT>\ewp_results_v3\corrected_inputs_summary.json"
