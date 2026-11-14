"""
Step 1 - Data Pipeline & Integrity Layer
Loads raw CSVs, runs data integrity checks (null audit, date continuity,
negative-sales check, orphan-record check, IQR outlier detection),
documents findings, and writes cleaned tables to output/data/ for
downstream EDA / modeling steps.
"""
import json
import os

import numpy as np
import pandas as pd

RAW_DIR = "."
OUT_DATA = "output/data"
OUT_REPORTS = "output/reports"
os.makedirs(OUT_DATA, exist_ok=True)
os.makedirs(OUT_REPORTS, exist_ok=True)

quality_report = {"checks": []}


def log_check(name, detail):
    print(f"[CHECK] {name}: {detail}")
    quality_report["checks"].append({"check": name, "detail": detail})


# ---------------------------------------------------------------------------
# Load raw tables
# ---------------------------------------------------------------------------
print("Loading raw CSVs...")

train = pd.read_csv(
    os.path.join(RAW_DIR, "train.csv"),
    parse_dates=["date"],
    dtype={"store_nbr": "int16", "family": "category", "onpromotion": "int32"},
)
stores = pd.read_csv(os.path.join(RAW_DIR, "stores.csv"))
oil = pd.read_csv(os.path.join(RAW_DIR, "oil.csv"), parse_dates=["date"])
holidays = pd.read_csv(os.path.join(RAW_DIR, "holidays_events.csv"), parse_dates=["date"])

print(f"train: {train.shape}, stores: {stores.shape}, oil: {oil.shape}, holidays: {holidays.shape}")

# ---------------------------------------------------------------------------
# Check 1 - Null audit per column, per table
# ---------------------------------------------------------------------------
null_summary = {}
for name, df in [("train", train), ("stores", stores), ("oil", oil), ("holidays", holidays)]:
    nulls = df.isnull().sum()
    pct = (nulls / len(df) * 100).round(3)
    tbl = {col: {"nulls": int(nulls[col]), "pct": float(pct[col])} for col in df.columns if nulls[col] > 0}
    null_summary[name] = tbl
    log_check(f"null_audit_{name}", tbl if tbl else "no nulls")

# oil.csv has missing dcoilwtico entries (weekends/holidays not traded) -> forward-fill then back-fill
oil_nulls_before = int(oil["dcoilwtico"].isnull().sum())
oil = oil.sort_values("date")
oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
log_check(
    "oil_price_imputation",
    f"{oil_nulls_before} missing oil prices ({oil_nulls_before/len(oil)*100:.1f}%) forward-filled "
    "(oil prices don't change on non-trading days) then back-filled for the series start.",
)

# ---------------------------------------------------------------------------
# Check 2 - Date continuity in the train time index
# ---------------------------------------------------------------------------
full_range = pd.date_range(train["date"].min(), train["date"].max(), freq="D")
present_dates = pd.DatetimeIndex(train["date"].unique())
missing_dates = full_range.difference(present_dates)
log_check(
    "date_continuity",
    f"Full range {full_range.min().date()} -> {full_range.max().date()} "
    f"({len(full_range)} days). {len(missing_dates)} calendar date(s) entirely absent from train.csv: "
    f"{[str(d.date()) for d in missing_dates]}. "
    "(Known Favorita quirk: Dec 25 has zero rows every year because all stores are closed.)",
)

# ---------------------------------------------------------------------------
# Check 3 - Negative sales sanity check
# ---------------------------------------------------------------------------
neg_mask = train["sales"] < 0
n_neg = int(neg_mask.sum())
log_check(
    "negative_sales",
    f"{n_neg} rows ({n_neg/len(train)*100:.4f}%) had negative sales (returns / data-entry errors). Removed.",
)
train = train.loc[~neg_mask].copy()

# ---------------------------------------------------------------------------
# Check 4 - Orphan store records (train store_nbr not present in stores.csv)
# ---------------------------------------------------------------------------
train_stores = set(train["store_nbr"].unique())
meta_stores = set(stores["store_nbr"].unique())
orphans = train_stores - meta_stores
log_check(
    "orphan_store_join",
    f"{len(orphans)} store_nbr value(s) in train.csv have no matching row in stores.csv: {sorted(orphans)}",
)

# ---------------------------------------------------------------------------
# Check 5 - Outlier detection via IQR (per product family, since sales is
# right-skewed and scale differs hugely across families -> Z-score is
# inappropriate here; IQR is distribution-agnostic).
# ---------------------------------------------------------------------------
outlier_flags = np.zeros(len(train), dtype=bool)
outlier_counts_by_family = {}

for fam, idx in train.groupby("family", observed=True).groups.items():
    s = train.loc[idx, "sales"]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 3 * iqr  # wide (3x) fence: flag extreme spikes only, don't strip legitimate promo/holiday peaks
    fam_outliers = s > upper
    if fam_outliers.any():
        outlier_counts_by_family[str(fam)] = int(fam_outliers.sum())
    outlier_flags[train.index.get_indexer(idx[fam_outliers])] = True

train["is_outlier"] = outlier_flags
n_outliers = int(outlier_flags.sum())
log_check(
    "iqr_outlier_detection",
    f"{n_outliers} rows ({n_outliers/len(train)*100:.3f}%) flagged as extreme outliers "
    f"(> Q3 + 3*IQR, computed per product family). Flagged, NOT removed (real promo/holiday spikes are "
    f"legitimate signal, not noise) - top families by flag count: "
    f"{dict(sorted(outlier_counts_by_family.items(), key=lambda x: -x[1])[:10])}",
)

# ---------------------------------------------------------------------------
# Persist cleaned tables + quality report
# ---------------------------------------------------------------------------
train.to_parquet(os.path.join(OUT_DATA, "fact_sales.parquet"), index=False)
stores.to_parquet(os.path.join(OUT_DATA, "dim_stores.parquet"), index=False)
oil.to_parquet(os.path.join(OUT_DATA, "dim_oil.parquet"), index=False)
holidays.to_parquet(os.path.join(OUT_DATA, "dim_holidays.parquet"), index=False)

quality_report["summary"] = {
    "rows_in": 3000888 + 1,
    "rows_out": len(train),
    "rows_removed": n_neg,
    "outliers_flagged": n_outliers,
    "date_range": [str(full_range.min().date()), str(full_range.max().date())],
    "n_stores": len(meta_stores),
    "n_families": train["family"].nunique(),
}
with open(os.path.join(OUT_REPORTS, "data_quality_report.json"), "w") as f:
    json.dump(quality_report, f, indent=2, default=str)

# Flat CSV version for the Power BI "Data Quality Report" page
rows = []
for name, tbl in null_summary.items():
    for col, v in tbl.items():
        rows.append({"table": name, "check_type": "null_audit", "column": col,
                      "count": v["nulls"], "pct": v["pct"]})
rows.append({"table": "train", "check_type": "negative_sales_removed", "column": "sales",
             "count": n_neg, "pct": round(n_neg/len(train)*100, 4)})
rows.append({"table": "train", "check_type": "outliers_flagged", "column": "sales",
             "count": n_outliers, "pct": round(n_outliers/len(train)*100, 4)})
rows.append({"table": "train", "check_type": "missing_calendar_dates", "column": "date",
             "count": len(missing_dates), "pct": round(len(missing_dates)/len(full_range)*100, 4)})
rows.append({"table": "train", "check_type": "orphan_store_records", "column": "store_nbr",
             "count": len(orphans), "pct": 0.0})
pd.DataFrame(rows).to_csv(os.path.join(OUT_REPORTS, "data_quality_report.csv"), index=False)

print("\nData integrity layer complete. Cleaned tables written to output/data/, report to output/reports/.")
