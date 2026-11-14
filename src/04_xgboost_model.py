"""
Step 4 - Model C: XGBoost with lag features (primary/production model).
Trained at store-family granularity with calendar, lag, rolling, oil and
promotion features. This is the model used downstream for scenario
analysis and stockout risk scoring.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from metrics_util import all_metrics, mase, zero_share

_t0 = time.time()


def log(msg):
    print(f"[{time.time()-_t0:7.1f}s] {msg}", flush=True)

OUT_DATA = "output/data"
OUT_MODELS = "output/models"
os.makedirs(OUT_MODELS, exist_ok=True)

HORIZON = 30  # days held out for evaluation / used as forecast horizon

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
log("Loading parquet tables...")
train = pd.read_parquet(os.path.join(OUT_DATA, "fact_sales.parquet"))
stores = pd.read_parquet(os.path.join(OUT_DATA, "dim_stores.parquet"))
oil = pd.read_parquet(os.path.join(OUT_DATA, "dim_oil.parquet"))
holidays = pd.read_parquet(os.path.join(OUT_DATA, "dim_holidays.parquet"))
log(f"Loaded. train={len(train):,} rows")

df = train.merge(stores, on="store_nbr", how="left")
df = df.merge(oil, on="date", how="left")
df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()

nat_holiday_dates = set(holidays.loc[holidays["locale"] == "National", "date"].dt.date)
df["is_holiday"] = df["date"].dt.date.isin(nat_holiday_dates).astype("int8")
log("Merged dimension tables.")

# ---------------------------------------------------------------------------
# Feature engineering (vectorized groupby-native ops, not per-group lambdas
# - a Python-level lambda transform over ~1,800 store-family groups is far
# slower than letting pandas' Cython groupby/rolling machinery do the work)
# ---------------------------------------------------------------------------
df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
log("Sorted.")

gkeys = ["store_nbr", "family"]
grp = df.groupby(gkeys, observed=True)["sales"]

df["lag_7"] = grp.shift(7)
log("lag_7 done.")
df["lag_14"] = grp.shift(14)
log("lag_14 done.")

df["_shifted_sales"] = grp.shift(1)
roll = df.groupby(gkeys, observed=True)["_shifted_sales"].rolling(28, min_periods=1).mean()
df["rolling_mean_28"] = roll.reset_index(level=[0, 1], drop=True)
df.drop(columns="_shifted_sales", inplace=True)
log("rolling_mean_28 done.")

df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["day_of_month"] = df["date"].dt.day
df["onpromotion_flag"] = (df["onpromotion"] > 0).astype("int8")

# Ordinal-encode categoricals to plain integer codes rather than using
# XGBoost's native categorical-split path (enable_categorical=True): the
# exhaustive categorical partitioning in hist mode was measured to be the
# dominant cost on this dataset (a multi-hour fit vs. seconds for numeric
# splits), so codes + normal numeric splits is the fast, standard choice.
cat_cols = ["family", "type", "cluster", "store_nbr"]
for c in cat_cols:
    df[c] = df[c].astype("category")
    df[c + "_code"] = df[c].cat.codes.astype("int32")
log("Feature engineering complete.")

feature_cols = [
    "lag_7", "lag_14", "rolling_mean_28",
    "day_of_week", "month", "day_of_month", "is_holiday",
    "onpromotion_flag", "dcoilwtico",
    "family_code", "type_code", "cluster_code", "store_nbr_code",
]

model_df = df.dropna(subset=["lag_7", "lag_14", "rolling_mean_28"]).copy()
log(f"Dropped NaN warm-up rows. model_df={len(model_df):,} rows")

max_date = model_df["date"].max()
cutoff = max_date - pd.Timedelta(days=HORIZON)
train_df = model_df.loc[model_df["date"] <= cutoff]
test_df = model_df.loc[model_df["date"] > cutoff]

log(f"Train rows: {len(train_df):,}  Test rows: {len(test_df):,}  Cutoff: {cutoff.date()}")

X_train, y_train = train_df[feature_cols], train_df["sales"]
X_test, y_test = test_df[feature_cols], test_df["sales"]

# ---------------------------------------------------------------------------
# Train XGBoost
# ---------------------------------------------------------------------------
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)
log("Training XGBoost...")
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
log("Training complete.")

pred = model.predict(X_test)
pred = np.clip(pred, 0, None)

# ---------------------------------------------------------------------------
# Evaluation (store-family level, 30-day holdout)
# Three XGBoost views + two naive benchmarks, same test rows:
#  - XGBoost one-step: features use ACTUAL lags inside the test window (what the
#    original script reported; optimistic vs a true 30-day-ahead run)
#  - XGBoost recursive: 30-day-ahead replay from the cutoff, own predictions are
#    fed back into lag/rolling features; true promo/oil/holiday inputs are used
#  - Naive lag-7: sales(t-7) from actuals;  Seasonal naive: last training week repeated
# ---------------------------------------------------------------------------
test_df = test_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
X_test, y_test = test_df[feature_cols], test_df["sales"]
pred = np.clip(model.predict(X_test), 0, None)
y = y_test.to_numpy()
n_series = test_df.groupby(gkeys, observed=True).ngroups
assert len(test_df) == n_series * HORIZON, "test window must be a full series x day grid"

# seasonal naive: repeat the last observed week of the training window
tail = df.loc[df["date"] > cutoff - pd.Timedelta(days=7)]
tail = tail.loc[tail["date"] <= cutoff].sort_values(gkeys + ["date"])
last7 = tail["sales"].to_numpy().reshape(n_series, 7)
snaive = np.tile(last7, (1, int(np.ceil(HORIZON / 7))))[:, :HORIZON].reshape(-1)
lag7 = test_df["lag_7"].to_numpy()

# recursive replay
hist28 = df.loc[(df["date"] > cutoff - pd.Timedelta(days=28)) & (df["date"] <= cutoff)].sort_values(gkeys + ["date"])
buf = hist28["sales"].to_numpy().reshape(n_series, 28).astype("float64")
days = sorted(test_df["date"].unique())
rec_cols = []
for k, d in enumerate(days):
    day = test_df.loc[test_df["date"] == d].sort_values(gkeys)
    Xk = day[feature_cols].copy()
    Xk["lag_7"], Xk["lag_14"], Xk["rolling_mean_28"] = buf[:, -7], buf[:, -14], buf.mean(axis=1)
    pk = np.clip(model.predict(Xk[feature_cols]), 0, None)
    buf = np.roll(buf, -1, axis=1)
    buf[:, -1] = pk
    rec_cols.append(pk)
recursive = np.stack(rec_cols, axis=1).reshape(-1)

# MASE scale per series: in-sample MAE of the lag-7 seasonal-naive on training rows
tr_scale = (train_df.assign(_e=(train_df["sales"] - train_df["lag_7"]).abs())
            .groupby(gkeys, observed=True)["_e"].mean().to_numpy())
ser_idx = np.repeat(np.arange(n_series), HORIZON)


def mase_series(p):
    mae_s = pd.Series(np.abs(y - p)).groupby(ser_idx).mean().to_numpy()
    ok = tr_scale > 0
    return float(np.mean(mae_s[ok] / tr_scale[ok]))


def mase_pooled(p):
    """Pooled MASE: sum|e| / sum(scale over the same rows). Volume-weighted, not dominated by tiny series."""
    ok = np.repeat(tr_scale > 0, HORIZON)
    return float(np.abs(y - p)[ok].sum() / np.repeat(tr_scale, HORIZON)[ok].sum())


views = {"XGBoost (one-step, actual lags)": pred, "XGBoost (recursive 30-day)": recursive,
         "Naive lag-7 (actuals)": lag7, "Seasonal naive (last week repeated)": snaive}
store_family_metrics = {}
for name, p in views.items():
    m = all_metrics(y, p)
    m["MASE"] = mase_series(p)          # mean of per-series MASE (equal weight per series)
    m["MASE_pooled"] = mase_pooled(p)   # volume-weighted
    store_family_metrics[name] = m
    print(f"{name:38s}", {k: round(v, 3) for k, v in m.items()})

# Aggregate (company-wide daily total) level - comparable with SARIMA / Prophet
agg = (test_df.assign(p_xgb=pred, p_rec=recursive, p_sn=snaive, p_l7=lag7)
       .groupby("date", observed=True)[["sales", "p_xgb", "p_rec", "p_sn", "p_l7"]].sum())
daily_tot = df.groupby("date", observed=True)["sales"].sum()
daily_train = daily_tot.loc[daily_tot.index <= cutoff].asfreq("D").fillna(0)
aggregate_metrics = {}
for name, col in [("XGBoost (one-step, actual lags)", "p_xgb"), ("XGBoost (recursive 30-day)", "p_rec"),
                  ("Seasonal naive (last week repeated)", "p_sn"), ("Naive lag-7 (actuals)", "p_l7")]:
    m = all_metrics(agg["sales"], agg[col])
    m["MASE"] = mase(agg["sales"], agg[col], daily_train)
    aggregate_metrics[name] = m

# Per-volume-bucket table (bucket = series mean daily sales over the last 90 TRAIN days: no test hindsight)
v90 = (df.loc[(df["date"] > cutoff - pd.Timedelta(days=90)) & (df["date"] <= cutoff)]
       .groupby(gkeys, observed=True)["sales"].mean().to_numpy())
edges = [-0.01, 0.5, 2, 10, 50, np.inf]
labels = ["<=0.5", "0.5-2", "2-10", "10-50", ">50"]
bucket = pd.cut(pd.Series(v90[ser_idx]), edges, labels=labels).astype(str).to_numpy()
rows = []
for lab in labels:
    mk = bucket == lab
    if not mk.any():
        continue
    r = {"volume_bucket": lab, "n_series": int(mk.sum() // HORIZON), "share_rows_pct": mk.mean() * 100,
         "share_sales_pct": y[mk].sum() / y.sum() * 100, "zero_actuals_pct": zero_share(y[mk])}
    for tag, p in [("xgb", pred), ("xgb_recursive", recursive), ("naive_lag7", lag7), ("seasonal_naive", snaive)]:
        mm = all_metrics(y[mk], p[mk])
        r.update({f"{tag}_MAPE": mm["MAPE"], f"{tag}_WAPE": mm["WAPE"], f"{tag}_sMAPE": mm["sMAPE"]})
    r["xgb_Bias_pct"] = all_metrics(y[mk], pred[mk])["Bias_pct"]
    rows.append(r)
bucket_df = pd.DataFrame(rows)
bucket_df.to_csv(os.path.join(OUT_MODELS, "xgboost_metrics_by_volume_bucket.csv"), index=False)
print(bucket_df[["volume_bucket", "n_series", "share_sales_pct", "xgb_MAPE", "xgb_WAPE", "seasonal_naive_WAPE"]].round(2).to_string(index=False))

head = store_family_metrics["XGBoost (one-step, actual lags)"]
metrics = {
    # legacy keys (unchanged definitions) so downstream readers keep working
    "MAPE": head["MAPE"], "RMSE": head["RMSE"], "Bias_pct": head["Bias_pct"],
    "n_train": len(train_df), "n_test": len(test_df),
    # new
    "WAPE": head["WAPE"], "sMAPE": head["sMAPE"], "MAE": head["MAE"], "MASE": head["MASE"], "MASE_pooled": head["MASE_pooled"],
    "zero_actuals_pct_in_test": zero_share(y),
    "store_family_level": store_family_metrics,
    "aggregate_level": aggregate_metrics,
}
print("XGBoost headline (store-family, one-step):", {k: round(v, 3) for k, v in head.items()})

# Feature importance
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importance:\n", importances)

model.save_model(os.path.join(OUT_MODELS, "xgboost_model.json"))
importances.to_csv(os.path.join(OUT_MODELS, "xgboost_feature_importance.csv"))

with open(os.path.join(OUT_MODELS, "xgboost_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# Save test predictions for actual-vs-forecast Power BI page (aggregated to daily total)
test_df = test_df.copy()
test_df["prediction"] = pred
test_df[["date", "store_nbr", "family", "sales", "prediction"]].assign(recursive_prediction=recursive).to_csv(
    os.path.join(OUT_MODELS, "xgboost_test_predictions.csv"), index=False)  # used by 05 for residual bands
agg = test_df.groupby("date", observed=True).agg(actual=("sales", "sum"), forecast=("prediction", "sum")).reset_index()
agg.to_csv(os.path.join(OUT_MODELS, "xgboost_actual_vs_forecast_daily.csv"), index=False)

# Save the full feature-engineered dataframe (last known row per store-family) for scenario analysis.
# Keep both the human-readable category values and their integer codes,
# since the model was trained on codes but downstream reporting wants labels.
last_rows = (
    model_df.sort_values("date")
    .groupby(["store_nbr", "family"], observed=True)
    .tail(1)
    .reset_index(drop=True)
)
keep_cols = ["date", "store_nbr", "family", "type", "cluster", "sales",
             "lag_7", "lag_14", "rolling_mean_28",
             "family_code", "type_code", "cluster_code", "store_nbr_code"]
last_rows[keep_cols].to_parquet(os.path.join(OUT_MODELS, "last_known_state.parquet"), index=False)

# Persist the category<->code mapping so scenario analysis can decode results
code_maps = {c: dict(enumerate(df[c].cat.categories)) for c in cat_cols}
with open(os.path.join(OUT_MODELS, "category_code_maps.json"), "w") as f:
    json.dump({c: {str(k): str(v) for k, v in m.items()} for c, m in code_maps.items()}, f, indent=2)

print("\nXGBoost model complete. Artifacts -> output/models/")
