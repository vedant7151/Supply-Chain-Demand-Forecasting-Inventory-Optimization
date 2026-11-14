"""
Step 3 - Baseline forecasting models: SARIMA and Prophet (+ naive benchmarks).
Run at AGGREGATE (total company-wide daily sales) level, per the guide's
"Why NOT only Prophet" / "classical baseline" reasoning: SARIMA/Prophet
are demonstrated here as interpretable benchmarks; XGBoost (04) is the
store-family-level production model.
"""
import json
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet

from metrics_util import all_metrics, mase

OUT_DATA = "output/data"
OUT_MODELS = "output/models"
os.makedirs(OUT_MODELS, exist_ok=True)

HORIZON = 30  # forecast horizon in days

train = pd.read_parquet(os.path.join(OUT_DATA, "fact_sales.parquet"))
holidays = pd.read_parquet(os.path.join(OUT_DATA, "dim_holidays.parquet"))

daily = train.groupby("date", observed=True)["sales"].sum().asfreq("D").fillna(0)
# NOTE: asfreq fills the 4 closed Dec-25 dates with 0 (stores closed). Kept for SARIMA (needs a regular grid).

train_series = daily.iloc[:-HORIZON]
test_series = daily.iloc[-HORIZON:]

metrics = {}


# ---------------------------------------------------------------------------
# Stationarity check (ADF test) - documented per the guide
# ---------------------------------------------------------------------------
adf_stat, adf_p, *_ = adfuller(train_series)
stationarity_note = (
    f"ADF statistic={adf_stat:.3f}, p-value={adf_p:.4f} -> "
    + ("series is stationary (p<=0.05), no differencing strictly required."
       if adf_p <= 0.05 else
       "series is non-stationary (p>0.05); SARIMA's d/D differencing terms handle this.")
)
print("[ADF]", stationarity_note)

# ---------------------------------------------------------------------------
# Model A: SARIMA  (weekly seasonality -> m=7)
# ---------------------------------------------------------------------------
print("Fitting SARIMA(1,1,1)(1,1,1,7) ...")
sarima_model = SARIMAX(
    train_series,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

sarima_fc = sarima_model.forecast(steps=HORIZON)
metrics["SARIMA"] = {**all_metrics(test_series, sarima_fc), "MASE": mase(test_series, sarima_fc, train_series)}
print("SARIMA:", metrics["SARIMA"])

# ---------------------------------------------------------------------------
# Model B: Prophet (with Ecuador national holidays)
# ---------------------------------------------------------------------------
print("Fitting Prophet ...")
holiday_df = (
    holidays.loc[holidays["locale"] == "National", ["date", "description"]]
    .rename(columns={"date": "ds", "description": "holiday"})
    .drop_duplicates(subset="ds")
)

prophet_train = train_series.reset_index()
prophet_train.columns = ["ds", "y"]

prophet_model = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False,
    holidays=holiday_df,
)
prophet_model.fit(prophet_train)

future = prophet_model.make_future_dataframe(periods=HORIZON)
prophet_fc_full = prophet_model.predict(future)
prophet_fc = prophet_fc_full.set_index("ds")["yhat"].iloc[-HORIZON:]

metrics["Prophet"] = {**all_metrics(test_series, prophet_fc), "MASE": mase(test_series, prophet_fc, train_series)}
print("Prophet:", metrics["Prophet"])

# ---------------------------------------------------------------------------
# Naive benchmarks (aggregate level)
#  - Seasonal naive (m=7): repeat the last observed training week over the horizon
#    (an honest 30-day-ahead forecast, uses nothing from the test window).
#  - Naive lag-7: y[t-7] using actuals (one-step-ahead; same information the
#    XGBoost lag_7 feature uses at evaluation time).
# ---------------------------------------------------------------------------
last_week = train_series.iloc[-7:].to_numpy()
snaive_fc = np.array([last_week[i % 7] for i in range(HORIZON)])
metrics["Seasonal naive (m=7)"] = {**all_metrics(test_series, snaive_fc), "MASE": mase(test_series, snaive_fc, train_series)}
lag7_fc = daily.shift(7).iloc[-HORIZON:].to_numpy()
metrics["Naive lag-7 (actuals)"] = {**all_metrics(test_series, lag7_fc), "MASE": mase(test_series, lag7_fc, train_series)}
for k in ("Seasonal naive (m=7)", "Naive lag-7 (actuals)"):
    print(k, metrics[k])

# ---------------------------------------------------------------------------
# Plot: actual vs both baselines
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
daily.iloc[-90:].plot(ax=ax, label="Actual", color="black")
sarima_fc.plot(ax=ax, label="SARIMA Forecast", color="#3B7DD8", linestyle="--")
ax.plot(prophet_fc.index, prophet_fc.values, label="Prophet Forecast", color="#D8433B", linestyle="--")
ax.axvline(train_series.index[-1], color="gray", linestyle=":")
ax.legend()
ax.set_title(f"Baseline Models: Actual vs Forecast (last {HORIZON} days)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_MODELS, "baseline_actual_vs_forecast.png"), dpi=130)
plt.close()

with open(os.path.join(OUT_MODELS, "baseline_metrics.json"), "w") as f:
    json.dump({"stationarity": stationarity_note, "metrics": metrics}, f, indent=2)

print("\nBaseline models complete. Metrics -> output/models/baseline_metrics.json")
