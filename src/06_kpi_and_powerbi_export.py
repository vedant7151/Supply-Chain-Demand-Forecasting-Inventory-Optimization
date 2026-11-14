"""
Step 6 - KPI table, Power BI flat tables, insights.md and recommendations.md.
All quantities are UNITS of sales (the dataset has no prices), not currency.
"""
import json
import os
import re

import pandas as pd

OUT_DATA = "output/data"
OUT_EDA = "output/eda"
OUT_MODELS = "output/models"
OUT_SCEN = "output/scenarios"
OUT_REPORTS = "output/reports"
OUT_PBI = "output/powerbi"
os.makedirs(OUT_PBI, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
with open(os.path.join(OUT_MODELS, "baseline_metrics.json")) as f:
    baseline = json.load(f)["metrics"]
with open(os.path.join(OUT_MODELS, "xgboost_metrics.json")) as f:
    xgbm = json.load(f)
with open(os.path.join(OUT_REPORTS, "data_quality_report.json")) as f:
    dq = json.load(f)
with open(os.path.join(OUT_SCEN, "stockout_pareto_summary.json")) as f:
    pareto = json.load(f)

scenario_summary = pd.read_csv(os.path.join(OUT_SCEN, "scenario_comparison_summary.csv"))
scenario_daily = pd.read_csv(os.path.join(OUT_SCEN, "scenario_daily_totals.csv"))
ablation = pd.read_csv(os.path.join(OUT_SCEN, "scenario_lever_ablation.csv"))
stockout_risk = pd.read_csv(os.path.join(OUT_SCEN, "stockout_risk_scores.csv"))
xgb_avf = pd.read_csv(os.path.join(OUT_MODELS, "xgboost_actual_vs_forecast_daily.csv"))
bucket = pd.read_csv(os.path.join(OUT_MODELS, "xgboost_metrics_by_volume_bucket.csv"))
fam_rev = pd.read_csv(os.path.join(OUT_EDA, "family_revenue_ranking.csv"))
with open(os.path.join(OUT_EDA, "eda_findings.md"), encoding="utf-8") as f:
    eda_text = f.read()

sf = xgbm["store_family_level"]
ag = xgbm["aggregate_level"]
sc = scenario_summary.set_index("scenario")
cols = ["MAPE", "WAPE", "sMAPE", "MASE", "RMSE", "Bias_pct"]

# ---------------------------------------------------------------------------
# Page 1 - model comparison (like-for-like grain is explicit) + actual vs forecast
# ---------------------------------------------------------------------------
rows = []
for name, m in baseline.items():
    rows.append({"model": name, "grain": "Aggregate (company daily)", "horizon_basis": "30-day ahead" if "lag-7" not in name else "one-step (actual lags)", **{c: m[c] for c in cols}})
rows.append({"model": "XGBoost aggregated to company daily", "grain": "Aggregate (company daily)", "horizon_basis": "30-day recursive", **{c: ag["XGBoost (recursive 30-day)"][c] for c in cols}})
rows.append({"model": "XGBoost aggregated to company daily (one-step)", "grain": "Aggregate (company daily)", "horizon_basis": "one-step (actual lags)", **{c: ag["XGBoost (one-step, actual lags)"][c] for c in cols}})
rows.append({"model": "XGBoost (production)", "grain": "Store-family", "horizon_basis": "one-step (actual lags)", **{c: sf["XGBoost (one-step, actual lags)"][c] for c in cols}})
rows.append({"model": "XGBoost (store-family, 30-day recursive)", "grain": "Store-family", "horizon_basis": "30-day recursive", **{c: sf["XGBoost (recursive 30-day)"][c] for c in cols}})
rows.append({"model": "Seasonal naive (store-family)", "grain": "Store-family", "horizon_basis": "30-day ahead", **{c: sf["Seasonal naive (last week repeated)"][c] for c in cols}})
rows.append({"model": "Naive lag-7 (store-family)", "grain": "Store-family", "horizon_basis": "one-step (actual lags)", **{c: sf["Naive lag-7 (actuals)"][c] for c in cols}})
model_comparison = pd.DataFrame(rows)
model_comparison.to_csv(os.path.join(OUT_PBI, "page1_model_comparison.csv"), index=False)
bucket.to_csv(os.path.join(OUT_PBI, "page1_metrics_by_volume_bucket.csv"), index=False)
xgb_avf.to_csv(os.path.join(OUT_PBI, "page1_actual_vs_forecast.csv"), index=False)

# ---------------------------------------------------------------------------
# Page 2 / 3 / 4
# ---------------------------------------------------------------------------
scenario_summary.to_csv(os.path.join(OUT_PBI, "page2_scenario_comparison_summary.csv"), index=False)
scenario_daily.to_csv(os.path.join(OUT_PBI, "page2_scenario_daily_totals.csv"), index=False)
ablation.to_csv(os.path.join(OUT_PBI, "page2_scenario_lever_ablation.csv"), index=False)
pd.read_csv(os.path.join(OUT_REPORTS, "data_quality_report.csv")).to_csv(os.path.join(OUT_PBI, "page3_data_quality_report.csv"), index=False)

top_risk = stockout_risk.sort_values("stockout_risk_units", ascending=False).head(200)
top_risk.to_csv(os.path.join(OUT_PBI, "page4_top_stockout_risk.csv"), index=False)
stockout_risk.to_csv(os.path.join(OUT_PBI, "page4_stockout_risk_all.csv"), index=False)

# ---------------------------------------------------------------------------
# KPI summary (WAPE is the accuracy headline; MAPE kept but labelled)
# ---------------------------------------------------------------------------
kpis = [
    ("Total Forecasted Demand, next 30 days, Base Case (units)", round(sc.loc["Base Case", "total_forecast_demand"], 0)),
    ("Base Case 30-day demand P10 (units)", round(sc.loc["Base Case", "p10_total_units"], 0)),
    ("Base Case 30-day demand P90 (units)", round(sc.loc["Base Case", "p90_total_units"], 0)),
    ("Forecast Accuracy - WAPE % (XGBoost, store-family, 30-day recursive)", round(sf["XGBoost (recursive 30-day)"]["WAPE"], 2)),
    ("Forecast Accuracy - WAPE % (XGBoost, store-family, one-step)", round(sf["XGBoost (one-step, actual lags)"]["WAPE"], 2)),
    ("Forecast Accuracy - WAPE % (XGBoost, aggregated to company daily)", round(ag["XGBoost (recursive 30-day)"]["WAPE"], 2)),
    ("MASE pooled (XGBoost store-family one-step; <1 beats seasonal naive in-sample)", round(sf["XGBoost (one-step, actual lags)"]["MASE_pooled"], 2)),
    ("MAPE % (XGBoost store-family, non-zero actuals only; secondary metric)", round(xgbm["MAPE"], 2)),
    ("Forecast Bias % (XGBoost store-family)", round(xgbm["Bias_pct"], 2)),
    ("Optimistic vs Base, % units (point)", round(sc.loc["Optimistic", "units_delta_pct_vs_base"], 1)),
    ("Optimistic vs Base P10, % units", round(sc.loc["Optimistic", "p10_delta_pct_vs_base_p50"], 1)),
    ("Optimistic vs Base P90, % units", round(sc.loc["Optimistic", "p90_delta_pct_vs_base_p50"], 1)),
    ("Pessimistic vs Base, % units (point)", round(sc.loc["Pessimistic", "units_delta_pct_vs_base"], 1)),
    ("Pessimistic vs Base P10, % units", round(sc.loc["Pessimistic", "p10_delta_pct_vs_base_p50"], 1)),
    ("Pessimistic vs Base P90, % units", round(sc.loc["Pessimistic", "p90_delta_pct_vs_base_p50"], 1)),
    ("Active Scenarios", scenario_summary["scenario"].nunique()),
    ("Store-Product Combinations Scored (sold in last 28 days)", pareto["active_combos_scored"]),
    ("Combinations Excluded (zero sales in last 28 days)", pareto["excluded_zero_sales_last_28d"]),
    ("Store-Product Combinations at Elevated Stockout Risk", pareto["combos_at_stockout_risk"]),
    ("Total Positive Stockout Risk (units, proxy-based)", pareto["total_positive_risk_units"]),
    ("% of Positive-Risk Combos Driving 80% of Risk Units (Pareto)", pareto["pct_combos_driving_80pct_of_risk"]),
    ("Rows Removed in Data Cleaning (negative sales)", dq["summary"]["rows_removed"]),
    ("Rows Flagged as Outliers (kept, not removed)", dq["summary"]["outliers_flagged"]),
]
kpi_summary = pd.DataFrame(kpis, columns=["kpi", "value"])
kpi_summary.to_csv(os.path.join(OUT_PBI, "kpi_summary.csv"), index=False)
print("KPI Summary:\n", kpi_summary.to_string(index=False))

# ---------------------------------------------------------------------------
# insights.md
# ---------------------------------------------------------------------------
fmt = lambda d: d.round(2).to_markdown(index=False)
bucket_view = bucket[["volume_bucket", "n_series", "share_sales_pct", "zero_actuals_pct", "xgb_MAPE", "xgb_WAPE",
                      "seasonal_naive_WAPE", "naive_lag7_WAPE"]]
scen_view = scenario_summary[["scenario", "total_forecast_demand", "units_delta_vs_base", "units_delta_pct_vs_base",
                              "p10_total_units", "p50_total_units", "p90_total_units",
                              "p10_delta_pct_vs_base_p50", "p90_delta_pct_vs_base_p50"]]
top5_fam_pct = fam_rev.head(5)["total_sales"].sum() / fam_rev["total_sales"].sum() * 100

insights_md = f"""# Insights — Demand Forecasting Project

All demand figures are **units of sales** (no price data exists, so nothing here is currency).

{eda_text}

## Model Performance (30-day holdout)

Headline accuracy metric: **WAPE** (volume-weighted). MAPE excludes zero-sales rows
({xgbm['zero_actuals_pct_in_test']:.1f}% of store-family test rows) and is dominated by tiny series, so it is secondary.

{fmt(model_comparison)}

- Like-for-like at company-daily level, XGBoost has WAPE {ag['XGBoost (recursive 30-day)']['WAPE']:.2f}% (30-day recursive) vs
  SARIMA {baseline['SARIMA']['WAPE']:.2f}%, Prophet {baseline['Prophet']['WAPE']:.2f}% and a seasonal-naive {baseline['Seasonal naive (m=7)']['WAPE']:.2f}%.
  SARIMA is only marginally better than seasonal-naive, and Prophet is worse.
- At store-family level XGBoost WAPE is {sf['XGBoost (one-step, actual lags)']['WAPE']:.2f}% one-step /
  {sf['XGBoost (recursive 30-day)']['WAPE']:.2f}% 30-day recursive, vs {sf['Seasonal naive (last week repeated)']['WAPE']:.2f}% for seasonal-naive.
- Store-family MAPE is {xgbm['MAPE']:.2f}% only because of low-volume series: see the volume-bucket table.

### Accuracy by series volume (bucket = mean daily sales over last 90 training days)

{fmt(bucket_view)}

XGBoost beats the naive benchmarks on the high-volume series that carry ~99% of units; on series selling
under ~10 units/day, seasonal-naive is as good or better.

## Scenario Analysis (30 days, units)

{fmt(scen_view)}

Point forecast = deterministic run; P10/P50/P90 = {200} Monte Carlo runs (promo share, oil, residual noise);
`*_delta_pct_vs_base_p50` compares each scenario quantile with the Base Case P50.

**Band caveat:** the Monte Carlo band covers promo-share, oil and independent per-series-day noise only. It is narrower than the
holdout's company-daily WAPE ({ag['XGBoost (recursive 30-day)']['WAPE']:.1f}%), because independent noise averages out across series and
systematic model error is not sampled. Treat it as a lower bound on uncertainty and plan on roughly +/-{ag['XGBoost (recursive 30-day)']['WAPE']:.0f}% around the Base Case.

### What drives the scenario gap (one lever at a time, deterministic)

{fmt(ablation)}

## Stockout Risk (units, inventory is a PROXY)

- {pareto['active_combos_scored']} store-product combinations are scored; {pareto['excluded_zero_sales_last_28d']} are excluded (no sales in the last 28 days).
- {pareto['positive_risk_combos']} have positive risk (forecast > proxy inventory); {pareto['combos_at_stockout_risk']} are "elevated" (risk > 5% of forecast demand).
- {pareto['n_combos_driving_80pct_of_risk_units']} combinations ({pareto['pct_combos_driving_80pct_of_risk']:.1f}% of positive-risk combos,
  {pareto['pct_of_active_combos_driving_80pct']:.1f}% of scored combos) account for 80% of the {pareto['total_positive_risk_units']:,.0f} risk units.
- Inventory proxy = trailing 28-day mean x 30. With no on-hand/inbound data these are prioritisation signals, not purchase orders.
"""
with open(os.path.join(OUT_REPORTS, "insights.md"), "w", encoding="utf-8") as f:
    f.write(insights_md)

# ---------------------------------------------------------------------------
# recommendations.md
# ---------------------------------------------------------------------------
m = re.search(r"run ([\d.]+)% higher", eda_text)
pre_hol = f"{m.group(1)}%" if m else "a small"
ab = ablation.set_index("lever")["delta_pct_vs_base"]
wape_ref = sf["XGBoost (recursive 30-day)"]["WAPE"]
recommendations_md = f"""# Business Recommendations

All quantities are units of sales, not currency. Evidence strength is stated explicitly.

| Recommendation | Based On | Evidence strength | For Whom |
|---|---|---|---|
| Treat pre-holiday procurement advances as optional: sales in the 7 days before national holidays are only {pre_hol} above the prior baseline | Pre-holiday EDA | Weak - small effect | Category Planners |
| Prefer promotion families with high lift AND high volume; do not use raw lift % (association, not causal; tiny baselines inflate it) | `output/eda/promotion_lift_by_family.csv` | Moderate - observational | Marketing & Supply Chain Manager |
| Review safety stock first for the {pareto['n_combos_driving_80pct_of_risk_units']} combinations ({pareto['pct_combos_driving_80pct_of_risk']:.0f}% of positive-risk combos) driving 80% of risk units, ranked by units (Page 4) | Pareto on risk units | Moderate - proxy inventory, not on-hand data | Supply Chain Manager |
| Monitor WTI oil, but do not trigger demand changes on it: model sensitivity is small (oil -10% only: {ab['Oil -10% only']:+.1f}%, oil +15% only: {ab['Oil +15% only']:+.1f}% of 30-day units) | Scenario lever ablation | Weak - correlation in EDA is trend-driven | Finance & Planning Team |
| Prioritise inventory accuracy for the top 5 families, {top5_fam_pct:.1f}% of total sales | Family ranking | Strong | Category Planners |
| Reforecast weekly; retrain when rolling store-family WAPE exceeds {wape_ref*1.5:.1f}% (1.5x the {wape_ref:.1f}% 30-day recursive benchmark) | Holdout WAPE | Moderate | Analytics Team |
| Scenario range for the next 30 days (Base {sc.loc['Base Case','total_forecast_demand']:,.0f} units, P10-P90 {sc.loc['Base Case','p10_total_units']:,.0f} to {sc.loc['Base Case','p90_total_units']:,.0f}): the Monte Carlo band is a lower bound; plan open purchase orders with roughly +/-{ag['XGBoost (recursive 30-day)']['WAPE']:.0f}% headroom (holdout company-daily WAPE) | Monte Carlo bands | Moderate | Supply Chain Manager |
| Pessimistic (promotions paused): units {sc.loc['Pessimistic','units_delta_pct_vs_base']:+.1f}% vs base - the gap is the promotion assumption, not oil; trim orders only if the promo calendar is actually cut | Scenario analysis | Moderate | Supply Chain Manager |
| Optimistic (promo frequency +30%, holiday flag, oil -10%): units {sc.loc['Optimistic','units_delta_pct_vs_base']:+.1f}% vs base (P10-P90 {sc.loc['Optimistic','p10_delta_pct_vs_base_p50']:+.1f}% to {sc.loc['Optimistic','p90_delta_pct_vs_base_p50']:+.1f}%) - pre-build stock only if promotions are confirmed | Scenario analysis | Moderate | Supply Chain Manager |
"""
with open(os.path.join(OUT_REPORTS, "recommendations.md"), "w", encoding="utf-8") as f:
    f.write(recommendations_md)

print("\nFinal export complete.")
