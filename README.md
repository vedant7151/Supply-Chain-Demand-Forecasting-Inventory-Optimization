# Supply Chain Demand Forecasting & Inventory Optimization

Implements the full analysis layer from `supply_chain_forecasting_guide.md`
(data integrity → EDA → KPIs → SARIMA/Prophet/XGBoost forecasting →
scenario analysis → stockout risk) against the Corporación Favorita
dataset. Power BI dashboard build is separate (manual).

## How to reproduce

```
pip install -r requirements.txt   # pinned versions
python src/01_data_integrity.py
python src/02_eda.py
python src/03_baseline_models.py       # SARIMA + Prophet (aggregate level)
python src/04_xgboost_model.py         # XGBoost (store-family level, production model)
python src/05_scenario_analysis.py     # Base/Optimistic/Pessimistic + stockout risk
python src/06_kpi_and_powerbi_export.py
```

Run in order — each step reads parquet/artifacts written by the previous one.

## Output layout

- `output/data/` — cleaned fact/dim tables (parquet)
- `output/reports/` — data quality report, `insights.md`, `recommendations.md`
- `output/eda/` — EDA charts (PNG) + supporting CSVs
- `output/models/` — trained SARIMA/Prophet/XGBoost metrics + XGBoost model file
- `output/scenarios/` — scenario forecasts, comparison summary, stockout risk scores
- `output/powerbi/` — **flat tables pre-shaped for the 4 dashboard pages** (below)

## Power BI table → page mapping

| Dashboard Page | Source file(s) |
|---|---|
| **Page 1 — Executive Overview** | `powerbi/kpi_summary.csv` (KPI cards), `powerbi/page1_actual_vs_forecast.csv` (line chart), `powerbi/page1_model_comparison.csv` (all models + naive baselines: MAPE/WAPE/sMAPE/MASE/RMSE/Bias, with grain), `powerbi/page1_metrics_by_volume_bucket.csv` |
| **Page 2 — Scenario Comparison** | `powerbi/page2_scenario_comparison_summary.csv` (bar chart, P10/P50/P90 + plain-English text per scenario), `powerbi/page2_scenario_daily_totals.csv` (daily trend lines + p10/p90 bands), `powerbi/page2_scenario_lever_ablation.csv` |
| **Page 3 — Data Quality Report** | `powerbi/page3_data_quality_report.csv` |
| **Page 4 — Operational Stockout Risk** | `powerbi/page4_top_stockout_risk.csv` (top 200 for the table), `powerbi/page4_stockout_risk_all.csv` (full set for the heatmap, store_nbr × family) |

Slicers for Page 1/4 (store type, product family, date range) can be built
by joining these tables back to `output/data/dim_stores.parquet` on `store_nbr`
if you want store `type`/`city`/`cluster` as slicer fields — Power BI can
read parquet directly, or re-export it to CSV if you need CSV-only sources.

## Key results

All demand figures are units of sales (no price data). Details and before/after: `CHANGELOG.md`.

- **Accuracy headline = WAPE** (MAPE drops zero-sales rows and is dominated by tiny series).
  - Company-daily level (30-day holdout): XGBoost aggregated WAPE 4.89% vs SARIMA 7.10%, Prophet 8.62%, seasonal-naive 7.28%.
  - Store-family level: XGBoost WAPE 13.36% (one-step) / 14.14% (30-day recursive) vs seasonal-naive 17.05%; MAPE 50.47%
    (secondary: 14.7% of test rows are zero and excluded; series selling <2 units/day have MAPE >100% but ~0.01% of volume).
- **Scenarios (30 days, units):** Base 24,764,504 (P10-P90 24,018,089-24,842,392); Optimistic +10.1%; Pessimistic -4.6%.
  Scenarios change inputs only (promo share vs the observed last-30-day share, oil, holiday flag); promotions are the main driver, oil is minor
  (`output/scenarios/scenario_lever_ablation.csv`). The Monte Carlo band is a lower bound on uncertainty (plan on about +/-5%).
- **Stockout risk (proxy inventory = trailing 28-day mean x 30; no on-hand data):** 1,668 combos scored (114 with no sales in 28 days excluded);
  556 elevated; 90 combos (10.9% of the 823 positive-risk combos) drive 80% of 412,218 risk units. Ranked by units.

Full plain-English write-up: `output/reports/insights.md` and
`output/reports/recommendations.md`.
