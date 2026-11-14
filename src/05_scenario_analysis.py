"""
Step 5 - Scenario analysis (what-if + Monte Carlo bands) and stockout-risk scoring.

Uses the trained XGBoost model to forecast 30 days ahead per store-family under
three scenarios. Design (changed from the first version - see CHANGELOG.md):

* The 28-day lag buffer is initialised from the TRUE last 28 days of sales.
* Scenarios change INPUTS only (promo share, oil, holiday flag). Nothing is
  multiplied onto the predictions, so nothing compounds through the lag buffer.
* Base-case promo input = each series' OBSERVED promo share over the last 30 days
  (share of days with onpromotion > 0), spread evenly over the horizon.
  Optimistic = share x 1.30 (capped at 1); Pessimistic = promotions paused (0).
* Features are clipped to training support: lags/rolling/predictions to
  [0, series max training sales]; oil to the training range; the promo flag is 0
  for series that never had a promotion in training (e.g. BOOKS).
* Monte Carlo: per scenario, N_MC runs with (a) Bernoulli daily promo flags with a
  lognormally perturbed share, (b) a lognormal oil shock, (c) per-series-day
  zero-mean observation noise bootstrapped from the out-of-sample 30-day recursive holdout
  relative residuals (04, by predicted-volume bucket), added to each reported series-day draw. Reported as P10/P50/P90.
* Stockout risk is expressed in UNITS (forecast_30d - assumed_inventory), ranks by
  units, excludes series with zero sales in the last 28 days, and the Pareto uses
  positive-risk combos as its denominator. There is still NO real inventory data:
  inventory is a proxy (trailing 28-day mean x 30).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

_t0 = time.time()


def log(msg):
    print(f"[{time.time()-_t0:7.1f}s] {msg}", flush=True)


OUT_DATA = "output/data"
OUT_MODELS = "output/models"
OUT_SCEN = "output/scenarios"
os.makedirs(OUT_SCEN, exist_ok=True)

HORIZON = 30
BUFFER_LEN = 28
SERVICE_LEVEL_Z = 1.65      # ~95% one-sided
LEAD_TIME_DAYS = 7          # assumed reorder cycle
ELEVATED_PCT = 0.05         # materiality threshold for "elevated" (5% of forecast demand)
N_MC = 200                  # Monte Carlo runs per scenario
MC_BATCH = 50               # runs predicted together
PROMO_SHARE_SIGMA = 0.10    # lognormal sigma on promo share (planning uncertainty, assumption)
SEED = 42

SCENARIOS = {
    "Base Case":   {"oil_delta_pct": 0.0,   "promo_share_mult": 1.0, "holiday_boost": False},
    "Optimistic":  {"oil_delta_pct": -0.10, "promo_share_mult": 1.3, "holiday_boost": True},
    "Pessimistic": {"oil_delta_pct": 0.15,  "promo_share_mult": 0.0, "holiday_boost": False},
}

feature_cols = [
    "lag_7", "lag_14", "rolling_mean_28",
    "day_of_week", "month", "day_of_month", "is_holiday",
    "onpromotion_flag", "dcoilwtico",
    "family_code", "type_code", "cluster_code", "store_nbr_code",
]

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
model = xgb.XGBRegressor()
model.load_model(os.path.join(OUT_MODELS, "xgboost_model.json"))
last_state = pd.read_parquet(os.path.join(OUT_MODELS, "last_known_state.parquet")).reset_index(drop=True)
fact = pd.read_parquet(os.path.join(OUT_DATA, "fact_sales.parquet"), columns=["date", "store_nbr", "family", "sales", "onpromotion"])
oil = pd.read_parquet(os.path.join(OUT_DATA, "dim_oil.parquet")).sort_values("date")
holidays = pd.read_parquet(os.path.join(OUT_DATA, "dim_holidays.parquet"))
resid = pd.read_csv(os.path.join(OUT_MODELS, "xgboost_test_predictions.csv"))
fact["family"] = fact["family"].astype(str)
last_state["family"] = last_state["family"].astype(str)

n = len(last_state)
keys = pd.MultiIndex.from_arrays([last_state["store_nbr"].to_numpy(), last_state["family"].to_numpy()])
codes = {c: last_state[c].to_numpy() for c in ["family_code", "type_code", "cluster_code", "store_nbr_code"]}
max_hist_date = fact["date"].max()
assert last_state["date"].max() == max_hist_date
future_dates = pd.date_range(max_hist_date + pd.Timedelta(days=1), periods=HORIZON)
nat_holiday_dates = set(holidays.loc[holidays["locale"] == "National", "date"].dt.date)
log(f"Loaded {n} series; last observed date {max_hist_date.date()}")

# True last-28-day history (n x 28), oldest -> newest
h_start = max_hist_date - pd.Timedelta(days=BUFFER_LEN - 1)
hist = (fact.loc[fact["date"] >= h_start].groupby(["store_nbr", "family", "date"])["sales"].sum()
        .unstack("date").reindex(keys))
assert hist.shape[1] == BUFFER_LEN and not hist.isna().any().any(), "28-day history must be complete"
hist28 = hist.to_numpy(dtype="float64")

# Observed promo share over the last 30 days (per series)
p_start = max_hist_date - pd.Timedelta(days=HORIZON - 1)
promo_share = (fact.loc[fact["date"] >= p_start].assign(f=lambda d: (d["onpromotion"] > 0).astype(float))
               .groupby(["store_nbr", "family"])["f"].mean().reindex(keys).fillna(0.0).to_numpy())

# Training support
ever_promoted = (fact.groupby(["store_nbr", "family"])["onpromotion"].max().reindex(keys).fillna(0).to_numpy() > 0)
series_cap = fact.groupby(["store_nbr", "family"])["sales"].max().reindex(keys).to_numpy(dtype="float64")
oil_lo, oil_hi = float(oil["dcoilwtico"].min()), float(oil["dcoilwtico"].max())
current_oil = float(oil["dcoilwtico"].iloc[-1])
oil_sigma = float(np.log(oil["dcoilwtico"]).diff(21).dropna().std())  # ~1 month of trading days
log(f"current oil {current_oil:.2f}; oil sigma(21 obs) {oil_sigma:.3f}; mean observed promo share {promo_share.mean():.3f}")

# Residual bootstrap pool: relative error of the 30-day RECURSIVE holdout, bucketed by predicted level
RES_EDGES = np.array([2.0, 10.0, 50.0])
rp = resid["recursive_prediction"].to_numpy()
rel_res = (resid["sales"].to_numpy() - rp) / np.maximum(rp, 1.0)
rbucket = np.digitize(rp, RES_EDGES)
# Centre each bucket (zero-mean noise): the band should reflect DISPERSION. Un-centred residuals carry the
# holdout's per-bucket bias (measured: -7.5% drift on the Base Case median when fed through the recursion).
res_pool = [rel_res[rbucket == b] - rel_res[rbucket == b].mean() for b in range(len(RES_EDGES) + 1)]


def draw_noise(pred, rng):
    b = np.digitize(pred, RES_EDGES)
    out = np.zeros_like(pred)
    for k, pool in enumerate(res_pool):
        m = b == k
        if m.any():
            out[m] = rng.choice(pool, size=int(m.sum()))
    return out * np.maximum(pred, 1.0)


# ---------------------------------------------------------------------------
# Simulation (vectorised over R runs x n series)
# ---------------------------------------------------------------------------
def simulate(params, R=1, stochastic=False, rng=None):
    """Returns array (HORIZON, R, n) of daily forecasts. Nothing is scaled after prediction."""
    share = np.clip(promo_share * params["promo_share_mult"], 0, 1) * ever_promoted      # (n,)
    if stochastic:
        share_run = np.clip(share[None, :] * rng.lognormal(0, PROMO_SHARE_SIGMA, size=R)[:, None], 0, 1)
        oil_run = np.clip(current_oil * (1 + params["oil_delta_pct"]) * np.exp(rng.normal(0, oil_sigma, size=R)), oil_lo, oil_hi)
    else:
        share_run = share[None, :]
        oil_run = np.array([np.clip(current_oil * (1 + params["oil_delta_pct"]), oil_lo, oil_hi)])
    cap = np.tile(series_cap, R)
    buf = np.tile(hist28, (R, 1))                                                          # (R*n, 28)
    fam, typ, clu, sto = (np.tile(codes[c], R) for c in ["family_code", "type_code", "cluster_code", "store_nbr_code"])
    oil_col = np.repeat(oil_run, n)
    out = np.zeros((HORIZON, R, n))
    for k, d in enumerate(future_dates):
        if stochastic:
            flag = (rng.random((R, n)) < share_run).astype("int8")
        else:  # deterministic, evenly spread: floor((k+1)p+.5)-floor(kp+.5)
            flag = (np.floor((k + 1) * share_run + 0.5) - np.floor(k * share_run + 0.5)).astype("int8")
            flag = np.repeat(flag, R, axis=0) if R > 1 else flag
        hol = int(d.date() in nat_holiday_dates) or int(params["holiday_boost"] and k < 5)
        X = pd.DataFrame({
            "lag_7": np.minimum(buf[:, -7], cap), "lag_14": np.minimum(buf[:, -14], cap),
            "rolling_mean_28": np.minimum(buf.mean(axis=1), cap),
            "day_of_week": d.dayofweek, "month": d.month, "day_of_month": d.day, "is_holiday": hol,
            "onpromotion_flag": flag.reshape(-1), "dcoilwtico": oil_col,
            "family_code": fam, "type_code": typ, "cluster_code": clu, "store_nbr_code": sto,
        })[feature_cols]
        p = np.minimum(np.clip(model.predict(X), 0, None), cap)
        buf = np.roll(buf, -1, axis=1)
        buf[:, -1] = p                                   # lag buffer carries the noise-free prediction
        # observation noise: added to the REPORTED draw only (feeding it back adds bias, not spread - measured)
        out[k] = (np.clip(p + draw_noise(p, rng), 0, None) if stochastic else p).reshape(R, n)
    return out


# ---------------------------------------------------------------------------
# Deterministic point forecasts + lever ablation
# ---------------------------------------------------------------------------
det = {}
frames = []
for name, prm in SCENARIOS.items():
    log(f"Deterministic scenario: {name}")
    det[name] = simulate(prm)[:, 0, :]                                   # (H, n)
    frames.append(pd.DataFrame({
        "date": np.repeat(future_dates.to_numpy(), n),
        "store_nbr": np.tile(last_state["store_nbr"].to_numpy(), HORIZON),
        "family": np.tile(last_state["family"].to_numpy(), HORIZON),
        "forecast_sales": det[name].reshape(-1), "scenario": name}))
scenario_all = pd.concat(frames, ignore_index=True)
scenario_all.to_csv(os.path.join(OUT_SCEN, "scenario_forecast_detail.csv"), index=False)

base_total = float(det["Base Case"].sum())
ablate = {
    "Oil -10% only": {"oil_delta_pct": -0.10, "promo_share_mult": 1.0, "holiday_boost": False},
    "Oil +15% only": {"oil_delta_pct": 0.15, "promo_share_mult": 1.0, "holiday_boost": False},
    "Promo share x1.3 only": {"oil_delta_pct": 0.0, "promo_share_mult": 1.3, "holiday_boost": False},
    "Promotions paused only": {"oil_delta_pct": 0.0, "promo_share_mult": 0.0, "holiday_boost": False},
    "Holiday flag (first 5 days) only": {"oil_delta_pct": 0.0, "promo_share_mult": 1.0, "holiday_boost": True},
}
ab_rows = []
for name, prm in ablate.items():
    t = float(simulate(prm)[:, 0, :].sum())
    ab_rows.append({"lever": name, "total_forecast_units": t, "delta_units_vs_base": t - base_total,
                    "delta_pct_vs_base": (t / base_total - 1) * 100})
pd.DataFrame(ab_rows).to_csv(os.path.join(OUT_SCEN, "scenario_lever_ablation.csv"), index=False)
log("Ablation done.")

# ---------------------------------------------------------------------------
# Monte Carlo bands
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
mc_totals, mc_daily = {}, {}
for name, prm in SCENARIOS.items():
    chunks = []
    for b in range(0, N_MC, MC_BATCH):
        chunks.append(simulate(prm, R=min(MC_BATCH, N_MC - b), stochastic=True, rng=rng))
        log(f"MC {name}: {b + MC_BATCH}/{N_MC}")
    arr = np.concatenate(chunks, axis=1)                  # (H, N_MC, n)
    mc_daily[name] = arr.sum(axis=2)                      # (H, N_MC)
    mc_totals[name] = mc_daily[name].sum(axis=0)          # (N_MC,)
pd.DataFrame(mc_totals).to_csv(os.path.join(OUT_SCEN, "scenario_mc_run_totals.csv"), index=False)

q = lambda a, p: float(np.percentile(a, p))
daily_rows = []
for name in SCENARIOS:
    for k, d in enumerate(future_dates):
        daily_rows.append({"scenario": name, "date": d, "forecast_sales": float(det[name][k].sum()),
                           "p10": q(mc_daily[name][k], 10), "p50": q(mc_daily[name][k], 50), "p90": q(mc_daily[name][k], 90)})
pd.DataFrame(daily_rows).to_csv(os.path.join(OUT_SCEN, "scenario_daily_totals.csv"), index=False)

base_p50 = q(mc_totals["Base Case"], 50)
rows = []
for name in SCENARIOS:
    t = float(det[name].sum())
    rows.append({
        "scenario": name, "total_forecast_demand": t,
        "units_delta_vs_base": t - base_total, "units_delta_pct_vs_base": (t / base_total - 1) * 100,
        "p10_total_units": q(mc_totals[name], 10), "p50_total_units": q(mc_totals[name], 50), "p90_total_units": q(mc_totals[name], 90),
        "p10_delta_pct_vs_base_p50": (q(mc_totals[name], 10) / base_p50 - 1) * 100,
        "p50_delta_pct_vs_base_p50": (q(mc_totals[name], 50) / base_p50 - 1) * 100,
        "p90_delta_pct_vs_base_p50": (q(mc_totals[name], 90) / base_p50 - 1) * 100,
    })
summary = pd.DataFrame(rows)
s = summary.set_index("scenario")
plain = {
    "Base Case": (f"Each store-family keeps its recently observed promotion frequency (mean share {promo_share.mean()*100:.0f}% of days) "
                  f"and oil stays at {current_oil:.1f}. Planning baseline: {s.loc['Base Case','total_forecast_demand']:,.0f} units over 30 days "
                  f"(P10-P90 {s.loc['Base Case','p10_total_units']:,.0f} to {s.loc['Base Case','p90_total_units']:,.0f})."),
    "Optimistic": (f"Promotion frequency +30%, oil -10%, and a holiday flag on the first 5 days: "
                   f"{s.loc['Optimistic','units_delta_pct_vs_base']:+.1f}% units vs base "
                   f"(P10-P90 {s.loc['Optimistic','p10_delta_pct_vs_base_p50']:+.1f}% to {s.loc['Optimistic','p90_delta_pct_vs_base_p50']:+.1f}%). "
                   f"Oil alone moves the forecast very little; most of the change is the promotion assumption."),
    "Pessimistic": (f"Promotions paused entirely and oil +15%: {s.loc['Pessimistic','units_delta_pct_vs_base']:+.1f}% units vs base "
                    f"(P10-P90 {s.loc['Pessimistic','p10_delta_pct_vs_base_p50']:+.1f}% to {s.loc['Pessimistic','p90_delta_pct_vs_base_p50']:+.1f}%). "
                    f"This is essentially a 'no promotions' case."),
}
summary["plain_english_summary"] = summary["scenario"].map(plain)
summary.to_csv(os.path.join(OUT_SCEN, "scenario_comparison_summary.csv"), index=False)
print("\nScenario summary (units):\n", summary.drop(columns="plain_english_summary").round(1).to_string(index=False))
print("\nLever ablation:\n", pd.DataFrame(ab_rows).round(1).to_string(index=False))

# ---------------------------------------------------------------------------
# Stockout risk (units). Base Case deterministic forecast vs inventory PROXY.
# ---------------------------------------------------------------------------
base = det["Base Case"]                                      # (H, n)
active = hist28.sum(axis=1) > 0                              # sold something in the last 28 days
risk = pd.DataFrame({
    "store_nbr": last_state["store_nbr"].to_numpy(), "family": last_state["family"].to_numpy(),
    "forecast_30d_demand": base.sum(axis=0), "forecast_daily_mean": base.mean(axis=0),
    "forecast_daily_std": base.std(axis=0, ddof=1),
    "assumed_available_inventory": hist28.mean(axis=1) * HORIZON,   # PROXY: trailing 28-day mean x 30
    "active_last_28d": active,
})
n_total, n_excluded = len(risk), int((~active).sum())
risk = risk.loc[risk["active_last_28d"]].drop(columns="active_last_28d").copy()
risk["forecast_daily_std"] = risk["forecast_daily_std"].fillna(risk["forecast_daily_mean"] * 0.15)
risk["stockout_risk_units"] = (risk["forecast_30d_demand"] - risk["assumed_available_inventory"]).round(1)
risk["stockout_risk_pct"] = (risk["stockout_risk_units"] / risk["forecast_30d_demand"].replace(0, np.nan)).clip(-1, 1).fillna(0).round(4)
risk["recommended_safety_stock"] = (SERVICE_LEVEL_Z * risk["forecast_daily_std"] * np.sqrt(LEAD_TIME_DAYS)).round(1)
risk["recommended_order_qty"] = (risk["stockout_risk_units"] + risk["recommended_safety_stock"]).clip(lower=0).round(1)
risk["elevated_risk"] = ((risk["stockout_risk_units"] > 0) & (risk["stockout_risk_pct"] > ELEVATED_PCT)).astype(int)
risk = risk.sort_values("stockout_risk_units", ascending=False).reset_index(drop=True)
risk["risk_rank"] = np.arange(1, len(risk) + 1)

pos = risk["stockout_risk_units"].clip(lower=0)
cum = pos.cumsum() / pos.sum() * 100
n_pos = int((pos > 0).sum())
n80 = int(np.searchsorted(cum.to_numpy(), 80.0) + 1) if pos.sum() > 0 else 0
risk["cum_pct_of_positive_risk_units"] = cum.round(2)
risk.to_csv(os.path.join(OUT_SCEN, "stockout_risk_scores.csv"), index=False)

pareto = {
    "total_store_family_combos": n_total,
    "excluded_zero_sales_last_28d": n_excluded,
    "active_combos_scored": int(len(risk)),
    "positive_risk_combos": n_pos,
    "combos_at_stockout_risk": int(risk["elevated_risk"].sum()),
    "n_combos_driving_80pct_of_risk_units": n80,
    "pct_combos_driving_80pct_of_risk": round(n80 / n_pos * 100, 1) if n_pos else 0.0,   # denominator = positive-risk combos
    "pct_of_active_combos_driving_80pct": round(n80 / len(risk) * 100, 1),
    "total_positive_risk_units": round(float(pos.sum()), 1),
}
with open(os.path.join(OUT_SCEN, "stockout_pareto_summary.json"), "w") as f:
    json.dump(pareto, f, indent=2)

print("\nTop 10 by risk units:\n", risk.head(10)[["store_nbr", "family", "forecast_30d_demand", "assumed_available_inventory",
                                                   "stockout_risk_units", "stockout_risk_pct", "recommended_order_qty"]].to_string(index=False))
print("\nPareto summary:", pareto)
log("Scenario analysis complete. Artifacts -> output/scenarios/")
