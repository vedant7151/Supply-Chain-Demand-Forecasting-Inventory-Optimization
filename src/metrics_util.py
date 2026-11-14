"""Shared forecast-accuracy metrics (used by 03 and 04)."""
import numpy as np


def _a(x):
    return np.asarray(x, dtype="float64")


def mape(y, p):
    """MAPE over rows with actual != 0 (zero actuals are excluded - see zero_share)."""
    y, p = _a(y), _a(p)
    m = y != 0
    return float(np.mean(np.abs(y[m] - p[m]) / y[m]) * 100) if m.any() else float("nan")


def wape(y, p):
    """Weighted APE = sum|e| / sum(y). Volume-weighted, defined with zeros."""
    y, p = _a(y), _a(p)
    return float(np.abs(y - p).sum() / y.sum() * 100) if y.sum() else float("nan")


def smape(y, p):
    """sMAPE in % (0..200); rows where actual and forecast are both 0 count as 0 error."""
    y, p = _a(y), _a(p)
    d = np.abs(y) + np.abs(p)
    r = np.where(d == 0, 0.0, 2 * np.abs(y - p) / np.where(d == 0, 1, d))
    return float(r.mean() * 100)


def rmse(y, p):
    y, p = _a(y), _a(p)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    y, p = _a(y), _a(p)
    return float(np.mean(np.abs(y - p)))


def bias_pct(y, p):
    y, p = _a(y), _a(p)
    return float((p.sum() - y.sum()) / y.sum() * 100) if y.sum() else float("nan")


def zero_share(y):
    return float((_a(y) == 0).mean() * 100)


def mase(y, p, train_y, m=7):
    """MASE for a single series: MAE / in-sample MAE of the seasonal-naive (lag m) forecast."""
    train_y = _a(train_y)
    scale = np.mean(np.abs(train_y[m:] - train_y[:-m]))
    return float(mae(y, p) / scale) if scale > 0 else float("nan")


def all_metrics(y, p):
    return {
        "MAPE": mape(y, p), "WAPE": wape(y, p), "sMAPE": smape(y, p),
        "RMSE": rmse(y, p), "MAE": mae(y, p), "Bias_pct": bias_pct(y, p),
    }
