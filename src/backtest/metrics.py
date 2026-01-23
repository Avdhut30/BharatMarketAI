# src/backtest/metrics.py

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def sharpe(daily_returns: pd.Series, rf_daily: float = 0.0) -> float:
    r = daily_returns.dropna()
    if r.std() == 0 or len(r) < 2:
        return 0.0
    return float((r.mean() - rf_daily) / r.std() * np.sqrt(252))


def cagr(equity: pd.Series) -> float:
    eq = equity.dropna()
    if len(eq) < 2:
        return 0.0
    start = eq.iloc[0]
    end = eq.iloc[-1]
    days = (eq.index[-1] - eq.index[0]).days
    if days <= 0 or start <= 0:
        return 0.0
    years = days / 365.25
    return float((end / start) ** (1 / years) - 1.0)


def summary_stats(equity: pd.Series) -> dict:
    daily_ret = equity.pct_change()
    return {
        "CAGR": cagr(equity),
        "Sharpe": sharpe(daily_ret),
        "MaxDrawdown": max_drawdown(equity),
        "TotalReturn": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0,
        "Days": int((equity.index[-1] - equity.index[0]).days) if len(equity) > 1 else 0
    }
