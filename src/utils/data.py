from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy import stats

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "commodities.yaml"

# Mapping from internal commodity name to yfinance ticker.
# Loaded from config where possible; fallback proxies for illiquid contracts.
SYMBOL_MAP: dict[str, str] = {
    "brent_crude": "BZ=F",
    "wti_crude": "CL=F",
    "natural_gas": "NG=F",
    "copper": "HG=F",
    "aluminum": "ALI=F",
    "zinc": "ZNC=F",
    "nickel": "NI=F",
    "iron_ore": "VALE",    # equity proxy - no liquid yfinance futures
    "thermal_coal": "BTU", # equity proxy
    "gold": "GC=F",
}


def _load_commodities_config() -> dict[str, Any]:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["commodities"]


def _resolve_ticker(commodity_name: str) -> str:
    """Resolve a commodity name to a yfinance ticker symbol."""
    key = commodity_name.lower().replace(" ", "_")
    if key in SYMBOL_MAP:
        return SYMBOL_MAP[key]
    # Try the config file for unknown commodities
    cfg = _load_commodities_config()
    if key in cfg:
        return cfg[key].get("yfinance_proxy", cfg[key]["symbol"])
    raise ValueError(f"Unknown commodity: {commodity_name}")


def get_historical_prices(
    symbol: str,
    days: int = 365,
    column: str = "Close",
) -> pd.Series:
    """Fetch historical daily closing prices from yfinance.

    Args:
        symbol: Commodity name (e.g. 'brent_crude') or raw yfinance ticker.
        days: Number of calendar days of history.
        column: OHLCV column to return.
    """
    ticker = _resolve_ticker(symbol) if "=" not in symbol and not symbol.isupper() else symbol
    period = f"{days}d" if days <= 730 else f"{days // 365}y"
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' (commodity: {symbol})")
    series = df[column].squeeze()
    series.name = symbol
    return series


def get_current_price(symbol: str) -> float:
    """Return the latest available closing price for a commodity."""
    series = get_historical_prices(symbol, days=5)
    return float(series.dropna().iloc[-1])


def compute_correlation_matrix(symbols: list[str], days: int = 365) -> pd.DataFrame:
    """Compute pairwise correlation matrix of daily returns.

    Args:
        symbols: List of commodity names or tickers.
        days: Lookback window in calendar days.
    """
    prices = {}
    for sym in symbols:
        try:
            prices[sym] = get_historical_prices(sym, days=days)
        except ValueError:
            continue

    if len(prices) < 2:
        raise ValueError("Need at least 2 valid symbols to compute correlations")

    price_df = pd.DataFrame(prices)
    returns = price_df.pct_change().dropna()
    return returns.corr()


def compute_var(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Compute Value at Risk for a return series.

    Args:
        returns: Daily returns (as decimals, not percentages).
        confidence: Confidence level (e.g. 0.95 for 95% VaR).
        method: 'historical' for empirical quantile, 'parametric' for normal assumption.

    Returns:
        VaR as a positive number representing the loss threshold.
    """
    r = np.asarray(returns)
    r = r[~np.isnan(r)]

    if len(r) < 10:
        raise ValueError("Insufficient data points for VaR calculation")

    alpha = 1 - confidence

    if method == "historical":
        var = -np.percentile(r, alpha * 100)
    elif method == "parametric":
        mu = np.mean(r)
        sigma = np.std(r, ddof=1)
        z = stats.norm.ppf(alpha)
        var = -(mu + z * sigma)
    else:
        raise ValueError(f"Unknown VaR method: {method}")

    return float(var)


def get_daily_returns(symbol: str, days: int = 365) -> pd.Series:
    """Fetch historical prices and return daily percentage returns."""
    prices = get_historical_prices(symbol, days=days)
    returns = prices.pct_change().dropna()
    returns.name = symbol
    return returns


def get_all_commodity_names() -> list[str]:
    cfg = _load_commodities_config()
    return list(cfg.keys())


def get_commodity_info(commodity_name: str) -> dict[str, Any]:
    cfg = _load_commodities_config()
    key = commodity_name.lower().replace(" ", "_")
    if key not in cfg:
        raise ValueError(f"Unknown commodity: {commodity_name}")
    return cfg[key]
