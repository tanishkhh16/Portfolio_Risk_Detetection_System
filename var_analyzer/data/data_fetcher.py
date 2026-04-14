"""Data fetching orchestration and cleaning utilities for price data."""

from __future__ import annotations

import logging
import re
from typing import Iterable

import pandas as pd

from var_analyzer.data.market_data_client import fetch_adjusted_close_prices
from var_analyzer.utils.exceptions import EmptyDataError, MarketDataError

LOGGER = logging.getLogger(__name__)


def _resolve_period_range(period: str = "2y") -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convert a compact period string into start and end dates.

    Parameters
    ----------
    period : str, optional
        Rolling lookback period in format ``<number><unit>`` where unit can be
        ``d`` (days), ``m`` (months), or ``y`` (years). Defaults to ``"2y"``.

    Returns
    -------
    tuple[pd.Timestamp, pd.Timestamp]
        Start and end timestamps to be used for data fetching.

    Raises
    ------
    MarketDataError
        If the period format is invalid.
    """
    match = re.fullmatch(r"(\d+)([dmy])", period.strip().lower())
    if not match:
        raise MarketDataError(
            "Invalid period format. Use values like '30d', '6m', or '2y'."
        )

    amount = int(match.group(1))
    unit = match.group(2)
    end_date = pd.Timestamp.today().normalize()

    if unit == "d":
        start_date = end_date - pd.DateOffset(days=amount)
    elif unit == "m":
        start_date = end_date - pd.DateOffset(months=amount)
    else:
        start_date = end_date - pd.DateOffset(years=amount)

    return start_date, end_date


def fetch_price_data(tickers: Iterable[str], period: str = "2y") -> pd.DataFrame:
    """Fetch and clean adjusted close price data for the given tickers.

    Parameters
    ----------
    tickers : Iterable[str]
        List or iterable of ticker symbols.
    period : str, optional
        Rolling lookback period (for example ``"2y"``). Defaults to ``"2y"``.

    Returns
    -------
    pd.DataFrame
        Cleaned adjusted close price data indexed by date and sorted
        chronologically.

    Raises
    ------
    EmptyDataError
        If no usable data remains after cleaning.
    MarketDataError
        If data fetching or period parsing fails.
    """
    start_date, end_date = _resolve_period_range(period)
    prices = fetch_adjusted_close_prices(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )

    # Ensure chronological order and clean sparse gaps.
    prices = prices.sort_index()
    prices = prices.ffill().dropna(how="any")

    if prices.empty:
        raise EmptyDataError(
            "Price data is empty after forward-fill and NaN drop operations."
        )

    LOGGER.info(
        "Prepared cleaned adjusted close data with %s rows and %s columns.",
        prices.shape[0],
        prices.shape[1],
    )
    return prices
