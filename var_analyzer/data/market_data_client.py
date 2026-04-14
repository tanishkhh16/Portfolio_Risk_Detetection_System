"""Market data client for fetching adjusted close prices via yfinance."""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd
import yfinance as yf

from var_analyzer.utils.exceptions import InvalidTickerError, MarketDataError

LOGGER = logging.getLogger(__name__)


def fetch_adjusted_close_prices(
    tickers: Iterable[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch adjusted close prices for the provided tickers and date range.

    Parameters
    ----------
    tickers : Iterable[str]
        Iterable of ticker symbols to fetch.
    start_date : pd.Timestamp
        Inclusive start date for historical data retrieval.
    end_date : pd.Timestamp
        Inclusive end date for historical data retrieval.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with one column per ticker containing
        adjusted close prices.

    Raises
    ------
    InvalidTickerError
        If one or more provided tickers are invalid or return no data.
    MarketDataError
        If yfinance fails after one retry.
    """
    ticker_list = [t.strip().upper() for t in tickers if str(t).strip()]
    if not ticker_list:
        raise InvalidTickerError("No valid ticker symbols provided.")

    attempts = 2
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            raw = yf.download(
                tickers=ticker_list,
                start=pd.Timestamp(start_date).date(),
                end=pd.Timestamp(end_date).date(),
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=False,
            )
            break
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            last_error = exc
            LOGGER.warning(
                "yfinance fetch attempt %s/%s failed: %s",
                attempt,
                attempts,
                exc,
            )
    else:
        raise MarketDataError(
            "Unable to fetch market data after retry."
        ) from last_error

    if raw is None or raw.empty:
        raise InvalidTickerError(
            f"No data returned for requested tickers: {ticker_list}"
        )

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" not in raw.columns.get_level_values(0):
            raise MarketDataError("Adjusted close data ('Adj Close') not available.")
        adj_close = raw["Adj Close"].copy()
    else:
        # Single-ticker response can be flattened in some yfinance versions.
        if "Adj Close" in raw.columns:
            adj_close = raw[["Adj Close"]].copy()
            adj_close.columns = [ticker_list[0]]
        else:
            raise MarketDataError("Adjusted close data ('Adj Close') not available.")

    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame(name=ticker_list[0])

    # Ensure all requested tickers are represented as columns.
    adj_close = adj_close.reindex(columns=ticker_list)

    invalid_tickers = [
        ticker for ticker in ticker_list if ticker not in adj_close.columns or adj_close[ticker].dropna().empty
    ]
    if invalid_tickers:
        raise InvalidTickerError(
            f"Invalid or non-tradable tickers with no data: {invalid_tickers}"
        )

    LOGGER.info(
        "Fetched adjusted close data for %s tickers with %s rows.",
        len(ticker_list),
        len(adj_close),
    )
    return adj_close
