"""Portfolio loading and validation utilities for the VaR Analyzer."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from var_analyzer.data.data_fetcher import fetch_price_data
from var_analyzer.utils.exceptions import PortfolioValidationError

DEFAULT_PORTFOLIO_WEIGHTS: Dict[str, float] = {
    "AAPL": 0.25,
    "MSFT": 0.25,
    "GOOGL": 0.25,
    "JPM": 0.25,
}


def validate_portfolio_weights(
    weights: Dict[str, float],
    tolerance: float = 1e-6,
) -> None:
    """Validate portfolio weights for non-negativity and unit sum.

    Parameters
    ----------
    weights : Dict[str, float]
        Mapping of ticker symbol to portfolio weight.
    tolerance : float, optional
        Absolute tolerance for validating that total weights sum to 1.0.

    Returns
    -------
    None
        This function returns nothing and raises on validation failure.

    Raises
    ------
    PortfolioValidationError
        If weights are empty, contain negatives, or do not sum to 1.0.
    """
    if not weights:
        raise PortfolioValidationError("Portfolio weights cannot be empty.")

    negative_weights = {ticker: w for ticker, w in weights.items() if w < 0}
    if negative_weights:
        raise PortfolioValidationError(
            f"Negative weights are not allowed: {negative_weights}"
        )

    total_weight = float(sum(weights.values()))
    if abs(total_weight - 1.0) > tolerance:
        raise PortfolioValidationError(
            f"Portfolio weights must sum to 1.0 within tolerance {tolerance}. "
            f"Current sum: {total_weight:.10f}"
        )


def load_portfolio_data() -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Load default portfolio, validate weights, and fetch historical prices.

    Parameters
    ----------
    None
        This function uses the default portfolio configuration defined in
        ``DEFAULT_PORTFOLIO_WEIGHTS``.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        A tuple containing:
        1) Clean adjusted close price DataFrame for default portfolio tickers
        2) Portfolio weight dictionary

    Raises
    ------
    PortfolioValidationError
        If default weights fail validation checks.
    """
    weights = DEFAULT_PORTFOLIO_WEIGHTS.copy()
    validate_portfolio_weights(weights=weights, tolerance=1e-6)

    tickers = list(weights.keys())
    prices = fetch_price_data(tickers=tickers, period="2y")
    return prices, weights
