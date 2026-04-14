"""Historical Value at Risk (VaR) model implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from var_analyzer.utils.exceptions import DataValidationError, ReturnCalculationError


def calculate_historical_var(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """Calculate historical VaR from portfolio return distribution.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Historical portfolio return series.
    confidence_level : float, optional
        Confidence level for VaR calculation (e.g., 0.95). Defaults to 0.95.

    Returns
    -------
    float
        Historical VaR as a negative number representing potential loss.

    Raises
    ------
    DataValidationError
        If input returns are empty or confidence level is invalid.
    ReturnCalculationError
        If VaR calculation fails.
    """
    if portfolio_returns is None or portfolio_returns.empty:
        raise DataValidationError("Portfolio returns are empty. Cannot compute historical VaR.")
    if not (0 < confidence_level < 1):
        raise DataValidationError("confidence_level must be between 0 and 1.")

    try:
        var_percentile = (1.0 - confidence_level) * 100.0
        var_value = float(np.percentile(portfolio_returns.dropna().to_numpy(), var_percentile))
        return var_value if var_value <= 0 else -var_value
    except Exception as exc:
        raise ReturnCalculationError(f"Failed to compute historical VaR: {exc}") from exc


