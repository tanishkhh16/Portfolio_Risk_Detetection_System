"""Conditional Value at Risk (CVaR) model implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from var_analyzer.utils.exceptions import DataValidationError, ReturnCalculationError


def calculate_cvar(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """Calculate CVaR (Expected Shortfall) from historical portfolio returns.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Historical portfolio return series.
    confidence_level : float, optional
        Confidence level for CVaR calculation (e.g., 0.95). Defaults to 0.95.

    Returns
    -------
    float
        CVaR as a negative number representing average tail loss.

    Raises
    ------
    DataValidationError
        If input returns are empty or confidence level is invalid.
    ReturnCalculationError
        If CVaR calculation fails.
    """
    if portfolio_returns is None or portfolio_returns.empty:
        raise DataValidationError("Portfolio returns are empty. Cannot compute CVaR.")
    if not (0 < confidence_level < 1):
        raise DataValidationError("confidence_level must be between 0 and 1.")

    try:
        clean_returns = portfolio_returns.dropna().to_numpy(dtype=float)
        if clean_returns.size == 0:
            raise DataValidationError("Portfolio returns contain only NaN values.")

        var_percentile = (1.0 - confidence_level) * 100.0
        var_threshold = float(np.percentile(clean_returns, var_percentile))

        tail_losses = clean_returns[clean_returns <= var_threshold]
        if tail_losses.size == 0:
            raise ReturnCalculationError("No tail losses found beyond VaR threshold.")

        cvar_value = float(np.mean(tail_losses))
        return cvar_value if cvar_value <= 0 else -cvar_value
    except (DataValidationError, ReturnCalculationError):
        raise
    except Exception as exc:
        raise ReturnCalculationError(f"Failed to compute CVaR: {exc}") from exc


