"""Parametric (variance-covariance) Value at Risk (VaR) model."""

from __future__ import annotations

import math

from var_analyzer.utils.exceptions import DataValidationError, ReturnCalculationError

_SUPPORTED_Z_SCORES = {
    0.95: 1.645,
    0.99: 2.326,
}


def calculate_parametric_var(
    mean_return: float,
    std_dev: float,
    confidence_level: float = 0.95,
    time_horizon: int = 1,
) -> float:
    """Calculate parametric VaR under normal distribution assumptions.

    Parameters
    ----------
    mean_return : float
        Mean portfolio return per base period.
    std_dev : float
        Portfolio return standard deviation per base period.
    confidence_level : float, optional
        Confidence level for VaR, supported: 0.95 and 0.99. Defaults to 0.95.
    time_horizon : int, optional
        Number of periods for VaR scaling. Defaults to 1.

    Returns
    -------
    float
        Parametric VaR as a negative number representing potential loss.

    Raises
    ------
    DataValidationError
        If inputs are invalid.
    ReturnCalculationError
        If VaR calculation fails.
    """
    if confidence_level not in _SUPPORTED_Z_SCORES:
        raise DataValidationError(
            "Unsupported confidence_level. Use 0.95 or 0.99 for parametric VaR."
        )
    if std_dev < 0:
        raise DataValidationError("std_dev must be non-negative.")
    if time_horizon <= 0:
        raise DataValidationError("time_horizon must be a positive integer.")

    try:
        z_score = _SUPPORTED_Z_SCORES[confidence_level]
        var_value = (float(mean_return) - z_score * float(std_dev)) * math.sqrt(time_horizon)
        return var_value if var_value <= 0 else -var_value
    except Exception as exc:
        raise ReturnCalculationError(f"Failed to compute parametric VaR: {exc}") from exc


