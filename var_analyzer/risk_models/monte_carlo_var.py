"""Monte Carlo Value at Risk (VaR) model implementation."""

from __future__ import annotations

import numpy as np

from var_analyzer.utils.exceptions import DataValidationError, ReturnCalculationError


def calculate_monte_carlo_var(
    mean_vector: np.ndarray,
    covariance_matrix: np.ndarray,
    weights: np.ndarray,
    num_simulations: int = 10000,
    confidence_level: float = 0.95,
) -> float:
    """Estimate VaR using Monte Carlo simulation of multivariate returns.

    Parameters
    ----------
    mean_vector : np.ndarray
        Mean returns vector for assets.
    covariance_matrix : np.ndarray
        Covariance matrix of asset returns.
    weights : np.ndarray
        Portfolio weights vector aligned with mean/covariance dimensions.
    num_simulations : int, optional
        Number of Monte Carlo simulation paths. Defaults to 10000.
    confidence_level : float, optional
        Confidence level for VaR calculation. Defaults to 0.95.

    Returns
    -------
    float
        Monte Carlo VaR as a negative number representing potential loss.

    Raises
    ------
    DataValidationError
        If inputs are empty or dimensionally incompatible.
    ReturnCalculationError
        If simulation or VaR computation fails.
    """
    if not (0 < confidence_level < 1):
        raise DataValidationError("confidence_level must be between 0 and 1.")
    if num_simulations <= 0:
        raise DataValidationError("num_simulations must be a positive integer.")

    mu = np.asarray(mean_vector, dtype=float)
    cov = np.asarray(covariance_matrix, dtype=float)
    w = np.asarray(weights, dtype=float)

    if mu.size == 0 or cov.size == 0 or w.size == 0:
        raise DataValidationError("mean_vector, covariance_matrix, and weights cannot be empty.")
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise DataValidationError("covariance_matrix must be a square 2D array.")
    if mu.ndim != 1 or w.ndim != 1:
        raise DataValidationError("mean_vector and weights must be 1D arrays.")
    if not (mu.shape[0] == cov.shape[0] == w.shape[0]):
        raise DataValidationError(
            "Dimension mismatch: mean_vector, covariance_matrix, and weights must align."
        )

    try:
        simulated_asset_returns = np.random.multivariate_normal(
            mean=mu,
            cov=cov,
            size=num_simulations,
        )
        simulated_portfolio_returns = np.dot(simulated_asset_returns, w)
        var_percentile = (1.0 - confidence_level) * 100.0
        var_value = float(np.percentile(simulated_portfolio_returns, var_percentile))
        return var_value if var_value <= 0 else -var_value
    except Exception as exc:
        raise ReturnCalculationError(f"Failed to compute Monte Carlo VaR: {exc}") from exc


