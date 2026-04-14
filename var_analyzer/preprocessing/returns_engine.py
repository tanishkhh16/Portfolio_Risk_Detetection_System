"""Return calculation utilities for portfolio risk modeling."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from var_analyzer.utils.exceptions import DataValidationError, ReturnCalculationError


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from price data.

    Parameters
    ----------
    prices : pd.DataFrame
        Price DataFrame indexed by date with asset tickers as columns.

    Returns
    -------
    pd.DataFrame
        Log return DataFrame with the same columns as input prices.

    Raises
    ------
    DataValidationError
        If the input price DataFrame is empty.
    ReturnCalculationError
        If return calculation fails.
    """
    if prices is None or prices.empty:
        raise DataValidationError("Price DataFrame is empty. Cannot compute returns.")

    try:
        log_returns = np.log(prices / prices.shift(1)).dropna(how="any")
    except Exception as exc:
        raise ReturnCalculationError(
            f"Failed to calculate log returns: {exc}"
        ) from exc

    if log_returns.empty:
        raise ReturnCalculationError("Log returns are empty after dropping NaN rows.")

    return log_returns


def calculate_portfolio_returns(
    returns: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """Compute weighted portfolio return series using asset returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return DataFrame indexed by date with ticker columns.
    weights : Dict[str, float]
        Mapping of ticker to portfolio weight.

    Returns
    -------
    pd.Series
        Portfolio return series indexed by date.

    Raises
    ------
    DataValidationError
        If returns/weights are empty or if weight keys do not align with columns.
    ReturnCalculationError
        If portfolio return computation fails.
    """
    if returns is None or returns.empty:
        raise DataValidationError("Returns DataFrame is empty. Cannot compute portfolio returns.")
    if not weights:
        raise DataValidationError("Weights dictionary is empty.")

    return_columns = list(returns.columns)
    weight_keys = list(weights.keys())

    missing_weights = [col for col in return_columns if col not in weight_keys]
    extra_weights = [ticker for ticker in weight_keys if ticker not in return_columns]
    if missing_weights or extra_weights:
        raise DataValidationError(
            "Weights and returns columns are not aligned. "
            f"Missing weights for: {missing_weights}; "
            f"Extra weights for: {extra_weights}."
        )

    try:
        ordered_weights = np.array([weights[col] for col in return_columns], dtype=float)
        portfolio_returns = returns.dot(ordered_weights)
        portfolio_returns.name = "portfolio_return"
        return portfolio_returns
    except Exception as exc:
        raise ReturnCalculationError(
            f"Failed to calculate portfolio returns: {exc}"
        ) from exc


def calculate_statistics(returns: pd.DataFrame) -> dict:
    """Calculate key return statistics used by risk models.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return DataFrame indexed by date with ticker columns.

    Returns
    -------
    dict
        Dictionary containing mean return vector, covariance matrix,
        and per-asset standard deviation:
        {"mean": ..., "cov": ..., "std": ...}

    Raises
    ------
    DataValidationError
        If returns input is empty.
    ReturnCalculationError
        If statistics computation fails.
    """
    if returns is None or returns.empty:
        raise DataValidationError("Returns DataFrame is empty. Cannot compute statistics.")

    try:
        stats = {
            "mean": returns.mean(),
            "cov": returns.cov(),
            "std": returns.std(),
        }
    except Exception as exc:
        raise ReturnCalculationError(
            f"Failed to calculate return statistics: {exc}"
        ) from exc

    return stats


def calculate_portfolio_volatility(
    weights: np.ndarray,
    covariance_matrix: pd.DataFrame,
) -> float:
    """Calculate portfolio volatility using covariance and weights.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights vector.
    covariance_matrix : pd.DataFrame
        Asset return covariance matrix.

    Returns
    -------
    float
        Scalar portfolio volatility computed as ``sqrt(w.T * Cov * w)``.

    Raises
    ------
    DataValidationError
        If weights/covariance inputs are empty or dimensionally incompatible.
    ReturnCalculationError
        If volatility computation fails.
    """
    if covariance_matrix is None or covariance_matrix.empty:
        raise DataValidationError("Covariance matrix is empty. Cannot compute volatility.")

    weights_array = np.asarray(weights, dtype=float)
    if weights_array.size == 0:
        raise DataValidationError("Weights array is empty. Cannot compute volatility.")

    cov_values = covariance_matrix.to_numpy(dtype=float)
    if cov_values.shape[0] != cov_values.shape[1]:
        raise DataValidationError("Covariance matrix must be square.")
    if cov_values.shape[0] != weights_array.shape[0]:
        raise DataValidationError(
            "Weights length and covariance matrix dimensions do not match. "
            f"weights={weights_array.shape[0]}, cov={cov_values.shape}"
        )

    try:
        variance = float(np.dot(weights_array.T, np.dot(cov_values, weights_array)))
        variance = max(variance, 0.0)
        return float(np.sqrt(variance))
    except Exception as exc:
        raise ReturnCalculationError(
            f"Failed to calculate portfolio volatility: {exc}"
        ) from exc
