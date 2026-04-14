"""Correlation analysis helpers for portfolio risk insights."""

from __future__ import annotations

import pandas as pd

from var_analyzer.utils.exceptions import DataValidationError, ReturnCalculationError


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute the asset return correlation matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return DataFrame with date index and asset columns.

    Returns
    -------
    pd.DataFrame
        Correlation matrix between assets.

    Raises
    ------
    DataValidationError
        If returns input is empty or has fewer than two columns.
    ReturnCalculationError
        If correlation calculation fails.
    """
    if returns is None or returns.empty:
        raise DataValidationError("Returns DataFrame is empty. Cannot compute correlation matrix.")
    if returns.shape[1] < 2:
        raise DataValidationError("At least two assets are required to compute correlations.")

    try:
        corr_matrix = returns.corr()
    except Exception as exc:
        raise ReturnCalculationError(f"Failed to compute correlation matrix: {exc}") from exc

    if corr_matrix.empty:
        raise ReturnCalculationError("Computed correlation matrix is empty.")

    return corr_matrix


