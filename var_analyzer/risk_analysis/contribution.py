"""Risk contribution analysis helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from var_analyzer.utils.exceptions import DataValidationError, ReturnCalculationError
    

def calculate_risk_contribution(
    weights: Iterable[float],
    covariance_matrix: pd.DataFrame,
    asset_names: Sequence[str] | None = None,
) -> pd.Series:
    """Calculate each asset's contribution to portfolio variance.

    Parameters
    ----------
    weights : Iterable[float]
        Portfolio weights aligned to covariance matrix column order.
    covariance_matrix : pd.DataFrame
        Asset covariance matrix.
    asset_names : Sequence[str] | None, optional
        Optional explicit asset names. If omitted, covariance matrix columns
        are used.

    Returns
    -------
    pd.Series
        Risk contribution by asset as variance contributions (sum equals
        portfolio variance).

    Raises
    ------
    DataValidationError
        If inputs are empty or dimensionally incompatible.
    ReturnCalculationError
        If computation fails.
    """
    if covariance_matrix is None or covariance_matrix.empty:
        raise DataValidationError("Covariance matrix is empty. Cannot compute risk contribution.")

    cov = covariance_matrix.to_numpy(dtype=float)
    if cov.shape[0] != cov.shape[1]:
        raise DataValidationError("Covariance matrix must be square.")

    weight_array = np.asarray(list(weights), dtype=float)
    if weight_array.size == 0:
        raise DataValidationError("Weights cannot be empty.")
    if weight_array.shape[0] != cov.shape[0]:
        raise DataValidationError(
            "Weights length and covariance matrix dimensions do not match. "
            f"weights={weight_array.shape[0]}, cov={cov.shape}."
        )

    labels = list(asset_names) if asset_names is not None else list(covariance_matrix.columns)
    if len(labels) != weight_array.shape[0]:
        raise DataValidationError("asset_names length must match weights length.")

    try:
        marginal_contrib = np.dot(cov, weight_array)
        contribution = weight_array * marginal_contrib
        return pd.Series(contribution, index=labels, name="risk_contribution")
    except Exception as exc:
        raise ReturnCalculationError(f"Failed to compute risk contribution: {exc}") from exc


