"""Portfolio concentration analysis helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np

from var_analyzer.utils.exceptions import DataValidationError


def calculate_concentration_metrics(weights: Dict[str, float]) -> dict:
    """Calculate concentration metrics for a portfolio weight vector.

    Parameters
    ----------
    weights : Dict[str, float]
        Mapping of asset ticker to portfolio weight.

    Returns
    -------
    dict
        Concentration metrics including max weight and Herfindahl-Hirschman
        Index (HHI).

    Raises
    ------
    DataValidationError
        If weights are empty or contain invalid values.
    """
    if not weights:
        raise DataValidationError("Weights dictionary is empty. Cannot compute concentration metrics.")

    values = np.asarray(list(weights.values()), dtype=float)
    if values.size == 0:
        raise DataValidationError("Weights dictionary is empty. Cannot compute concentration metrics.")
    if np.any(values < 0):
        raise DataValidationError("Weights contain negative values.")

    total_weight = float(np.sum(values))
    if total_weight <= 0:
        raise DataValidationError("Total weights must be positive.")

    normalized = values / total_weight
    hhi = float(np.sum(np.square(normalized)))
    max_index = int(np.argmax(normalized))
    max_ticker = list(weights.keys())[max_index]
    max_weight = float(normalized[max_index])

    return {
        "hhi": hhi,
        "max_weight": max_weight,
        "max_weight_ticker": max_ticker,
        "effective_num_assets": float(1.0 / hhi) if hhi > 0 else float("inf"),
    }


