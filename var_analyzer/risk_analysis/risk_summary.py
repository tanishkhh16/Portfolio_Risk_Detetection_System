"""Unified risk insight engine combining VaR, correlation, contribution, and concentration."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from var_analyzer.risk_analysis.concentration import calculate_concentration_metrics
from var_analyzer.risk_analysis.contribution import calculate_risk_contribution
from var_analyzer.risk_analysis.correlation import calculate_correlation_matrix
from var_analyzer.utils.exceptions import DataValidationError


def _extract_extreme_correlation_pairs(corr_matrix: pd.DataFrame) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """Extract highest and lowest correlation asset pairs from correlation matrix."""
    if corr_matrix.shape[0] < 2:
        raise DataValidationError("At least two assets are required to extract correlation pairs.")

    mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    pair_values = corr_matrix.where(mask).stack()
    if pair_values.empty:
        raise DataValidationError("Unable to derive pair correlations from matrix.")

    highest_pair = tuple(pair_values.idxmax())
    lowest_pair = tuple(pair_values.idxmin())
    return highest_pair, lowest_pair


def generate_full_risk_report(
    portfolio_value: float,
    weights: dict,
    portfolio_returns: pd.Series,
    returns: pd.DataFrame,
    covariance_matrix: pd.DataFrame,
    var_value: float,
    cvar_value: float,
) -> dict:
    """Generate a full risk insight report for portfolio decision-making.

    Parameters
    ----------
    portfolio_value : float
        Current monetary value of the portfolio.
    weights : dict
        Asset weights mapping (ticker -> weight).
    portfolio_returns : pd.Series
        Historical portfolio returns.
    returns : pd.DataFrame
        Historical asset returns.
    covariance_matrix : pd.DataFrame
        Asset return covariance matrix.
    var_value : float
        Portfolio VaR in return terms (negative expected loss).
    cvar_value : float
        Portfolio CVaR in return terms (negative tail expected loss).

    Returns
    -------
    dict
        Unified risk report including risk metrics, diagnostics, insight text,
        and actionable recommendation.
    """
    if portfolio_value <= 0:
        raise DataValidationError("portfolio_value must be positive.")
    if not weights:
        raise DataValidationError("weights cannot be empty.")
    if portfolio_returns is None or portfolio_returns.empty:
        raise DataValidationError("portfolio_returns cannot be empty.")
    if returns is None or returns.empty:
        raise DataValidationError("returns cannot be empty.")
    if covariance_matrix is None or covariance_matrix.empty:
        raise DataValidationError("covariance_matrix cannot be empty.")
    if list(covariance_matrix.columns) != list(covariance_matrix.index):
        raise DataValidationError("covariance_matrix must have matching index/column asset labels.")

    # STEP 1: Convert VaR/CVaR to monetary amounts.
    var_amount = abs(float(var_value) * float(portfolio_value))
    cvar_amount = abs(float(cvar_value) * float(portfolio_value))

    # STEP 2: Correlation insights.
    corr_matrix = calculate_correlation_matrix(returns)
    highest_pair, lowest_pair = _extract_extreme_correlation_pairs(corr_matrix)
    highest_corr_value = float(corr_matrix.loc[highest_pair[0], highest_pair[1]])

    # STEP 3: Risk contribution.
    ordered_assets = list(covariance_matrix.columns)
    missing_weight_assets = [asset for asset in ordered_assets if asset not in weights]
    if missing_weight_assets:
        raise DataValidationError(
            f"weights missing entries for covariance assets: {missing_weight_assets}"
        )
    weight_vector = np.array([weights[a] for a in ordered_assets], dtype=float)
    risk_contribution = calculate_risk_contribution(
        weights=weight_vector,
        covariance_matrix=covariance_matrix,
        asset_names=ordered_assets,
    )
    top_risk_contributor = str(risk_contribution.idxmax())

    # STEP 4: Concentration metrics.
    concentration_metrics = calculate_concentration_metrics(weights)
    concentration_flag = bool(concentration_metrics["max_weight"] > 0.40)

    # STEP 5: Risk level classification based on absolute VaR percent.
    var_percent_abs = abs(float(var_value))
    if var_percent_abs < 0.01:
        risk_level = "Low"
    elif var_percent_abs <= 0.02:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    # STEP 6 + 7: Insight and recommendation generation.
    if concentration_flag:
        insight = "High concentration detected in one asset, increasing portfolio-specific risk."
        recommendation = "Reduce concentration in top asset and rebalance across additional holdings."
    elif highest_corr_value > 0.80:
        insight = "Assets are highly correlated, reducing diversification benefit."
        recommendation = "Add lower or negatively correlated assets to improve diversification."
    elif risk_level == "High":
        insight = "Portfolio downside risk is high relative to current VaR threshold."
        recommendation = "Reduce high-risk exposures and review hedging opportunities."
    else:
        insight = "Portfolio is reasonably diversified with manageable risk characteristics."
        recommendation = "Portfolio structure is balanced; continue regular risk monitoring."

    return {
        "portfolio_value": float(portfolio_value),
        "var_percent": float(var_value),
        "cvar_percent": float(cvar_value),
        "var_amount": float(var_amount),
        "cvar_amount": float(cvar_amount),
        "risk_level": risk_level,
        "top_risk_contributor": top_risk_contributor,
        "highest_correlation_pair": highest_pair,
        "lowest_correlation_pair": lowest_pair,
        "concentration_flag": concentration_flag,
        "insight": insight,
        "recommendation": recommendation,
    }
