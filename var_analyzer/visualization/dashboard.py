"""Streamlit dashboard for portfolio risk monitoring and insights."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

try:
    from var_analyzer.data.portfolio_loader import load_portfolio_data
    from var_analyzer.preprocessing.returns_engine import (
        calculate_log_returns,
        calculate_portfolio_returns,
        calculate_statistics,
    )
    from var_analyzer.reporting.excel_exporter import export_risk_report_to_excel
    from var_analyzer.risk_models.cvar import calculate_cvar
    from var_analyzer.risk_models.historical_var import calculate_historical_var
    from var_analyzer.risk_analysis.risk_summary import generate_full_risk_report
except ImportError:
    # Fallback when running from inside the var_analyzer folder.
    from data.portfolio_loader import load_portfolio_data
    from preprocessing.returns_engine import (
        calculate_log_returns,
        calculate_portfolio_returns,
        calculate_statistics,
    )
    from reporting.excel_exporter import export_risk_report_to_excel
    from risk_models.cvar import calculate_cvar
    from risk_models.historical_var import calculate_historical_var
    from risk_analysis.risk_summary import generate_full_risk_report


def main() -> None:
    """Run the Portfolio Risk Dashboard Streamlit application."""
    st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")
    st.title("Portfolio Risk Dashboard")

    st.sidebar.header("Dashboard Controls")
    portfolio_value = st.sidebar.number_input(
        "Portfolio Value (₹)",
        min_value=1_000.0,
        value=1_000_000.0,
        step=50_000.0,
        format="%.2f",
    )
    confidence_level = st.sidebar.selectbox(
        "Confidence Level",
        options=[0.95, 0.99],
        index=0,
    )

    st.header("Risk Computation")
    try:
        prices, weights = load_portfolio_data()
        if prices is None or prices.empty:
            st.error("No price data available. Please verify data source and portfolio settings.")
            return

        returns = calculate_log_returns(prices)
        portfolio_returns = calculate_portfolio_returns(returns, weights)
        stats = calculate_statistics(returns)

        var_value = calculate_historical_var(portfolio_returns, confidence_level=confidence_level)
        cvar_value = calculate_cvar(portfolio_returns, confidence_level=confidence_level)

        risk_report = generate_full_risk_report(
            portfolio_value=portfolio_value,
            weights=weights,
            portfolio_returns=portfolio_returns,
            returns=returns,
            covariance_matrix=stats["cov"],
            var_value=var_value,
            cvar_value=cvar_value,
        )
    except Exception as exc:
        st.error(f"Unable to build dashboard due to data/model error: {exc}")
        return

    st.subheader("Key Risk Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("VaR (%)", f"{risk_report['var_percent'] * 100:.2f}%")
    col2.metric("CVaR (%)", f"{risk_report['cvar_percent'] * 100:.2f}%")
    col3.metric("VaR (₹)", f"₹{risk_report['var_amount']:,.2f}")
    col4.metric("Risk Level", str(risk_report["risk_level"]))

    st.header("Portfolio Returns")
    st.subheader("Daily Portfolio Return Series")
    st.line_chart(portfolio_returns)

    st.header("Correlation Analysis")
    st.subheader("Asset Correlation Heatmap")
    corr_matrix = returns.corr()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

    st.header("Risk Summary Insights")
    st.markdown(f"**Insight:** {risk_report['insight']}")
    st.markdown(f"**Recommendation:** {risk_report['recommendation']}")

    st.subheader("Additional Diagnostics")
    st.write(
        {
            "Top Risk Contributor": risk_report["top_risk_contributor"],
            "Highest Correlation Pair": risk_report["highest_correlation_pair"],
            "Lowest Correlation Pair": risk_report["lowest_correlation_pair"],
            "Concentration Flag": risk_report["concentration_flag"],
        }
    )

    st.header("Report Export")
    try:
        excel_data = export_risk_report_to_excel(
            file_name="risk_report.xlsx",
            risk_report=risk_report,
            portfolio_returns=portfolio_returns,
            returns=returns,
            correlation_matrix=returns.corr(),
        )
        st.download_button(
            label="📥 Download Excel Report",
            data=excel_data,
            file_name="portfolio_risk_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as exc:
        st.warning(f"Excel export could not be generated: {exc}")


if __name__ == "__main__":
    main()
