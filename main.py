"""Single entry-point runner for end-to-end portfolio risk analysis."""
from __future__ import annotations

from pathlib import Path

from var_analyzer.data.portfolio_loader import load_portfolio_data
from var_analyzer.preprocessing.returns_engine import (
    calculate_log_returns,
    calculate_portfolio_returns,
    calculate_statistics,
)
from var_analyzer.reporting.excel_exporter import export_risk_report_to_excel
from var_analyzer.risk_analysis.risk_summary import generate_full_risk_report
from var_analyzer.risk_models.cvar import calculate_cvar
from var_analyzer.risk_models.historical_var import calculate_historical_var

print("hello")
def run_portfolio_risk_analysis(
    portfolio_value: float = 1_000_000.0,
    confidence_level: float = 0.95,
    output_file: str = "portfolio_risk_report.xlsx",
) -> None:
    """Run full pipeline and generate a local Excel risk report.

    Parameters
    ----------
    portfolio_value : float, optional
        Portfolio value used to convert risk percentages into monetary amounts.
    confidence_level : float, optional
        Confidence level for Historical VaR and CVaR.
    output_file : str, optional
        Output excel file name saved to local disk.
    """
    try:
        # 1) Load data
        prices, weights = load_portfolio_data()

        # 2) Preprocessing
        returns = calculate_log_returns(prices)
        portfolio_returns = calculate_portfolio_returns(returns, weights)
        stats = calculate_statistics(returns)

        # 3) Risk models
        var_value = calculate_historical_var(portfolio_returns, confidence_level=confidence_level)
        cvar_value = calculate_cvar(portfolio_returns, confidence_level=confidence_level)

        # 4) Risk analysis
        risk_report = generate_full_risk_report(
            portfolio_value=portfolio_value,
            weights=weights,
            portfolio_returns=portfolio_returns,
            returns=returns,
            covariance_matrix=stats["cov"],
            var_value=var_value,
            cvar_value=cvar_value,
        )

        # 5) CLI output
        print("\n=== Portfolio Risk Analysis Summary ===")
        print(f"VaR %: {risk_report['var_percent']:.6f}")
        print(f"CVaR %: {risk_report['cvar_percent']:.6f}")
        print(f"VaR Amount: {risk_report['var_amount']:.2f}")
        print(f"Risk Level: {risk_report['risk_level']}")
        print(f"Insight: {risk_report['insight']}")
        print(f"Recommendation: {risk_report['recommendation']}")

        # 6) Export report
        correlation_matrix = returns.corr()
        excel_bytes = export_risk_report_to_excel(
            file_name=output_file,
            risk_report=risk_report,
            portfolio_returns=portfolio_returns,
            returns=returns,
            correlation_matrix=correlation_matrix,
        )
        Path(output_file).write_bytes(excel_bytes)
        print(f"\nExcel report generated successfully: {Path(output_file).resolve()}")

    except Exception as exc:
        print("Portfolio risk analysis failed. Please check data availability and configuration.")
        print(f"Error details: {exc}")


if __name__ == "__main__":
    run_portfolio_risk_analysis()
