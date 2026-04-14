"""Excel export utility for portfolio risk dashboard outputs."""

from __future__ import annotations

from io import BytesIO

import pandas as pd


def export_risk_report_to_excel(
    file_name: str,
    risk_report: dict,
    portfolio_returns: pd.Series,
    returns: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
) -> bytes:
    """Export risk analytics into a multi-sheet Excel report and return it as bytes.

    Parameters
    ----------
    file_name : str
        Logical file name for the report. Included for caller context.
    risk_report : dict
        Dictionary containing summary risk fields and insight text.
    portfolio_returns : pd.Series
        Portfolio return series.
    returns : pd.DataFrame
        Asset return dataframe.
    correlation_matrix : pd.DataFrame
        Asset correlation matrix.

    Returns
    -------
    bytes
        In-memory Excel file content as bytes.
    """
    _ = file_name  # Retained for API compatibility and future metadata usage.

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Portfolio Value",
                "VaR (%)",
                "CVaR (%)",
                "VaR Amount",
                "CVaR Amount",
                "Risk Level",
            ],
            "Value": [
                risk_report.get("portfolio_value"),
                risk_report.get("var_percent"),
                risk_report.get("cvar_percent"),
                risk_report.get("var_amount"),
                risk_report.get("cvar_amount"),
                risk_report.get("risk_level"),
            ],
        }
    )

    portfolio_returns_df = pd.DataFrame(
        {
            "Date": portfolio_returns.index,
            "Portfolio Return": portfolio_returns.values,
        }
    )

    asset_returns_df = returns.reset_index().rename(columns={returns.index.name or "index": "Date"})
    corr_df = correlation_matrix.reset_index().rename(
        columns={correlation_matrix.index.name or "index": "Asset"}
    )

    insights_df = pd.DataFrame(
        {
            "Field": [
                "Insight",
                "Recommendation",
                "Top Risk Contributor",
                "Highest Correlation Pair",
                "Lowest Correlation Pair",
                "Concentration Flag",
            ],
            "Value": [
                risk_report.get("insight"),
                risk_report.get("recommendation"),
                risk_report.get("top_risk_contributor"),
                str(risk_report.get("highest_correlation_pair")),
                str(risk_report.get("lowest_correlation_pair")),
                risk_report.get("concentration_flag"),
            ],
        }
    )

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        portfolio_returns_df.to_excel(writer, sheet_name="Portfolio Returns", index=False)
        asset_returns_df.to_excel(writer, sheet_name="Asset Returns", index=False)
        corr_df.to_excel(writer, sheet_name="Correlation Matrix", index=False)
        insights_df.to_excel(writer, sheet_name="Insights", index=False)

        workbook = writer.book
        header_format = workbook.add_format({"bold": True, "bg_color": "#DDEBF7", "border": 1})

        for sheet_name, dataframe in {
            "Summary": summary_df,
            "Portfolio Returns": portfolio_returns_df,
            "Asset Returns": asset_returns_df,
            "Correlation Matrix": corr_df,
            "Insights": insights_df,
        }.items():
            worksheet = writer.sheets[sheet_name]
            for col_num, value in enumerate(dataframe.columns.values):
                worksheet.write(0, col_num, value, header_format)
                column_width = max(14, len(str(value)) + 2)
                worksheet.set_column(col_num, col_num, column_width)

    return output.getvalue()
