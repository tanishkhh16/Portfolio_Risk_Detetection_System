"""Custom exception hierarchy for the VaR Analyzer project."""


class VarAnalyzerError(Exception):
    """Base exception for all project-specific errors."""


class MarketDataError(VarAnalyzerError):
    """Raised when market data cannot be fetched or processed."""


class InvalidTickerError(MarketDataError):
    """Raised when one or more provided tickers are invalid."""


class EmptyDataError(MarketDataError):
    """Raised when returned market data is empty after cleaning."""


class PortfolioValidationError(VarAnalyzerError):
    """Raised when portfolio weights fail validation checks."""


class DataValidationError(VarAnalyzerError):
    """Raised when input market/return data fails validation checks."""


class ReturnCalculationError(VarAnalyzerError):
    """Raised when return or portfolio risk calculations fail."""
