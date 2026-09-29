"""
AI Multi-Factor Stock Intelligence & Portfolio Decision-Support System
backend/app/core/exceptions.py

Domain exception hierarchy.

Rules:
  - All application exceptions inherit from BaseDomainError.
  - Exceptions carry a machine-readable `code` so the API layer can map them
    to HTTP status codes without inspecting exception types directly.
  - Never raise bare Exception or generic ValueError from business logic.
  - The API layer catches these and converts to HTTP responses.
"""


class BaseDomainError(Exception):
    """Root of the exception hierarchy. All system errors inherit from this."""

    code: str = "DOMAIN_ERROR"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.code!r}, message={self.message!r})"


# ---------------------------------------------------------------------------
# Data layer errors
# ---------------------------------------------------------------------------


class DataProviderError(BaseDomainError):
    """Raised when an external data provider call fails or returns unexpected data."""

    code = "DATA_PROVIDER_ERROR"


class DataQualityError(BaseDomainError):
    """Raised when data fails a quality check and cannot be safely used."""

    code = "DATA_QUALITY_ERROR"


class DataNotFoundError(BaseDomainError):
    """Raised when requested data does not exist in the system."""

    code = "DATA_NOT_FOUND"


class DataStalenessError(BaseDomainError):
    """Raised when data exceeds its acceptable freshness threshold."""

    code = "DATA_STALE"


class ProviderAuthError(BaseDomainError):
    """Raised when a data provider rejects authentication."""

    code = "PROVIDER_AUTH_ERROR"


class ProviderRateLimitError(BaseDomainError):
    """Raised when a data provider returns a rate-limit response."""

    code = "PROVIDER_RATE_LIMIT"


# ---------------------------------------------------------------------------
# Domain / business logic errors
# ---------------------------------------------------------------------------


class InstrumentNotFoundError(BaseDomainError):
    """Raised when a requested instrument/symbol is not in the security master."""

    code = "INSTRUMENT_NOT_FOUND"


class FeatureComputationError(BaseDomainError):
    """Raised when a factor or feature cannot be computed (e.g. insufficient data)."""

    code = "FEATURE_COMPUTATION_ERROR"


class PredictionError(BaseDomainError):
    """Raised when the prediction engine cannot produce a valid output."""

    code = "PREDICTION_ERROR"


class RiskComputationError(BaseDomainError):
    """Raised when a risk metric cannot be calculated."""

    code = "RISK_COMPUTATION_ERROR"


class PortfolioError(BaseDomainError):
    """Raised when a portfolio operation violates constraints or is invalid."""

    code = "PORTFOLIO_ERROR"


class BacktestError(BaseDomainError):
    """Raised when a backtest configuration or execution is invalid."""

    code = "BACKTEST_ERROR"


class SignalError(BaseDomainError):
    """Raised when a signal lifecycle transition is invalid."""

    code = "SIGNAL_ERROR"


# ---------------------------------------------------------------------------
# Configuration / infrastructure errors
# ---------------------------------------------------------------------------


class ConfigurationError(BaseDomainError):
    """Raised when required configuration is missing or invalid at startup."""

    code = "CONFIGURATION_ERROR"


class RepositoryError(BaseDomainError):
    """Raised when a database repository operation fails unexpectedly."""

    code = "REPOSITORY_ERROR"
