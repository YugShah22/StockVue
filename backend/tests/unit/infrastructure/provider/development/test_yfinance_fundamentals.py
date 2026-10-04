from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pandas as pd
import pytest

from app.core.exceptions import DataProviderError
from app.domain.entities.fundamental_record import FundamentalRecord
from app.infrastructure.providers.development.yfinance_fundamentals import (
    YFinanceFundamentalsProvider,
)


def create_instrument_repository() -> MagicMock:
    repository = MagicMock()
    repository.get_by_id.return_value = MagicMock(
        symbol="RELIANCE",
        exchange_code="NSE",
    )
    return repository


def test_get_fundamentals_returns_records() -> None:
    instrument_id = uuid4()
    provider = YFinanceFundamentalsProvider(
        create_instrument_repository()
    )

    income_statement = pd.DataFrame(
        {
            pd.Timestamp("2026-03-31"): {
                "Total Revenue": 1000000,
                "Net Income": 200000,
            },
            pd.Timestamp("2025-03-31"): {
                "Total Revenue": 900000,
                "Net Income": 150000,
            },
        }
    )

    mock_ticker = MagicMock()
    mock_ticker.get_income_stmt.return_value = income_statement

    with patch(
        "app.infrastructure.providers.development.yfinance_fundamentals.yf.Ticker",
        return_value=mock_ticker,
    ) as mock_ticker_constructor:
        records = provider.get_fundamentals(
            instrument_id=instrument_id,
            start_period=date(2025, 1, 1),
            end_period=date(2026, 12, 31),
        )

    mock_ticker_constructor.assert_called_once_with("RELIANCE.NS")
    assert len(records) == 4

    revenue = next(
        record
        for record in records
        if record.metric_name == "total revenue"
        and record.period_end == date(2026, 3, 31)
    )

    assert isinstance(revenue, FundamentalRecord)
    assert revenue.instrument_id == instrument_id
    assert revenue.value == Decimal("1000000")


def test_get_fundamentals_filters_period_range() -> None:
    instrument_id = uuid4()
    provider = YFinanceFundamentalsProvider(
        create_instrument_repository()
    )

    income_statement = pd.DataFrame(
        {
            pd.Timestamp("2026-03-31"): {
                "Total Revenue": 1000000,
            },
            pd.Timestamp("2025-03-31"): {
                "Total Revenue": 900000,
            },
        }
    )

    mock_ticker = MagicMock()
    mock_ticker.get_income_stmt.return_value = income_statement

    with patch(
        "app.infrastructure.providers.development.yfinance_fundamentals.yf.Ticker",
        return_value=mock_ticker,
    ):
        records = provider.get_fundamentals(
            instrument_id=instrument_id,
            start_period=date(2026, 1, 1),
            end_period=date(2026, 12, 31),
        )

    assert len(records) == 1
    assert records[0].period_end == date(2026, 3, 31)


def test_empty_income_statement_returns_empty_list() -> None:
    instrument_id = uuid4()
    provider = YFinanceFundamentalsProvider(
        create_instrument_repository()
    )

    mock_ticker = MagicMock()
    mock_ticker.get_income_stmt.return_value = pd.DataFrame()

    with patch(
        "app.infrastructure.providers.development.yfinance_fundamentals.yf.Ticker",
        return_value=mock_ticker,
    ):
        records = provider.get_fundamentals(
            instrument_id=instrument_id,
            start_period=date(2025, 1, 1),
            end_period=date(2026, 12, 31),
        )

    assert records == []


def test_missing_instrument_raises_error() -> None:
    repository = MagicMock()
    repository.get_by_id.return_value = None

    provider = YFinanceFundamentalsProvider(repository)

    with pytest.raises(DataProviderError):
        provider.get_fundamentals(
            instrument_id=uuid4(),
            start_period=date(2025, 1, 1),
            end_period=date(2026, 12, 31),
        )


def test_unsupported_exchange_raises_error() -> None:
    repository = MagicMock()
    repository.get_by_id.return_value = MagicMock(
        symbol="RELIANCE",
        exchange_code="NYSE",
    )

    provider = YFinanceFundamentalsProvider(repository)

    with pytest.raises(DataProviderError):
        provider.get_fundamentals(
            instrument_id=uuid4(),
            start_period=date(2025, 1, 1),
            end_period=date(2026, 12, 31),
        )


def test_bse_symbol_is_resolved() -> None:
    instrument_id = uuid4()

    repository = MagicMock()
    repository.get_by_id.return_value = MagicMock(
        symbol="RELIANCE",
        exchange_code="BSE",
    )

    provider = YFinanceFundamentalsProvider(repository)

    mock_ticker = MagicMock()
    mock_ticker.get_income_stmt.return_value = pd.DataFrame()

    with patch(
        "app.infrastructure.providers.development.yfinance_fundamentals.yf.Ticker",
        return_value=mock_ticker,
    ) as mock_ticker_constructor:
        provider.get_fundamentals(
            instrument_id=instrument_id,
            start_period=date(2025, 1, 1),
            end_period=date(2026, 12, 31),
        )

    mock_ticker_constructor.assert_called_once_with("RELIANCE.BO")


def test_provider_failure_is_wrapped() -> None:
    instrument_id = uuid4()
    provider = YFinanceFundamentalsProvider(
        create_instrument_repository()
    )

    with patch(
        "app.infrastructure.providers.development.yfinance_fundamentals.yf.Ticker",
        side_effect=RuntimeError("Yahoo unavailable"),
    ):
        with pytest.raises(DataProviderError):
            provider.get_fundamentals(
                instrument_id=instrument_id,
                start_period=date(2025, 1, 1),
                end_period=date(2026, 12, 31),
            )
