from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pandas as pd
import pytest

from app.core.exceptions import (
    DataProviderError,
    InstrumentNotFoundError,
    ProviderRateLimitError,
)
from app.infrastructure.providers.development.yfinance_market_data import (
    YFinanceMarketDataProvider,
)


def create_instrument(exchange_code: str = "NSE") -> MagicMock:
    instrument = MagicMock()
    instrument.symbol = "RELIANCE"
    instrument.exchange_code = exchange_code
    return instrument


def create_provider(
    exchange_code: str = "NSE",
) -> tuple[YFinanceMarketDataProvider, MagicMock, UUID]:
    repository = MagicMock()
    instrument_id = uuid4()

    repository.get_by_id.return_value = create_instrument(exchange_code=exchange_code)

    provider = YFinanceMarketDataProvider(repository)

    return provider, repository, instrument_id


@patch("app.infrastructure.providers.development.yfinance_market_data.yf.Ticker")
def test_get_bars_normalizes_yfinance_data(mock_ticker: MagicMock) -> None:
    provider, repository, instrument_id = create_provider()

    index = pd.DatetimeIndex(
        [
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
        ]
    )

    data = pd.DataFrame(
        {
            "Open": [100.0, 105.0],
            "High": [110.0, 115.0],
            "Low": [95.0, 100.0],
            "Close": [105.0, 110.0],
            "Volume": [1000.0, 1200.0],
        },
        index=index,
    )

    mock_ticker.return_value.history.return_value = data

    bars = provider.get_bars(
        instrument_id,
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 3, tzinfo=UTC),
        "1d",
    )

    assert len(bars) == 2
    assert isinstance(bars[0].close, Decimal)
    assert isinstance(bars[0].volume, Decimal)
    assert bars[0].close == Decimal("105.0")
    assert bars[1].close == Decimal("110.0")

    mock_ticker.assert_called_once_with("RELIANCE.NS")
    repository.get_by_id.assert_called_once_with(instrument_id)


@patch("app.infrastructure.providers.development.yfinance_market_data.yf.Ticker")
def test_bse_symbol_mapping(mock_ticker: MagicMock) -> None:
    provider, repository, instrument_id = create_provider(exchange_code="BSE")

    mock_ticker.return_value.history.return_value = pd.DataFrame()

    bars = provider.get_bars(
        instrument_id,
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 3, tzinfo=UTC),
        "1d",
    )

    assert bars == []
    mock_ticker.assert_called_once_with("RELIANCE.BO")
    repository.get_by_id.assert_called_once_with(instrument_id)


def test_unsupported_exchange_raises_data_provider_error() -> None:
    provider, _, instrument_id = create_provider(exchange_code="NASDAQ")

    with pytest.raises(DataProviderError, match="yfinance symbol mapping is not supported"):
        provider.get_latest_bar(instrument_id, "1d")


@patch("app.infrastructure.providers.development.yfinance_market_data.yf.Ticker")
def test_get_latest_bar_returns_latest_observation(mock_ticker: MagicMock) -> None:
    provider, _, instrument_id = create_provider()

    index = pd.DatetimeIndex(
        [
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
        ]
    )

    data = pd.DataFrame(
        {
            "Open": [100.0, 105.0],
            "High": [110.0, 115.0],
            "Low": [95.0, 100.0],
            "Close": [105.0, 110.0],
            "Volume": [1000.0, 1200.0],
        },
        index=index,
    )

    mock_ticker.return_value.history.return_value = data

    bar = provider.get_latest_bar(instrument_id, "1d")

    assert isinstance(bar.close, Decimal)
    assert isinstance(bar.volume, Decimal)
    assert bar.close == Decimal("110.0")
    assert bar.volume == Decimal("1200.0")

@patch("app.infrastructure.providers.development.yfinance_market_data.yf.Ticker")
def test_get_latest_bar_raises_when_no_data(mock_ticker: MagicMock) -> None:
    provider, _, instrument_id = create_provider()

    mock_ticker.return_value.history.return_value = pd.DataFrame()

    with pytest.raises(DataProviderError, match="No market data available"):
        provider.get_latest_bar(instrument_id, "1d")


@patch("app.infrastructure.providers.development.yfinance_market_data.yf.Ticker")
def test_get_quote_returns_latest_price(mock_ticker: MagicMock) -> None:
    provider, _, instrument_id = create_provider()

    mock_ticker.return_value.fast_info.last_price = 1500.0

    quote = provider.get_quote(instrument_id)

    assert quote.instrument_id == instrument_id
    assert quote.price == 1500.0
    assert quote.timestamp.tzinfo is not None


@patch("app.infrastructure.providers.development.yfinance_market_data.yf.Ticker")
def test_get_bars_returns_empty_list_when_provider_has_no_data(
    mock_ticker: MagicMock,
) -> None:
    provider, _, instrument_id = create_provider()

    mock_ticker.return_value.history.return_value = pd.DataFrame()

    bars = provider.get_bars(
        instrument_id,
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 3, tzinfo=UTC),
        "1d",
    )

    assert bars == []


@patch("app.infrastructure.providers.development.yfinance_market_data.yf.Ticker")
def test_get_bars_translates_provider_failure(
    mock_ticker: MagicMock,
) -> None:
    provider, _, instrument_id = create_provider()

    mock_ticker.return_value.history.side_effect = Exception(
        "Yahoo request failed"
    )

    with pytest.raises(DataProviderError):
        provider.get_bars(
            instrument_id,
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 3, tzinfo=UTC),
            "1d",
        )


def test_provider_rejects_unknown_instrument() -> None:
    repository = MagicMock()
    repository.get_by_id.return_value = None

    provider = YFinanceMarketDataProvider(repository)

    instrument_id = uuid4()

    with pytest.raises(InstrumentNotFoundError):
        provider.get_latest_bar(instrument_id, "1d")


@patch("app.infrastructure.providers.development.yfinance_market_data.yf.Ticker")
def test_get_bars_translates_rate_limit_error(
    mock_ticker: MagicMock,
) -> None:
    provider, _, instrument_id = create_provider()

    mock_ticker.return_value.history.side_effect = Exception(
        "429 rate limit exceeded"
    )

    with pytest.raises(ProviderRateLimitError):
        provider.get_bars(
            instrument_id,
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 3, tzinfo=UTC),
            "1d",
        )


@patch("app.infrastructure.providers.development.yfinance_market_data.yf.Ticker")
def test_generic_rate_message_does_not_trigger_rate_limit_error(
    mock_ticker: MagicMock,
) -> None:
    provider, _, instrument_id = create_provider()

    mock_ticker.return_value.history.side_effect = Exception(
        "failed to parse exchange rate"
    )

    with pytest.raises(DataProviderError) as exc_info:
        provider.get_bars(
            instrument_id,
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 3, tzinfo=UTC),
            "1d",
        )

    assert not isinstance(exc_info.value, ProviderRateLimitError)


@patch("app.infrastructure.providers.development.yfinance_market_data.yf.Ticker")
def test_too_many_requests_triggers_rate_limit_error(
    mock_ticker: MagicMock,
) -> None:
    provider, _, instrument_id = create_provider()

    mock_ticker.return_value.history.side_effect = Exception(
        "Client error: Too Many Requests"
    )

    with pytest.raises(ProviderRateLimitError):
        provider.get_bars(
            instrument_id,
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 3, tzinfo=UTC),
            "1d",
        )
