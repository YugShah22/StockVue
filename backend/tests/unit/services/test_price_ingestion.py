from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from app.core.exceptions import DataProviderError, InstrumentNotFoundError
from app.domain.entities.market_bar import MarketBar
from app.services.price_ingestion import (
    PriceIngestionResult,
    PriceIngestionService,
)


@dataclass
class ServiceContext:
    service: PriceIngestionService
    instrument_repository: MagicMock
    market_bar_repository: MagicMock
    market_data_provider: MagicMock


def create_bar(
    instrument_id: UUID,
    timestamp: datetime,
    *,
    open_price: float = 100.0,
    high: float = 110.0,
    low: float = 90.0,
    close: float = 105.0,
    volume: float = 1000.0,
) -> MarketBar:
    return MarketBar(
        instrument_id=instrument_id,
        timestamp=timestamp,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


@pytest.fixture
def instrument_id() -> UUID:
    return uuid4()


@pytest.fixture
def start() -> datetime:
    return datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture
def end() -> datetime:
    return datetime(2026, 9, 5, tzinfo=UTC)


@pytest.fixture
def service_context() -> ServiceContext:
    instrument_repository = MagicMock()
    market_bar_repository = MagicMock()
    market_data_provider = MagicMock()

    service = PriceIngestionService(
        instrument_repository=instrument_repository,
        market_bar_repository=market_bar_repository,
        market_data_provider=market_data_provider,
    )

    return ServiceContext(
        service=service,
        instrument_repository=instrument_repository,
        market_bar_repository=market_bar_repository,
        market_data_provider=market_data_provider,
    )


def test_ingest_new_bars(
    service_context: ServiceContext,
    instrument_id: UUID,
    start: datetime,
    end: datetime,
) -> None:
    service = service_context.service
    instrument_repository = service_context.instrument_repository
    market_bar_repository = service_context.market_bar_repository
    market_data_provider = service_context.market_data_provider

    instrument_repository.get_by_id.return_value = MagicMock()

    bars = [
        create_bar(
            instrument_id,
            datetime(2026, 9, 1, tzinfo=UTC),
        ),
        create_bar(
            instrument_id,
            datetime(2026, 9, 2, tzinfo=UTC),
        ),
    ]

    market_data_provider.get_bars.return_value = bars
    market_bar_repository.get_by_instrument_and_timestamp.return_value = None

    result = service.ingest_bars(
        instrument_id,
        start,
        end,
        "1d",
    )

    assert isinstance(result, PriceIngestionResult)
    assert result.received_count == 2
    assert result.inserted_count == 2
    assert result.updated_count == 0
    assert result.skipped_count == 0

    assert market_bar_repository.save.call_count == 2


def test_missing_instrument_raises_error(
    service_context: ServiceContext,
    instrument_id: UUID,
    start: datetime,
    end: datetime,
) -> None:
    service = service_context.service
    instrument_repository = service_context.instrument_repository
    market_data_provider = service_context.market_data_provider

    instrument_repository.get_by_id.return_value = None

    with pytest.raises(InstrumentNotFoundError):
        service.ingest_bars(
            instrument_id,
            start,
            end,
            "1d",
        )

    market_data_provider.get_bars.assert_not_called()


def test_empty_provider_response(
    service_context: ServiceContext,
    instrument_id: UUID,
    start: datetime,
    end: datetime,
) -> None:
    service = service_context.service
    instrument_repository = service_context.instrument_repository
    market_bar_repository = service_context.market_bar_repository
    market_data_provider = service_context.market_data_provider

    instrument_repository.get_by_id.return_value = MagicMock()
    market_data_provider.get_bars.return_value = []  # type: ignore[assignment]

    result = service.ingest_bars(
        instrument_id,
        start,
        end,
        "1d",
    )

    assert result.received_count == 0
    assert result.inserted_count == 0
    assert result.updated_count == 0
    assert result.skipped_count == 0

    market_bar_repository.save.assert_not_called()


def test_existing_identical_bar_is_skipped(
    service_context: ServiceContext,
    instrument_id: UUID,
    start: datetime,
    end: datetime,
) -> None:
    service = service_context.service
    instrument_repository = service_context.instrument_repository
    market_bar_repository = service_context.market_bar_repository
    market_data_provider = service_context.market_data_provider

    instrument_repository.get_by_id.return_value = MagicMock()

    timestamp = datetime(2026, 9, 1, tzinfo=UTC)

    incoming = create_bar(instrument_id, timestamp)
    existing = create_bar(instrument_id, timestamp)

    market_data_provider.get_bars.return_value = [incoming]
    market_bar_repository.get_by_instrument_and_timestamp.return_value = existing

    result = service.ingest_bars(
        instrument_id,
        start,
        end,
        "1d",
    )

    assert result.received_count == 1
    assert result.inserted_count == 0
    assert result.updated_count == 0
    assert result.skipped_count == 1

    market_bar_repository.save.assert_not_called()


def test_existing_different_bar_is_updated(
    service_context: ServiceContext,
    instrument_id: UUID,
    start: datetime,
    end: datetime,
) -> None:
    service = service_context.service
    instrument_repository = service_context.instrument_repository
    market_bar_repository = service_context.market_bar_repository
    market_data_provider = service_context.market_data_provider

    instrument_repository.get_by_id.return_value = MagicMock()

    timestamp = datetime(2026, 9, 1, tzinfo=UTC)

    existing = create_bar(
        instrument_id,
        timestamp,
        close=100.0,
    )

    incoming = create_bar(
        instrument_id,
        timestamp,
        close=105.0,
    )

    market_data_provider.get_bars.return_value = [incoming]
    market_bar_repository.get_by_instrument_and_timestamp.return_value = existing

    result = service.ingest_bars(
        instrument_id,
        start,
        end,
        "1d",
    )

    assert result.received_count == 1
    assert result.inserted_count == 0
    assert result.updated_count == 1
    assert result.skipped_count == 0

    market_bar_repository.save.assert_called_once_with(incoming)


def test_provider_error_propagates(
    service_context: ServiceContext,
    instrument_id: UUID,
    start: datetime,
    end: datetime,
) -> None:
    service = service_context.service
    instrument_repository = service_context.instrument_repository
    market_data_provider = service_context.market_data_provider

    instrument_repository.get_by_id.return_value = MagicMock()

    error = DataProviderError("provider failed")
    market_data_provider.get_bars.side_effect = error

    with pytest.raises(DataProviderError, match="provider failed"):
        service.ingest_bars(
            instrument_id,
            start,
            end,
            "1d",
        )


def test_wrong_instrument_bar_is_rejected(
    service_context: ServiceContext,
    instrument_id: UUID,
    start: datetime,
    end: datetime,
) -> None:
    service = service_context.service
    instrument_repository = service_context.instrument_repository
    market_bar_repository = service_context.market_bar_repository
    market_data_provider = service_context.market_data_provider

    instrument_repository.get_by_id.return_value = MagicMock()

    wrong_instrument_id = uuid4()

    bar = create_bar(
        wrong_instrument_id,
        datetime(2026, 9, 1, tzinfo=UTC),
    )

    market_data_provider.get_bars.return_value = [bar]

    with pytest.raises(
        DataProviderError,
        match="wrong instrument",
    ):
        service.ingest_bars(
            instrument_id,
            start,
            end,
            "1d",
        )

    market_bar_repository.save.assert_not_called()


def test_bar_outside_requested_range_is_rejected(
    service_context: ServiceContext,
    instrument_id: UUID,
    start: datetime,
    end: datetime,
) -> None:
    service = service_context.service
    instrument_repository = service_context.instrument_repository
    market_bar_repository = service_context.market_bar_repository
    market_data_provider = service_context.market_data_provider

    instrument_repository.get_by_id.return_value = MagicMock()

    bar = create_bar(
        instrument_id,
        datetime(2026, 9, 10, tzinfo=UTC),
    )

    market_data_provider.get_bars.return_value = [bar]

    with pytest.raises(
        DataProviderError,
        match="outside the requested range",
    ):
        service.ingest_bars(
            instrument_id,
            start,
            end,
            "1d",
        )

    market_bar_repository.save.assert_not_called()


def test_naive_timestamp_is_rejected(
    service_context: ServiceContext,
    instrument_id: UUID,
    start: datetime,
    end: datetime,
) -> None:
    service = service_context.service
    instrument_repository = service_context.instrument_repository
    market_bar_repository = service_context.market_bar_repository
    market_data_provider = service_context.market_data_provider

    instrument_repository.get_by_id.return_value = MagicMock()

    bar = create_bar(
        instrument_id,
        datetime(2026, 9, 1),
    )

    market_data_provider.get_bars.return_value = [bar]

    with pytest.raises(
        DataProviderError,
        match="naive timestamp",
    ):
        service.ingest_bars(
            instrument_id,
            start,
            end,
            "1d",
        )

    market_bar_repository.save.assert_not_called()
