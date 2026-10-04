from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from app.core.exceptions import InstrumentNotFoundError
from app.domain.entities.fundamental_record import FundamentalRecord
from app.services.fundamentals_ingestion import (
    FundamentalsIngestionService,
)


@dataclass
class ServiceContext:
    service: FundamentalsIngestionService
    instrument_repository: MagicMock
    fundamental_record_repository: MagicMock
    fundamentals_provider: MagicMock


def create_context() -> ServiceContext:
    instrument_repository = MagicMock()
    fundamental_record_repository = MagicMock()
    fundamentals_provider = MagicMock()

    instrument_repository.get_by_id.return_value = MagicMock()

    service = FundamentalsIngestionService(
        instrument_repository=instrument_repository,
        fundamental_record_repository=fundamental_record_repository,
        fundamentals_provider=fundamentals_provider,
    )

    return ServiceContext(
        service=service,
        instrument_repository=instrument_repository,
        fundamental_record_repository=fundamental_record_repository,
        fundamentals_provider=fundamentals_provider,
    )


def create_record(
    instrument_id: UUID,
    metric_name: str = "revenue",
    value: str = "1000000",
) -> FundamentalRecord:
    return FundamentalRecord(
        instrument_id=instrument_id,
        period_end=date(2026, 3, 31),
        metric_name=metric_name,
        value=Decimal(value),
        fiscal_year=2026,
        fiscal_quarter=4,
        currency="INR",
    )


def test_ingest_new_records() -> None:
    context = create_context()
    instrument_id = uuid4()
    record = create_record(instrument_id)

    context.fundamentals_provider.get_fundamentals.return_value = [record]
    context.fundamental_record_repository.get_by_instrument_and_period.return_value = [] #type: ignore[assignment]

    result = context.service.ingest(
        instrument_id=instrument_id,
        start_period=date(2026, 1, 1),
        end_period=date(2026, 12, 31),
    )

    assert result.received_count == 1
    assert result.inserted_count == 1
    assert result.updated_count == 0
    assert result.skipped_count == 0

    context.fundamental_record_repository.save.assert_called_once_with(record)


def test_ingest_skips_unchanged_record() -> None:
    context = create_context()
    instrument_id = uuid4()

    existing = create_record(instrument_id)
    incoming = create_record(
        instrument_id,
        value="1000000",
    )

    context.fundamentals_provider.get_fundamentals.return_value = [incoming]
    context.fundamental_record_repository.get_by_instrument_and_period.return_value = [
        existing
    ]

    result = context.service.ingest(
        instrument_id=instrument_id,
        start_period=date(2026, 1, 1),
        end_period=date(2026, 12, 31),
    )

    assert result.received_count == 1
    assert result.inserted_count == 0
    assert result.updated_count == 0
    assert result.skipped_count == 1

    context.fundamental_record_repository.save.assert_not_called()


def test_ingest_updates_changed_record() -> None:
    context = create_context()
    instrument_id = uuid4()

    existing = create_record(
        instrument_id,
        value="900000",
    )
    incoming = create_record(
        instrument_id,
        value="1000000",
    )

    context.fundamentals_provider.get_fundamentals.return_value = [incoming]
    context.fundamental_record_repository.get_by_instrument_and_period.return_value = [
        existing
    ]

    result = context.service.ingest(
        instrument_id=instrument_id,
        start_period=date(2026, 1, 1),
        end_period=date(2026, 12, 31),
    )

    assert result.received_count == 1
    assert result.inserted_count == 0
    assert result.updated_count == 1
    assert result.skipped_count == 0

    context.fundamental_record_repository.save.assert_called_once_with(
        incoming
    )

    assert incoming.fundamental_record_id == existing.fundamental_record_id


def test_ingest_raises_when_instrument_does_not_exist() -> None:
    context = create_context()
    instrument_id = uuid4()

    context.instrument_repository.get_by_id.return_value = None

    with pytest.raises(InstrumentNotFoundError):
        context.service.ingest(
            instrument_id=instrument_id,
            start_period=date(2026, 1, 1),
            end_period=date(2026, 12, 31),
        )

    context.fundamentals_provider.get_fundamentals.assert_not_called()


def test_ingest_handles_multiple_records() -> None:
    context = create_context()
    instrument_id = uuid4()

    revenue = create_record(
        instrument_id,
        metric_name="revenue",
        value="1000000",
    )
    income = create_record(
        instrument_id,
        metric_name="net income",
        value="200000",
    )

    context.fundamentals_provider.get_fundamentals.return_value = [
        revenue,
        income,
    ]
    context.fundamental_record_repository.get_by_instrument_and_period.return_value = [] #type: ignore[assignment]

    result = context.service.ingest(
        instrument_id=instrument_id,
        start_period=date(2026, 1, 1),
        end_period=date(2026, 12, 31),
    )

    assert result.received_count == 2
    assert result.inserted_count == 2
    assert result.updated_count == 0
    assert result.skipped_count == 0

    assert context.fundamental_record_repository.save.call_count == 2
