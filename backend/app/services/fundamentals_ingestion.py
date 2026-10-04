from dataclasses import dataclass
from datetime import date
from uuid import UUID

from app.core.exceptions import InstrumentNotFoundError
from app.domain.entities.fundamental_record import FundamentalRecord
from app.domain.providers.fundamentals import FundamentalsProvider
from app.domain.repositories.fundamental_record_repository import (
    FundamentalRecordRepository,
)
from app.domain.repositories.instrument_repository import InstrumentRepository


@dataclass(frozen=True)
class FundamentalsIngestionResult:
    instrument_id: UUID
    requested_start: date
    requested_end: date
    received_count: int
    inserted_count: int
    updated_count: int
    skipped_count: int


class FundamentalsIngestionService:
    def __init__(
        self,
        instrument_repository: InstrumentRepository,
        fundamental_record_repository: FundamentalRecordRepository,
        fundamentals_provider: FundamentalsProvider,
    ) -> None:
        self._instrument_repository = instrument_repository
        self._fundamental_record_repository = fundamental_record_repository
        self._fundamentals_provider = fundamentals_provider

    def ingest(
        self,
        instrument_id: UUID,
        start_period: date,
        end_period: date,
    ) -> FundamentalsIngestionResult:
        instrument = self._instrument_repository.get_by_id(instrument_id)

        if instrument is None:
            raise InstrumentNotFoundError(
                f"Instrument not found: {instrument_id}"
            )

        records = self._fundamentals_provider.get_fundamentals(
            instrument_id=instrument_id,
            start_period=start_period,
            end_period=end_period,
        )

        inserted_count = 0
        updated_count = 0
        skipped_count = 0

        for record in records:
            existing_records = (
                self._fundamental_record_repository
                .get_by_instrument_and_period(
                    instrument_id=instrument_id,
                    period_end=record.period_end,
                )
            )

            existing = next(
                (
                    item
                    for item in existing_records
                    if item.metric_name == record.metric_name
                    and item.fiscal_year == record.fiscal_year
                    and item.fiscal_quarter == record.fiscal_quarter
                ),
                None,
            )

            if existing is None:
                self._fundamental_record_repository.save(record)
                inserted_count += 1
                continue

            if self._records_are_equal(existing, record):
                skipped_count += 1
                continue

            record.fundamental_record_id = existing.fundamental_record_id
            self._fundamental_record_repository.save(record)
            updated_count += 1

        return FundamentalsIngestionResult(
            instrument_id=instrument_id,
            requested_start=start_period,
            requested_end=end_period,
            received_count=len(records),
            inserted_count=inserted_count,
            updated_count=updated_count,
            skipped_count=skipped_count,
        )

    @staticmethod
    def _records_are_equal(
        existing: FundamentalRecord,
        incoming: FundamentalRecord,
    ) -> bool:
        return (
            existing.instrument_id == incoming.instrument_id
            and existing.period_end == incoming.period_end
            and existing.metric_name == incoming.metric_name
            and existing.value == incoming.value
            and existing.fiscal_year == incoming.fiscal_year
            and existing.fiscal_quarter == incoming.fiscal_quarter
            and existing.currency == incoming.currency
            and existing.published_at == incoming.published_at
            and existing.available_at == incoming.available_at
        )
