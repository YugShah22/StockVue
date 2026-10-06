from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.core.exceptions import DataProviderError, InstrumentNotFoundError
from app.domain.entities.ingestion_run import IngestionRun
from app.domain.entities.market_bar import MarketBar
from app.domain.providers.market_data import MarketDataProvider
from app.domain.repositories.ingestion_run_repository import (
    IngestionRunRepository,
)
from app.domain.repositories.instrument_repository import InstrumentRepository
from app.domain.repositories.market_bar_repository import MarketBarRepository
from app.domain.services.market_bar_validator import MarketBarValidator


@dataclass(frozen=True)
class PriceIngestionResult:
    instrument_id: UUID
    requested_start: datetime
    requested_end: datetime
    interval: str
    received_count: int
    inserted_count: int
    updated_count: int
    skipped_count: int


class PriceIngestionService:
    def __init__(
        self,
        instrument_repository: InstrumentRepository,
        market_bar_repository: MarketBarRepository,
        market_data_provider: MarketDataProvider,
        ingestion_run_repository: IngestionRunRepository | None = None,
    ) -> None:
        self._instrument_repository = instrument_repository
        self._market_bar_repository = market_bar_repository
        self._market_data_provider = market_data_provider
        self._ingestion_run_repository = ingestion_run_repository

    def ingest_bars(
        self,
        instrument_id: UUID,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> PriceIngestionResult:
        provider_name = getattr(
            self._market_data_provider,
            "name",
            self._market_data_provider.__class__.__name__,
        )
        started_at = datetime.now(UTC)
        run = IngestionRun(
            run_id=uuid4(),
            ingestion_type="price",
            instrument_id=instrument_id,
            provider=provider_name,
            started_at=started_at,
            status="RUNNING",
        )
        if self._ingestion_run_repository is not None:
            self._ingestion_run_repository.save(run)

        try:
            result = self._execute_ingest(
                instrument_id=instrument_id,
                start=start,
                end=end,
                interval=interval,
            )
            if self._ingestion_run_repository is not None:
                run.status = "SUCCESS"
                run.completed_at = datetime.now(UTC)
                run.received_count = result.received_count
                run.inserted_count = result.inserted_count
                run.updated_count = result.updated_count
                run.skipped_count = result.skipped_count
                self._ingestion_run_repository.save(run)
            return result
        except Exception as exc:
            if self._ingestion_run_repository is not None:
                run.status = "FAILED"
                run.completed_at = datetime.now(UTC)
                run.error_message = str(exc)
                self._ingestion_run_repository.save(run)
            raise

    def _execute_ingest(
        self,
        instrument_id: UUID,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> PriceIngestionResult:
        instrument = self._instrument_repository.get_by_id(instrument_id)

        if instrument is None:
            raise InstrumentNotFoundError(
                f"Instrument not found: {instrument_id}"
            )

        bars = self._market_data_provider.get_bars(
            instrument_id=instrument_id,
            start=start,
            end=end,
            interval=interval,
        )

        if not bars:
            return PriceIngestionResult(
                instrument_id=instrument_id,
                requested_start=start,
                requested_end=end,
                interval=interval,
                received_count=0,
                inserted_count=0,
                updated_count=0,
                skipped_count=0,
            )

        MarketBarValidator.validate_batch(
            bars=bars,
            instrument_id=instrument_id,
            start=start,
            end=end,
        )

        existing_bars = self._market_bar_repository.get_bars(
            instrument_id=instrument_id,
            start=start,
            end=end,
        )

        existing_by_ts: dict[datetime, MarketBar] = {
            bar.timestamp: bar for bar in existing_bars
        }

        inserted_count = 0
        updated_count = 0
        skipped_count = 0
        bars_to_upsert: list[MarketBar] = []

        for bar in bars:
            existing = existing_by_ts.get(bar.timestamp)

            if existing is None:
                inserted_count += 1
                bars_to_upsert.append(bar)
                existing_by_ts[bar.timestamp] = bar
                continue

            if self._bars_are_equal(existing, bar):
                skipped_count += 1
                continue

            updated_count += 1
            bars_to_upsert.append(bar)
            existing_by_ts[bar.timestamp] = bar

        if bars_to_upsert:
            self._market_bar_repository.upsert_bars(bars_to_upsert)

        return PriceIngestionResult(
            instrument_id=instrument_id,
            requested_start=start,
            requested_end=end,
            interval=interval,
            received_count=len(bars),
            inserted_count=inserted_count,
            updated_count=updated_count,
            skipped_count=skipped_count,
        )

    @staticmethod
    def _validate_bar(
        bar: MarketBar,
        instrument_id: UUID,
        start: datetime,
        end: datetime,
    ) -> None:
        if bar.instrument_id != instrument_id:
            raise DataProviderError(
                "Provider returned a market bar for the wrong instrument"
            )

        if bar.timestamp.tzinfo is None:
            raise DataProviderError(
                "Provider returned a market bar with a naive timestamp"
            )

        if bar.timestamp < start or bar.timestamp > end:
            raise DataProviderError(
                "Provider returned a market bar outside the requested range"
            )

    @staticmethod
    def _bars_are_equal(existing: MarketBar, incoming: MarketBar) -> bool:
        return (
            existing.instrument_id == incoming.instrument_id
            and existing.timestamp == incoming.timestamp
            and existing.open == incoming.open
            and existing.high == incoming.high
            and existing.low == incoming.low
            and existing.close == incoming.close
            and existing.volume == incoming.volume
        )
