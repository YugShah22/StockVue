from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from app.core.exceptions import DataProviderError, InstrumentNotFoundError
from app.domain.entities.market_bar import MarketBar
from app.domain.providers.market_data import MarketDataProvider
from app.domain.repositories.instrument_repository import InstrumentRepository
from app.domain.repositories.market_bar_repository import MarketBarRepository


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
    ) -> None:
        self._instrument_repository = instrument_repository
        self._market_bar_repository = market_bar_repository
        self._market_data_provider = market_data_provider

    def ingest_bars(
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

        inserted_count = 0
        updated_count = 0
        skipped_count = 0

        for bar in bars:
            self._validate_bar(bar, instrument_id, start, end)

            existing = (
                self._market_bar_repository.get_by_instrument_and_timestamp(
                    instrument_id=instrument_id,
                    timestamp=bar.timestamp,
                )
            )

            if existing is None:
                self._market_bar_repository.save(bar)
                inserted_count += 1
                continue

            if self._bars_are_equal(existing, bar):
                skipped_count += 1
                continue

            self._market_bar_repository.save(bar)
            updated_count += 1

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
