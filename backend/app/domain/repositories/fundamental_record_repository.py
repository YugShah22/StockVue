from abc import ABC, abstractmethod
from datetime import date, datetime
from uuid import UUID

from app.domain.entities.fundamental_record import FundamentalRecord


class FundamentalRecordRepository(ABC):
    @abstractmethod
    def get_by_id(
        self,
        fundamental_record_id: UUID,
    ) -> FundamentalRecord | None:
        raise NotImplementedError

    @abstractmethod
    def get_by_instrument_and_period(
        self,
        instrument_id: UUID,
        period_end: date,
        as_of: datetime | None = None,
    ) -> list[FundamentalRecord]:
        raise NotImplementedError

    @abstractmethod
    def get_as_of(
        self,
        instrument_id: UUID,
        as_of: datetime,
        start_period: date | None = None,
        end_period: date | None = None,
        metric_name: str | None = None,
    ) -> list[FundamentalRecord]:
        """
        Retrieve fundamental records for an instrument that were available
        at or before the given as_of timestamp (point-in-time / no look-ahead bias).

        Only records where available_at is not None and available_at <= as_of
        are returned.
        """
        raise NotImplementedError

    @abstractmethod
    def save(
        self,
        record: FundamentalRecord,
    ) -> None:
        raise NotImplementedError
