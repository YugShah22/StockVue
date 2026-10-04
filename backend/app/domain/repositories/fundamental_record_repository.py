from abc import ABC, abstractmethod
from datetime import date
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
    ) -> list[FundamentalRecord]:
        raise NotImplementedError

    @abstractmethod
    def save(
        self,
        record: FundamentalRecord,
    ) -> None:
        raise NotImplementedError
