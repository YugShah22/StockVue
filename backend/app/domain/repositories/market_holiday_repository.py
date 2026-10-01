from abc import ABC, abstractmethod
from datetime import date
from uuid import UUID

from app.domain.entities.market_holiday import MarketHoliday


class MarketHolidayRepository(ABC):
    @abstractmethod
    def get_by_exchange(
        self,
        exchange_id: UUID,
        year: int | None = None,
    ) -> list[MarketHoliday]:
        raise NotImplementedError

    @abstractmethod
    def get_by_date(
        self,
        exchange_id: UUID,
        holiday_date: date,
    ) -> MarketHoliday | None:
        raise NotImplementedError

    @abstractmethod
    def save(self, holiday: MarketHoliday) -> None:
        raise NotImplementedError
