from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from app.domain.entities.market_bar import MarketBar


class MarketBarRepository(ABC):
    @abstractmethod
    def get_by_instrument_and_timestamp(
        self,
        instrument_id: UUID,
        timestamp: datetime,
    ) -> MarketBar | None:
        raise NotImplementedError

    @abstractmethod
    def get_bars(
        self,
        instrument_id: UUID,
        start: datetime,
        end: datetime,
    ) -> list[MarketBar]:
        raise NotImplementedError

    @abstractmethod
    def get_latest_bar(
        self,
        instrument_id: UUID,
    ) -> MarketBar | None:
        raise NotImplementedError

    @abstractmethod
    def save(self, market_bar: MarketBar) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert_bars(self, bars: Sequence[MarketBar]) -> None:
        raise NotImplementedError


