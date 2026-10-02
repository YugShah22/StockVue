from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from app.domain.entities.market_bar import MarketBar


@dataclass(frozen=True)
class MarketQuote:

    instrument_id: UUID
    timestamp: datetime
    price: float
    volume: float | None = None
    bid: float | None = None
    ask: float | None = None


class MarketDataProvider(Protocol):

    def get_bars(
        self,
        instrument_id: UUID,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> list[MarketBar]:
        ...

    def get_latest_bar(
        self,
        instrument_id: UUID,
        interval: str,
    ) -> MarketBar:
        ...

    def get_quote(
        self,
        instrument_id: UUID,
    ) -> MarketQuote:
        ...
