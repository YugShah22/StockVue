from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from uuid import UUID

from app.domain.entities.fundamental_record import FundamentalRecord
from app.domain.entities.market_bar import MarketBar


@dataclass(frozen=True)
class FeatureCalculationContext:
    instrument_id: UUID
    as_of_date: date
    market_bars: Sequence[MarketBar] = ()
    fundamentals: Sequence[FundamentalRecord] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.instrument_id, UUID):
            raise TypeError(
                f"instrument_id must be a UUID instance, got {type(self.instrument_id).__name__}"
            )

        if isinstance(self.as_of_date, datetime):
            raise TypeError("as_of_date must be a date, not a datetime")

        if not isinstance(self.as_of_date, date):
            raise TypeError(
                f"as_of_date must be a date instance, got {type(self.as_of_date).__name__}"
            )

        object.__setattr__(self, "market_bars", tuple(self.market_bars))
        object.__setattr__(self, "fundamentals", tuple(self.fundamentals))
