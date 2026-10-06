from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4


@dataclass
class MarketBar:
    instrument_id: UUID
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    market_bar_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if self.open < 0:
            raise ValueError("Open price cannot be negative")
        if self.high < 0:
            raise ValueError("High price cannot be negative")
        if self.low < 0:
            raise ValueError("Low price cannot be negative")
        if self.close < 0:
            raise ValueError("Close price cannot be negative")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
        if self.high < self.low:
            raise ValueError("High price cannot be lower than low price")
        if not self.low <= self.open <= self.high:
            raise ValueError("Open price must be between low and high")
        if not self.low <= self.close <= self.high:
            raise ValueError("Close price must be between low and high")
