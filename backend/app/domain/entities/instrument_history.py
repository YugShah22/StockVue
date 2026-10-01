from dataclasses import dataclass
from datetime import date
from uuid import UUID


@dataclass
class InstrumentHistory:
    history_id: UUID
    instrument_id: UUID
    symbol: str
    exchange_id: UUID
    effective_from: date
    effective_to: date | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        self.symbol = self.symbol.strip().upper()

        if not self.symbol:
            raise ValueError("InstrumentHistory symbol cannot be empty")

        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError(
                "effective_to cannot be earlier than effective_from"
            )

    @property
    def is_current(self) -> bool:
        return self.effective_to is None
