from dataclasses import dataclass
from uuid import UUID
from zoneinfo import ZoneInfo

from app.domain.value_objects.exchange_code import ExchangeCode


@dataclass
class Exchange:
    exchange_id: UUID
    code: ExchangeCode
    name: str
    country: str
    timezone: ZoneInfo

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.country = self.country.strip().upper()

        if not self.name:
            raise ValueError("Exchange name cannot be empty.")

        if not self.country:
            raise ValueError("Exchange country cannot be empty.")
