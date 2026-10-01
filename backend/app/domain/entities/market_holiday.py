from dataclasses import dataclass, field
from datetime import date
from uuid import UUID


@dataclass
class MarketHoliday:
    holiday_id: UUID
    exchange_id: UUID
    holiday_date: date
    name: str
    session_type: str = field(default="full_day")

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.session_type = self.session_type.strip().lower()

        if not self.name:
            raise ValueError("MarketHoliday name cannot be empty")

        allowed_session_types = {"full_day", "half_day", "early_close"}
        if self.session_type not in allowed_session_types:
            raise ValueError(
                f"Invalid session_type {self.session_type!r}. "
                f"Must be one of: {allowed_session_types}"
            )
