from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID, uuid4


@dataclass
class FundamentalRecord:
    instrument_id: UUID
    period_end: date
    metric_name: str
    value: Decimal

    fiscal_year: int | None = None
    fiscal_quarter: int | None = None

    currency: str | None = None

    published_at: datetime | None = None
    available_at: datetime | None = None

    fundamental_record_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        self.metric_name = self.metric_name.strip().lower()

        if not self.metric_name:
            raise ValueError("Metric name cannot be empty")

        if self.currency is not None:
            self.currency = self.currency.strip().upper()

            if not self.currency:
                raise ValueError("Currency cannot be empty")

        if self.fiscal_year is not None and self.fiscal_year < 1900:
            raise ValueError("Fiscal year must be valid")

        if self.fiscal_quarter is not None and not 1 <= self.fiscal_quarter <= 4:
            raise ValueError("Fiscal quarter must be between 1 and 4")

        if (
            self.published_at is not None
            and self.available_at is not None
            and self.available_at < self.published_at
        ):
            raise ValueError(
                "Available timestamp cannot be earlier than published timestamp"
            )
