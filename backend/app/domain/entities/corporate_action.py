from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from uuid import UUID, uuid4

from app.domain.enums.corporate_action_type import CorporateActionType


@dataclass
class CorporateAction:
    instrument_id: UUID
    action_type: CorporateActionType
    execution_date: date
    value: Decimal
    currency: str | None = None
    description: str | None = None
    corporate_action_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if isinstance(self.action_type, str):
            try:
                self.action_type = CorporateActionType(
                    self.action_type.strip().lower()
                )
            except ValueError:
                allowed = [t.value for t in CorporateActionType]
                raise ValueError(
                    f"Invalid corporate action type: {self.action_type!r}. "
                    f"Must be one of: {allowed}"
                ) from None

        if self.value <= Decimal("0"):
            raise ValueError("Corporate action value must be strictly positive")

        if self.currency is not None:
            self.currency = self.currency.strip().upper()
            if not self.currency:
                raise ValueError("Currency cannot be empty string")

        if self.description is not None:
            self.description = self.description.strip() or None
