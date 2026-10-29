from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from uuid import UUID, uuid4

from app.domain.enums.corporate_action_type import CorporateActionType


@dataclass
class CorporateAction:
    """Represents a corporate action affecting an instrument.

    Corporate actions model events such as stock splits and cash dividends:
    - ``CorporateActionType.SPLIT``: ``value`` represents the split ratio defined
      as ``new_shares / old_shares``.
      For example:
        - A 2-for-1 forward split is represented by ``value = Decimal("2")``
          (two new shares for each old share). Historical pre-split prices are
          divided by 2, and volume is multiplied by 2.
        - A 1-for-2 reverse split is represented by ``value = Decimal("0.5")``
          (one new share for every two old shares).
    - ``CorporateActionType.DIVIDEND``: ``value`` represents the cash dividend per
      share distributed to shareholders as of the ex-dividend date (``execution_date``),
      expressed in ``currency``.

    Conventions:
    - ``execution_date`` is the effective market execution date (ex-date).
    - ``value`` must be a finite, strictly positive ``Decimal``. Zero, negative,
      NaN, and infinite values are strictly rejected.
    """

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
                self.action_type = CorporateActionType(self.action_type.strip().lower())
            except ValueError:
                allowed = [t.value for t in CorporateActionType]
                raise ValueError(
                    f"Invalid corporate action type: {self.action_type!r}. "
                    f"Must be one of: {allowed}"
                ) from None

        if not isinstance(self.value, Decimal):
            raise TypeError(
                f"Corporate action value must be a Decimal instance, "
                f"got {type(self.value).__name__}"
            )

        if not self.value.is_finite():
            raise ValueError(f"Corporate action value must be a finite Decimal, got {self.value}")

        if self.value <= Decimal("0"):
            raise ValueError("Corporate action value must be strictly positive")

        if self.currency is not None:
            self.currency = self.currency.strip().upper()
            if not self.currency:
                raise ValueError("Currency cannot be empty string")

        if self.description is not None:
            self.description = self.description.strip() or None
