from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from app.domain.prediction.target_definition import TargetDefinition


@dataclass(frozen=True)
class TargetValue:
    """Immutable result produced by a target calculator for a single observation.

    Attributes:
        target: The :class:`TargetDefinition` that produced this value.
        instrument_id: UUID of the instrument this target value relates to.
        observation_date: The date of the observation bar (``P_t``).
        value: The fractional forward return, e.g. ``Decimal("0.0345")`` for +3.45 %.
        realization_date: The date of the future bar (``P_{t+h}``) whose close
            was used to compute the return.
    """

    target: TargetDefinition
    instrument_id: UUID
    observation_date: date
    value: Decimal
    realization_date: date

    def __post_init__(self) -> None:
        # --- target ---------------------------------------------------------
        if not isinstance(self.target, TargetDefinition):
            raise TypeError(
                f"TargetValue.target must be a TargetDefinition instance, "
                f"got {type(self.target).__name__}"
            )

        # --- instrument_id --------------------------------------------------
        if not isinstance(self.instrument_id, UUID):
            raise TypeError(
                f"TargetValue.instrument_id must be a UUID instance, "
                f"got {type(self.instrument_id).__name__}"
            )

        # --- observation_date -----------------------------------------------
        if isinstance(self.observation_date, datetime):
            raise TypeError("TargetValue.observation_date must be a date, not a datetime")
        if not isinstance(self.observation_date, date):
            raise TypeError(
                f"TargetValue.observation_date must be a date instance, "
                f"got {type(self.observation_date).__name__}"
            )

        # --- value ----------------------------------------------------------
        if not isinstance(self.value, Decimal):
            raise TypeError(
                f"TargetValue.value must be a Decimal instance, got {type(self.value).__name__}"
            )

        # --- realization_date -----------------------------------------------
        if isinstance(self.realization_date, datetime):
            raise TypeError("TargetValue.realization_date must be a date, not a datetime")
        if not isinstance(self.realization_date, date):
            raise TypeError(
                f"TargetValue.realization_date must be a date instance, "
                f"got {type(self.realization_date).__name__}"
            )

        # realization_date must be strictly after observation_date
        if self.realization_date <= self.observation_date:
            raise ValueError(
                f"TargetValue.realization_date ({self.realization_date}) must be "
                f"strictly after observation_date ({self.observation_date})"
            )
