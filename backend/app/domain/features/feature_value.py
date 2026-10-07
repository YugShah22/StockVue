from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from app.domain.features.feature_definition import FeatureDefinition


@dataclass
class FeatureValue:
    feature: FeatureDefinition
    instrument_id: UUID
    observation_date: date
    value: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.feature, FeatureDefinition):
            raise TypeError(
                f"feature must be a FeatureDefinition instance, got {type(self.feature).__name__}"
            )

        if not isinstance(self.instrument_id, UUID):
            raise TypeError(
                f"instrument_id must be a UUID instance, got {type(self.instrument_id).__name__}"
            )

        if isinstance(self.observation_date, datetime):
            raise TypeError("observation_date must be a date, not a datetime")

        if not isinstance(self.observation_date, date):
            raise TypeError(
                f"observation_date must be a date instance, got {type(self.observation_date).__name__}"
            )

        if not isinstance(self.value, Decimal):
            raise TypeError(
                f"value must be a Decimal instance, got {type(self.value).__name__}"
            )
