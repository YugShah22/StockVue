from abc import ABC, abstractmethod
from decimal import Decimal

from app.domain.features.feature_calculation_context import FeatureCalculationContext
from app.domain.features.feature_definition import FeatureDefinition


class FeatureCalculator(ABC):
    @property
    @abstractmethod
    def feature_name(self) -> str:
        """The canonical feature family name handled by this calculator (e.g. 'return', 'sma')."""
        ...

    @abstractmethod
    def calculate(
        self,
        definition: FeatureDefinition,
        context: FeatureCalculationContext,
    ) -> Decimal | None:
        """Calculate the feature value for the given definition and context.

        Returns None if history is insufficient or value is mathematically undefined.
        """
        ...
