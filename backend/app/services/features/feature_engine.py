from decimal import Decimal

from app.domain.features.feature_calculation_context import FeatureCalculationContext
from app.domain.features.feature_definition import FeatureDefinition
from app.domain.features.feature_value import FeatureValue
from app.services.features.feature_calculator_registry import FeatureCalculatorRegistry


class FeatureEngine:
    def __init__(self, registry: FeatureCalculatorRegistry) -> None:
        if not isinstance(registry, FeatureCalculatorRegistry):
            raise TypeError(
                f"registry must be a FeatureCalculatorRegistry instance, got {type(registry).__name__}"
            )
        self._registry = registry

    @property
    def registry(self) -> FeatureCalculatorRegistry:
        return self._registry

    def calculate(
        self,
        definition: FeatureDefinition,
        context: FeatureCalculationContext,
    ) -> FeatureValue | None:
        if not isinstance(definition, FeatureDefinition):
            raise TypeError(
                f"definition must be a FeatureDefinition instance, got {type(definition).__name__}"
            )
        if not isinstance(context, FeatureCalculationContext):
            raise TypeError(
                f"context must be a FeatureCalculationContext instance, got {type(context).__name__}"
            )

        calculator = self._registry.get(definition.name)
        result = calculator.calculate(definition, context)

        if result is None:
            return None

        if not isinstance(result, Decimal):
            raise TypeError(
                f"Calculator for '{definition.name}' returned non-Decimal value: {type(result).__name__}"
            )

        return FeatureValue(
            feature=definition,
            instrument_id=context.instrument_id,
            observation_date=context.as_of_date,
            value=result,
        )
