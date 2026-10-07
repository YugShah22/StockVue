from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest

from app.domain.features.feature_calculation_context import FeatureCalculationContext
from app.domain.features.feature_calculator import FeatureCalculator
from app.domain.features.feature_category import FeatureCategory
from app.domain.features.feature_definition import FeatureDefinition
from app.domain.features.feature_value import FeatureValue
from app.services.features.feature_calculator_registry import FeatureCalculatorRegistry
from app.services.features.feature_engine import FeatureEngine

# ===========================================================================
# Minimal Fake Calculators for Service Tests
# ===========================================================================


class FakeCalculator(FeatureCalculator):
    def __init__(self, name: str, return_value: Decimal | None) -> None:
        self._name = name
        self._return_value = return_value

    @property
    def feature_name(self) -> str:
        return self._name

    def calculate(
        self,
        definition: FeatureDefinition,
        context: FeatureCalculationContext,
    ) -> Decimal | None:
        return self._return_value


class BadReturnCalculator(FeatureCalculator):
    @property
    def feature_name(self) -> str:
        return "bad_return"

    def calculate(
        self,
        definition: FeatureDefinition,
        context: FeatureCalculationContext,
    ) -> Decimal | None:
        return "not-a-decimal"  # type: ignore[return-value]


# ===========================================================================
# FeatureCalculatorRegistry Tests
# ===========================================================================


class TestFeatureCalculatorRegistry:
    def test_register_and_get_calculator(self) -> None:
        registry = FeatureCalculatorRegistry()
        calc = FakeCalculator("test_feature", Decimal("100.0"))

        registry.register(calc)
        resolved = registry.get("test_feature")

        assert resolved is calc
        assert "test_feature" in registry
        assert len(registry) == 1

    def test_case_normalization(self) -> None:
        registry = FeatureCalculatorRegistry()
        calc = FakeCalculator("Return_20D", Decimal("0.05"))

        registry.register(calc)

        # Lookup with lowercase, uppercase, and mixed case
        assert registry.get("return_20d") is calc
        assert registry.get("RETURN_20D") is calc
        assert registry.get("Return_20d") is calc
        assert "RETURN_20D" in registry

    def test_whitespace_normalization(self) -> None:
        registry = FeatureCalculatorRegistry()
        calc = FakeCalculator("   sma_50   ", Decimal("50.0"))

        registry.register(calc)

        assert registry.get("sma_50") is calc
        assert registry.get("   sma_50   ") is calc
        assert "  sma_50  " in registry

    def test_duplicate_registration_raises(self) -> None:
        registry = FeatureCalculatorRegistry()
        calc1 = FakeCalculator("return", Decimal("1.0"))
        calc2 = FakeCalculator("RETURN", Decimal("2.0"))

        registry.register(calc1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(calc2)

    def test_unknown_calculator_lookup_raises(self) -> None:
        registry = FeatureCalculatorRegistry()
        with pytest.raises(KeyError, match="No calculator registered for feature family"):
            registry.get("unknown_feat")

    def test_empty_feature_name_raises(self) -> None:
        registry = FeatureCalculatorRegistry()
        with pytest.raises(ValueError, match="Feature name cannot be empty"):
            registry.get("   ")

    def test_register_empty_name_calculator_raises(self) -> None:
        registry = FeatureCalculatorRegistry()
        calc = FakeCalculator("   ", Decimal("1.0"))
        with pytest.raises(ValueError, match="Calculator feature_name cannot be empty"):
            registry.register(calc)

    def test_register_invalid_type_raises(self) -> None:
        registry = FeatureCalculatorRegistry()
        with pytest.raises(TypeError, match="must be a FeatureCalculator instance"):
            registry.register("not-a-calculator")  # type: ignore[arg-type]

    def test_get_invalid_type_raises(self) -> None:
        registry = FeatureCalculatorRegistry()
        with pytest.raises(TypeError, match="feature_name must be a string"):
            registry.get(123)  # type: ignore[arg-type]


# ===========================================================================
# FeatureEngine Tests
# ===========================================================================


class TestFeatureEngine:
    def test_engine_init_validates_registry(self) -> None:
        with pytest.raises(TypeError, match="must be a FeatureCalculatorRegistry"):
            FeatureEngine("not-a-registry")  # type: ignore[arg-type]

    def test_valid_decimal_result_produces_feature_value(self) -> None:
        registry = FeatureCalculatorRegistry()
        expected_val = Decimal("42.500000")
        calc = FakeCalculator("momentum", expected_val)
        registry.register(calc)

        engine = FeatureEngine(registry)

        definition = FeatureDefinition(
            name="momentum",
            category=FeatureCategory.TECHNICAL,
            description="Momentum indicator",
            parameters={"window": 14},
        )
        inst_id = uuid4()
        obs_date = date(2026, 9, 1)
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=obs_date,
        )

        result = engine.calculate(definition, context)

        assert isinstance(result, FeatureValue)
        assert result.feature is definition
        assert result.instrument_id == inst_id
        assert result.observation_date == obs_date
        assert result.value == expected_val

    def test_calculator_returning_none_causes_engine_to_return_none(self) -> None:
        registry = FeatureCalculatorRegistry()
        calc = FakeCalculator("momentum", None)  # Simulates insufficient history
        registry.register(calc)

        engine = FeatureEngine(registry)

        definition = FeatureDefinition(
            name="momentum",
            category=FeatureCategory.TECHNICAL,
            description="Momentum indicator",
        )
        context = FeatureCalculationContext(
            instrument_id=uuid4(),
            as_of_date=date(2026, 9, 1),
        )

        result = engine.calculate(definition, context)
        assert result is None

    def test_unknown_feature_propagates_registry_error(self) -> None:
        registry = FeatureCalculatorRegistry()
        engine = FeatureEngine(registry)

        definition = FeatureDefinition(
            name="unregistered_feat",
            category=FeatureCategory.TECHNICAL,
            description="Unregistered feature",
        )
        context = FeatureCalculationContext(
            instrument_id=uuid4(),
            as_of_date=date(2026, 9, 1),
        )

        with pytest.raises(KeyError, match="No calculator registered for feature family"):
            engine.calculate(definition, context)

    def test_invalid_definition_type_raises(self) -> None:
        registry = FeatureCalculatorRegistry()
        engine = FeatureEngine(registry)
        context = FeatureCalculationContext(
            instrument_id=uuid4(),
            as_of_date=date(2026, 9, 1),
        )

        with pytest.raises(TypeError, match="definition must be a FeatureDefinition instance"):
            engine.calculate("not-a-definition", context)  # type: ignore[arg-type]

    def test_invalid_context_type_raises(self) -> None:
        registry = FeatureCalculatorRegistry()
        engine = FeatureEngine(registry)
        definition = FeatureDefinition(
            name="test",
            category=FeatureCategory.TECHNICAL,
            description="Test",
        )

        with pytest.raises(TypeError, match="context must be a FeatureCalculationContext instance"):
            engine.calculate(definition, "not-a-context")  # type: ignore[arg-type]

    def test_non_decimal_return_value_raises(self) -> None:
        registry = FeatureCalculatorRegistry()
        calc = BadReturnCalculator()
        registry.register(calc)

        engine = FeatureEngine(registry)
        definition = FeatureDefinition(
            name="bad_return",
            category=FeatureCategory.TECHNICAL,
            description="Returns non-decimal",
        )
        context = FeatureCalculationContext(
            instrument_id=uuid4(),
            as_of_date=date(2026, 9, 1),
        )

        with pytest.raises(TypeError, match="returned non-Decimal value"):
            engine.calculate(definition, context)
