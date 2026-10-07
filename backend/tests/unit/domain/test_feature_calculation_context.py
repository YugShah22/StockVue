from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from app.domain.entities.fundamental_record import FundamentalRecord
from app.domain.entities.market_bar import MarketBar
from app.domain.features.feature_calculation_context import FeatureCalculationContext
from app.domain.features.feature_calculator import FeatureCalculator
from app.domain.features.feature_category import FeatureCategory
from app.domain.features.feature_definition import FeatureDefinition

# ===========================================================================
# Minimal Fake Calculator for Contract Testing
# ===========================================================================


class DummyCalculator(FeatureCalculator):
    @property
    def feature_name(self) -> str:
        return "dummy"

    def calculate(
        self,
        definition: FeatureDefinition,
        context: FeatureCalculationContext,
    ) -> Decimal | None:
        if not context.market_bars:
            return None
        return context.market_bars[0].close


class MockStaticCalculator(FeatureCalculator):
    def __init__(self, name: str, return_value: Decimal | None) -> None:
        self._name = name
        self._return_value = return_value
        self.received_definition: FeatureDefinition | None = None
        self.received_context: FeatureCalculationContext | None = None

    @property
    def feature_name(self) -> str:
        return self._name

    def calculate(
        self,
        definition: FeatureDefinition,
        context: FeatureCalculationContext,
    ) -> Decimal | None:
        self.received_definition = definition
        self.received_context = context
        return self._return_value


# ===========================================================================
# FeatureCalculationContext Tests
# ===========================================================================


class TestFeatureCalculationContext:
    def test_valid_context_creation(self) -> None:
        inst_id = uuid4()
        as_of = date(2026, 9, 1)
        bar = MarketBar(
            instrument_id=inst_id,
            timestamp=datetime(2026, 9, 1, 15, 30, tzinfo=UTC),
            open=Decimal("100.00"),
            high=Decimal("105.00"),
            low=Decimal("99.00"),
            close=Decimal("104.00"),
            volume=Decimal("10000"),
        )
        record = FundamentalRecord(
            instrument_id=inst_id,
            period_end=date(2026, 6, 30),
            metric_name="net_income",
            value=Decimal("5000000"),
        )

        ctx = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=as_of,
            market_bars=[bar],
            fundamentals=[record],
        )

        assert ctx.instrument_id == inst_id
        assert ctx.as_of_date == as_of
        assert ctx.market_bars == (bar,)
        assert ctx.fundamentals == (record,)

    def test_default_empty_sequences(self) -> None:
        inst_id = uuid4()
        as_of = date(2026, 9, 1)

        ctx = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=as_of,
        )

        assert ctx.market_bars == ()
        assert ctx.fundamentals == ()

    def test_immutability_of_context(self) -> None:
        inst_id = uuid4()
        as_of = date(2026, 9, 1)
        ctx = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=as_of,
        )

        with pytest.raises(FrozenInstanceError):
            ctx.as_of_date = date(2026, 9, 2)  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            ctx.market_bars = ()  # type: ignore[misc]

    def test_sequences_are_defensively_copied_as_tuples(self) -> None:
        inst_id = uuid4()
        as_of = date(2026, 9, 1)
        bar = MarketBar(
            instrument_id=inst_id,
            timestamp=datetime(2026, 9, 1, 15, 30, tzinfo=UTC),
            open=Decimal("100.00"),
            high=Decimal("105.00"),
            low=Decimal("99.00"),
            close=Decimal("104.00"),
            volume=Decimal("10000"),
        )
        bar_list = [bar]

        ctx = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=as_of,
            market_bars=bar_list,
        )

        # Mutating original input list does not mutate context internal tuple
        bar_list.clear()
        assert len(ctx.market_bars) == 1
        assert isinstance(ctx.market_bars, tuple)

    def test_invalid_instrument_id_type_raises(self) -> None:
        with pytest.raises(TypeError, match="instrument_id must be a UUID instance"):
            FeatureCalculationContext(
                instrument_id="not-a-uuid",  # type: ignore[arg-type]
                as_of_date=date(2026, 9, 1),
            )

    def test_datetime_as_of_date_raises(self) -> None:
        with pytest.raises(TypeError, match="as_of_date must be a date, not a datetime"):
            FeatureCalculationContext(
                instrument_id=uuid4(),
                as_of_date=datetime(2026, 9, 1, 12, 0),
            )

    def test_non_date_as_of_date_raises(self) -> None:
        with pytest.raises(TypeError, match="as_of_date must be a date instance"):
            FeatureCalculationContext(
                instrument_id=uuid4(),
                as_of_date="2026-09-01",  # type: ignore[arg-type]
            )


# ===========================================================================
# FeatureCalculator Contract Tests
# ===========================================================================


class TestFeatureCalculatorContract:
    def test_abstract_class_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            FeatureCalculator()  # type: ignore[abstract]

    def test_calculator_exposes_feature_family(self) -> None:
        calc = DummyCalculator()
        assert calc.feature_name == "dummy"

    def test_calculator_receives_definition_and_context(self) -> None:
        calc = MockStaticCalculator("test_feat", Decimal("42.50"))
        definition = FeatureDefinition(
            name="test_feat",
            category=FeatureCategory.TECHNICAL,
            description="Test feature",
        )
        ctx = FeatureCalculationContext(
            instrument_id=uuid4(),
            as_of_date=date(2026, 9, 1),
        )

        result = calc.calculate(definition, ctx)

        assert result == Decimal("42.50")
        assert calc.received_definition is definition
        assert calc.received_context is ctx

    def test_calculator_can_return_none(self) -> None:
        calc = DummyCalculator()
        definition = FeatureDefinition(
            name="dummy",
            category=FeatureCategory.TECHNICAL,
            description="Dummy feature",
        )
        ctx = FeatureCalculationContext(
            instrument_id=uuid4(),
            as_of_date=date(2026, 9, 1),
            market_bars=(),
        )

        result = calc.calculate(definition, ctx)
        assert result is None

    def test_calculator_can_return_decimal(self) -> None:
        calc = DummyCalculator()
        definition = FeatureDefinition(
            name="dummy",
            category=FeatureCategory.TECHNICAL,
            description="Dummy feature",
        )
        inst_id = uuid4()
        bar = MarketBar(
            instrument_id=inst_id,
            timestamp=datetime(2026, 9, 1, 15, 30, tzinfo=UTC),
            open=Decimal("10.0"),
            high=Decimal("12.0"),
            low=Decimal("9.0"),
            close=Decimal("11.50"),
            volume=Decimal("100"),
        )
        ctx = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 1),
            market_bars=(bar,),
        )

        result = calc.calculate(definition, ctx)
        assert result == Decimal("11.50")
