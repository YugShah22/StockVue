from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from app.domain.entities.market_bar import MarketBar
from app.domain.features.calculators.return_calculator import ReturnCalculator
from app.domain.features.calculators.rsi_calculator import RSICalculator
from app.domain.features.calculators.sma_calculator import SMACalculator
from app.domain.features.feature_calculation_context import FeatureCalculationContext
from app.domain.features.feature_category import FeatureCategory
from app.domain.features.feature_definition import FeatureDefinition
from app.domain.features.feature_value import FeatureValue
from app.services.features.feature_calculator_registry import create_default_feature_registry
from app.services.features.feature_engine import FeatureEngine


def make_bar(timestamp: datetime, close: Decimal, instrument_id: UUID) -> MarketBar:
    return MarketBar(
        instrument_id=instrument_id,
        timestamp=timestamp,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=Decimal("5000"),
    )


class TestFeatureEngineIntegration:
    def test_default_registry_has_all_three_calculators(self) -> None:
        registry = create_default_feature_registry()
        assert len(registry) == 3
        assert "return" in registry
        assert "sma" in registry
        assert "rsi" in registry
        assert isinstance(registry.get("return"), ReturnCalculator)
        assert isinstance(registry.get("sma"), SMACalculator)
        assert isinstance(registry.get("rsi"), RSICalculator)

    def test_all_three_calculator_families_work_side_by_side(self) -> None:
        registry = create_default_feature_registry()
        engine = FeatureEngine(registry)

        inst_id = uuid4()
        as_of = date(2026, 9, 20)
        base_time = datetime(2026, 9, 1, 15, 30, tzinfo=UTC)

        # 20 bars: strictly increasing prices 100, 102, 104, ..., 138
        bars = [
            make_bar(base_time + timedelta(days=i), Decimal(str(100 + i * 2)), inst_id)
            for i in range(20)
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=as_of,
            market_bars=bars,
        )

        def_return = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="5-period return",
            parameters={"window": 5},
        )
        def_sma = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="5-period SMA",
            parameters={"window": 5},
        )
        def_rsi = FeatureDefinition(
            name="rsi",
            category=FeatureCategory.TECHNICAL,
            description="14-period RSI",
            parameters={"window": 14},
        )

        val_return = engine.calculate(def_return, context)
        val_sma = engine.calculate(def_sma, context)
        val_rsi = engine.calculate(def_rsi, context)

        # Verify Return calculation result
        assert isinstance(val_return, FeatureValue)
        assert val_return.feature is def_return
        assert val_return.instrument_id == inst_id
        assert val_return.observation_date == as_of
        # latest is bars[19] (138), 5 ago is bars[14] (128) -> (138 / 128) - 1
        expected_return = (Decimal("138") / Decimal("128")) - Decimal("1")
        assert val_return.value == expected_return

        # Verify SMA calculation result
        assert isinstance(val_sma, FeatureValue)
        assert val_sma.feature is def_sma
        assert val_sma.instrument_id == inst_id
        assert val_sma.observation_date == as_of
        # last 5 bars: 130, 132, 134, 136, 138 -> sum 670 / 5 = 134
        expected_sma = Decimal("134")
        assert val_sma.value == expected_sma

        # Verify RSI calculation result
        assert isinstance(val_rsi, FeatureValue)
        assert val_rsi.feature is def_rsi
        assert val_rsi.instrument_id == inst_id
        assert val_rsi.observation_date == as_of
        # Monotonically increasing prices with zero losses -> RSI = 100
        assert val_rsi.value == Decimal("100")

    def test_parameterized_families_on_same_context(self) -> None:
        registry = create_default_feature_registry()
        engine = FeatureEngine(registry)

        inst_id = uuid4()
        as_of = date(2026, 9, 25)
        base_time = datetime(2026, 9, 1, 15, 30, tzinfo=UTC)

        # 25 bars with constant price 100
        bars = [
            make_bar(base_time + timedelta(days=i), Decimal("100.00"), inst_id)
            for i in range(25)
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=as_of,
            market_bars=bars,
        )

        # Multiple parameterized definitions
        ret_1 = FeatureDefinition(
            name="return", category=FeatureCategory.TECHNICAL, description="1d", parameters={"window": 1}
        )
        ret_5 = FeatureDefinition(
            name="return", category=FeatureCategory.TECHNICAL, description="5d", parameters={"window": 5}
        )
        sma_5 = FeatureDefinition(
            name="sma", category=FeatureCategory.TECHNICAL, description="5d", parameters={"window": 5}
        )
        sma_20 = FeatureDefinition(
            name="sma", category=FeatureCategory.TECHNICAL, description="20d", parameters={"window": 20}
        )
        rsi_14 = FeatureDefinition(
            name="rsi", category=FeatureCategory.TECHNICAL, description="14d", parameters={"window": 14}
        )

        res_ret_1 = engine.calculate(ret_1, context)
        res_ret_5 = engine.calculate(ret_5, context)
        res_sma_5 = engine.calculate(sma_5, context)
        res_sma_20 = engine.calculate(sma_20, context)
        res_rsi_14 = engine.calculate(rsi_14, context)

        assert isinstance(res_ret_1, FeatureValue)
        assert res_ret_1.value == Decimal("0")

        assert isinstance(res_ret_5, FeatureValue)
        assert res_ret_5.value == Decimal("0")

        assert isinstance(res_sma_5, FeatureValue)
        assert res_sma_5.value == Decimal("100.00")

        assert isinstance(res_sma_20, FeatureValue)
        assert res_sma_20.value == Decimal("100.00")

        assert isinstance(res_rsi_14, FeatureValue)
        # Flat prices -> neutral 50
        assert res_rsi_14.value == Decimal("50")

    def test_insufficient_history_propagates_none_through_engine(self) -> None:
        registry = create_default_feature_registry()
        engine = FeatureEngine(registry)

        inst_id = uuid4()
        as_of = date(2026, 9, 2)
        base_time = datetime(2026, 9, 1, 15, 30, tzinfo=UTC)

        # Only 2 bars
        bars = [
            make_bar(base_time, Decimal("100.00"), inst_id),
            make_bar(base_time + timedelta(days=1), Decimal("102.00"), inst_id),
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=as_of,
            market_bars=bars,
        )

        # 5-period return needs 6 bars -> returns None
        def_ret_5 = FeatureDefinition(
            name="return", category=FeatureCategory.TECHNICAL, description="5d", parameters={"window": 5}
        )
        assert engine.calculate(def_ret_5, context) is None

        # 20-period SMA needs 20 bars -> returns None
        def_sma_20 = FeatureDefinition(
            name="sma", category=FeatureCategory.TECHNICAL, description="20d", parameters={"window": 20}
        )
        assert engine.calculate(def_sma_20, context) is None

        # 14-period RSI needs 15 bars -> returns None
        def_rsi_14 = FeatureDefinition(
            name="rsi", category=FeatureCategory.TECHNICAL, description="14d", parameters={"window": 14}
        )
        assert engine.calculate(def_rsi_14, context) is None

    def test_unknown_calculator_name_raises_key_error(self) -> None:
        registry = create_default_feature_registry()
        engine = FeatureEngine(registry)

        def_unknown = FeatureDefinition(
            name="macd", category=FeatureCategory.TECHNICAL, description="MACD", parameters={"fast": 12}
        )
        context = FeatureCalculationContext(
            instrument_id=uuid4(),
            as_of_date=date(2026, 9, 1),
        )

        with pytest.raises(KeyError, match="No calculator registered for feature family: 'macd'"):
            engine.calculate(def_unknown, context)
