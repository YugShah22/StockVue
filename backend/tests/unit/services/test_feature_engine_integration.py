from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from app.domain.entities.market_bar import MarketBar
from app.domain.features.calculators.atr_calculator import ATRCalculator
from app.domain.features.calculators.bollinger_bands_calculator import BollingerBandsCalculator
from app.domain.features.calculators.return_calculator import ReturnCalculator
from app.domain.features.calculators.rsi_calculator import RSICalculator
from app.domain.features.calculators.sma_calculator import SMACalculator
from app.domain.features.feature_calculation_context import FeatureCalculationContext
from app.domain.features.feature_category import FeatureCategory
from app.domain.features.feature_definition import FeatureDefinition
from app.domain.features.feature_value import FeatureValue
from app.services.features.feature_calculator_registry import create_default_feature_registry
from app.services.features.feature_engine import FeatureEngine


def make_bar(
    timestamp: datetime,
    close: Decimal,
    instrument_id: UUID,
    high: Decimal | None = None,
    low: Decimal | None = None,
) -> MarketBar:
    return MarketBar(
        instrument_id=instrument_id,
        timestamp=timestamp,
        open=close,
        high=high if high is not None else close,
        low=low if low is not None else close,
        close=close,
        volume=Decimal("5000"),
    )


class TestFeatureEngineIntegration:
    def test_default_registry_has_all_five_calculators(self) -> None:
        registry = create_default_feature_registry()
        assert len(registry) == 5
        assert "return" in registry
        assert "sma" in registry
        assert "rsi" in registry
        assert "atr" in registry
        assert "bollinger_bands" in registry
        assert isinstance(registry.get("return"), ReturnCalculator)
        assert isinstance(registry.get("sma"), SMACalculator)
        assert isinstance(registry.get("rsi"), RSICalculator)
        assert isinstance(registry.get("atr"), ATRCalculator)
        assert isinstance(registry.get("bollinger_bands"), BollingerBandsCalculator)

    def test_all_five_calculator_families_work_side_by_side(self) -> None:
        registry = create_default_feature_registry()
        engine = FeatureEngine(registry)

        inst_id = uuid4()
        as_of = date(2026, 9, 20)
        base_time = datetime(2026, 9, 1, 15, 30, tzinfo=UTC)

        # 20 bars: close = 100, 102, 104, ..., 138; high = close + 2; low = close - 2
        bars = [
            make_bar(
                base_time + timedelta(days=i),
                close=Decimal(str(100 + i * 2)),
                instrument_id=inst_id,
                high=Decimal(str(102 + i * 2)),
                low=Decimal(str(98 + i * 2)),
            )
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
        def_atr = FeatureDefinition(
            name="atr",
            category=FeatureCategory.TECHNICAL,
            description="14-period ATR",
            parameters={"window": 14},
        )
        def_bb_mid = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="5-period BB middle",
            parameters={"window": 5, "k": 2, "band": "middle"},
        )
        def_bb_upper = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="5-period BB upper",
            parameters={"window": 5, "k": 2, "band": "upper"},
        )
        def_bb_lower = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="5-period BB lower",
            parameters={"window": 5, "k": 2, "band": "lower"},
        )

        val_return = engine.calculate(def_return, context)
        val_sma = engine.calculate(def_sma, context)
        val_rsi = engine.calculate(def_rsi, context)
        val_atr = engine.calculate(def_atr, context)
        val_bb_mid = engine.calculate(def_bb_mid, context)
        val_bb_upper = engine.calculate(def_bb_upper, context)
        val_bb_lower = engine.calculate(def_bb_lower, context)

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

        # Verify ATR calculation result
        assert isinstance(val_atr, FeatureValue)
        assert val_atr.feature is def_atr
        assert val_atr.instrument_id == inst_id
        assert val_atr.observation_date == as_of
        # Every bar has high - low = 4, |high - prev_close| = 4 -> TR = 4 -> ATR = 4
        assert val_atr.value == Decimal("4")

        # Verify Bollinger Bands calculation results (middle, upper, lower)
        expected_variance = Decimal("40") / Decimal("5")  # sum of squared deviations from mean 134 is 40
        expected_std_dev = expected_variance.sqrt()

        assert isinstance(val_bb_mid, FeatureValue)
        assert val_bb_mid.feature is def_bb_mid
        assert val_bb_mid.instrument_id == inst_id
        assert val_bb_mid.observation_date == as_of
        assert val_bb_mid.value == Decimal("134")

        assert isinstance(val_bb_upper, FeatureValue)
        assert val_bb_upper.feature is def_bb_upper
        assert val_bb_upper.instrument_id == inst_id
        assert val_bb_upper.observation_date == as_of
        assert val_bb_upper.value == Decimal("134") + (Decimal("2") * expected_std_dev)

        assert isinstance(val_bb_lower, FeatureValue)
        assert val_bb_lower.feature is def_bb_lower
        assert val_bb_lower.instrument_id == inst_id
        assert val_bb_lower.observation_date == as_of
        assert val_bb_lower.value == Decimal("134") - (Decimal("2") * expected_std_dev)

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
        atr_14 = FeatureDefinition(
            name="atr", category=FeatureCategory.TECHNICAL, description="14d", parameters={"window": 14}
        )
        bb_20_mid = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="20d mid",
            parameters={"window": 20, "band": "middle"},
        )
        bb_20_upper = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="20d upper",
            parameters={"window": 20, "k": Decimal("2.5"), "band": "upper"},
        )
        bb_20_lower = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="20d lower",
            parameters={"window": 20, "k": Decimal("2.5"), "band": "lower"},
        )

        res_ret_1 = engine.calculate(ret_1, context)
        res_ret_5 = engine.calculate(ret_5, context)
        res_sma_5 = engine.calculate(sma_5, context)
        res_sma_20 = engine.calculate(sma_20, context)
        res_rsi_14 = engine.calculate(rsi_14, context)
        res_atr_14 = engine.calculate(atr_14, context)
        res_bb_20_mid = engine.calculate(bb_20_mid, context)
        res_bb_20_upper = engine.calculate(bb_20_upper, context)
        res_bb_20_lower = engine.calculate(bb_20_lower, context)

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

        assert isinstance(res_atr_14, FeatureValue)
        # Flat prices -> 0 true range
        assert res_atr_14.value == Decimal("0")

        assert isinstance(res_bb_20_mid, FeatureValue)
        # Flat prices -> standard deviation 0 -> middle is 100.00
        assert res_bb_20_mid.value == Decimal("100.00")

        assert isinstance(res_bb_20_upper, FeatureValue)
        # Flat prices -> standard deviation 0 -> upper is 100.00
        assert res_bb_20_upper.value == Decimal("100.00")

        assert isinstance(res_bb_20_lower, FeatureValue)
        # Flat prices -> standard deviation 0 -> lower is 100.00
        assert res_bb_20_lower.value == Decimal("100.00")

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

        # 14-period ATR needs 15 bars -> returns None
        def_atr_14 = FeatureDefinition(
            name="atr", category=FeatureCategory.TECHNICAL, description="14d", parameters={"window": 14}
        )
        assert engine.calculate(def_atr_14, context) is None

        # 20-period Bollinger Bands needs 20 bars -> returns None
        def_bb_20 = FeatureDefinition(
            name="bollinger_bands", category=FeatureCategory.TECHNICAL, description="20d", parameters={"window": 20}
        )
        assert engine.calculate(def_bb_20, context) is None

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
