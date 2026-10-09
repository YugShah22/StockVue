from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from app.domain.entities.market_bar import MarketBar
from app.domain.features.calculators.rsi_calculator import RSICalculator
from app.domain.features.feature_calculation_context import FeatureCalculationContext
from app.domain.features.feature_category import FeatureCategory
from app.domain.features.feature_definition import FeatureDefinition


def make_bar(
    timestamp: datetime,
    close: Decimal,
    instrument_id: UUID | None = None,
) -> MarketBar:
    return MarketBar(
        instrument_id=instrument_id or uuid4(),
        timestamp=timestamp,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=Decimal("1000"),
    )


class TestRSICalculator:
    def test_feature_name_is_rsi(self) -> None:
        calc = RSICalculator()
        assert calc.feature_name == "rsi"

    def test_default_window_is_fourteen(self) -> None:
        calc = RSICalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # 14 bars (needs 15 for default window of 14) -> returns None
        bars_14 = [
            make_bar(base_time + timedelta(days=i), Decimal("100.00"), inst_id)
            for i in range(14)
        ]
        ctx_14 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 14),
            market_bars=bars_14,
        )
        definition = FeatureDefinition(
            name="rsi",
            category=FeatureCategory.TECHNICAL,
            description="14-period RSI",
            parameters={},  # window omitted
        )
        assert calc.calculate(definition, ctx_14) is None

        # 15 bars -> valid result
        bars_15 = [
            make_bar(base_time + timedelta(days=i), Decimal(str(100 + i)), inst_id)
            for i in range(15)
        ]
        ctx_15 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 15),
            market_bars=bars_15,
        )
        res_15 = calc.calculate(definition, ctx_15)
        assert isinstance(res_15, Decimal)
        # All gains, no losses -> 100
        assert res_15 == Decimal("100")

    def test_configurable_window_one(self) -> None:
        calc = RSICalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bars = [
            make_bar(base_time, Decimal("100.00"), inst_id),
            make_bar(base_time + timedelta(days=1), Decimal("105.00"), inst_id),
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 2),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="rsi",
            category=FeatureCategory.TECHNICAL,
            description="1-period RSI",
            parameters={"window": 1},
        )
        result = calc.calculate(definition, context)
        assert result == Decimal("100")

    def test_insufficient_history_returns_none(self) -> None:
        calc = RSICalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # Empty context
        ctx_empty = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 1),
            market_bars=(),
        )
        def_14 = FeatureDefinition(
            name="rsi",
            category=FeatureCategory.TECHNICAL,
            description="RSI",
            parameters={"window": 14},
        )
        assert calc.calculate(def_14, ctx_empty) is None

        # Window 5 with 5 bars (needs 6)
        bars_5 = [
            make_bar(base_time + timedelta(days=i), Decimal("100.00"), inst_id)
            for i in range(5)
        ]
        ctx_5 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 5),
            market_bars=bars_5,
        )
        def_5 = FeatureDefinition(
            name="rsi",
            category=FeatureCategory.TECHNICAL,
            description="RSI",
            parameters={"window": 5},
        )
        assert calc.calculate(def_5, ctx_5) is None

    def test_hand_calculated_initial_averages(self) -> None:
        # Window = 2, with exactly 3 bars (2 changes)
        # Closes: 100, 104, 102
        # Delta 1: +4 (gain=4, loss=0)
        # Delta 2: -2 (gain=0, loss=2)
        # Avg gain = 4/2 = 2
        # Avg loss = 2/2 = 1
        # RSI = 100 * 2 / (2 + 1) = 200 / 3
        calc = RSICalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bars = [
            make_bar(base_time, Decimal("100.00"), inst_id),
            make_bar(base_time + timedelta(days=1), Decimal("104.00"), inst_id),
            make_bar(base_time + timedelta(days=2), Decimal("102.00"), inst_id),
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 3),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="rsi",
            category=FeatureCategory.TECHNICAL,
            description="2-period RSI",
            parameters={"window": 2},
        )
        expected = Decimal("200") / Decimal("3")
        result = calc.calculate(definition, context)
        assert result == expected

    def test_hand_calculated_wilder_smoothing(self) -> None:
        # Window = 2, with 4 bars (3 changes)
        # Closes: 100, 104, 102, 108
        # Initial 2 changes:
        #   Delta 1: +4 (gain=4, loss=0)
        #   Delta 2: -2 (gain=0, loss=2)
        #   Initial avg_gain = 2, avg_loss = 1
        # Change 3 (Wilder smoothed):
        #   Delta 3: +6 (gain=6, loss=0)
        #   avg_gain = (2 * 1 + 6) / 2 = 4
        #   avg_loss = (1 * 1 + 0) / 2 = 0.5
        #   RSI = 100 * 4 / (4 + 0.5) = 400 / 4.5 = 800 / 9
        calc = RSICalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bars = [
            make_bar(base_time, Decimal("100.00"), inst_id),
            make_bar(base_time + timedelta(days=1), Decimal("104.00"), inst_id),
            make_bar(base_time + timedelta(days=2), Decimal("102.00"), inst_id),
            make_bar(base_time + timedelta(days=3), Decimal("108.00"), inst_id),
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 4),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="rsi",
            category=FeatureCategory.TECHNICAL,
            description="2-period RSI",
            parameters={"window": 2},
        )
        expected = Decimal("800") / Decimal("9")
        result = calc.calculate(definition, context)
        assert result == expected

    def test_multi_step_wilder_smoothing_case(self) -> None:
        # Window = 2, with 5 bars (4 changes)
        # Closes: 100, 104, 102, 106, 103
        # Initial 2 changes:
        #   Delta 1: +4 (gain=4, loss=0)
        #   Delta 2: -2 (gain=0, loss=2)
        #   Initial avg_gain = 4/2 = 2.0
        #   Initial avg_loss = 2/2 = 1.0
        # Change 3 (Wilder smoothed):
        #   Delta 3: +4 (gain=4, loss=0)
        #   avg_gain = (2.0 * 1 + 4) / 2 = 3.0
        #   avg_loss = (1.0 * 1 + 0) / 2 = 0.5
        # Change 4 (Wilder smoothed):
        #   Delta 4: -3 (gain=0, loss=3)
        #   avg_gain = (3.0 * 1 + 0) / 2 = 1.5
        #   avg_loss = (0.5 * 1 + 3) / 2 = 1.75
        # RSI = 100 * 1.5 / (1.5 + 1.75) = 150 / 3.25 = 600 / 13
        calc = RSICalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bars = [
            make_bar(base_time, Decimal("100.00"), inst_id),
            make_bar(base_time + timedelta(days=1), Decimal("104.00"), inst_id),
            make_bar(base_time + timedelta(days=2), Decimal("102.00"), inst_id),
            make_bar(base_time + timedelta(days=3), Decimal("106.00"), inst_id),
            make_bar(base_time + timedelta(days=4), Decimal("103.00"), inst_id),
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 5),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="rsi",
            category=FeatureCategory.TECHNICAL,
            description="2-period RSI",
            parameters={"window": 2},
        )
        expected = Decimal("600") / Decimal("13")
        result = calc.calculate(definition, context)
        assert result == expected

    def test_increasing_prices_no_losses_returns_100(self) -> None:
        calc = RSICalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bars = [
            make_bar(base_time + timedelta(days=i), Decimal(str(100 + i * 5)), inst_id)
            for i in range(10)
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 10),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="rsi",
            category=FeatureCategory.TECHNICAL,
            description="RSI",
            parameters={"window": 5},
        )
        result = calc.calculate(definition, context)
        assert result == Decimal("100")

    def test_decreasing_prices_no_gains_returns_0(self) -> None:
        calc = RSICalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bars = [
            make_bar(base_time + timedelta(days=i), Decimal(str(100 - i * 5)), inst_id)
            for i in range(10)
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 10),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="rsi",
            category=FeatureCategory.TECHNICAL,
            description="RSI",
            parameters={"window": 5},
        )
        result = calc.calculate(definition, context)
        assert result == Decimal("0")

    def test_completely_flat_prices_returns_50(self) -> None:
        calc = RSICalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bars = [
            make_bar(base_time + timedelta(days=i), Decimal("150.00"), inst_id)
            for i in range(15)
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 15),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="rsi",
            category=FeatureCategory.TECHNICAL,
            description="RSI",
            parameters={"window": 5},
        )
        result = calc.calculate(definition, context)
        assert result == Decimal("50")

    def test_unsorted_bars_produce_same_result(self) -> None:
        calc = RSICalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bar_0 = make_bar(base_time, Decimal("100.00"), inst_id)
        bar_1 = make_bar(base_time + timedelta(days=1), Decimal("104.00"), inst_id)
        bar_2 = make_bar(base_time + timedelta(days=2), Decimal("102.00"), inst_id)
        bar_3 = make_bar(base_time + timedelta(days=3), Decimal("108.00"), inst_id)

        # Scrambled order: [bar_2, bar_0, bar_3, bar_1]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 4),
            market_bars=[bar_2, bar_0, bar_3, bar_1],
        )
        definition = FeatureDefinition(
            name="rsi",
            category=FeatureCategory.TECHNICAL,
            description="2-period RSI",
            parameters={"window": 2},
        )
        expected = Decimal("800") / Decimal("9")
        assert calc.calculate(definition, context) == expected

    def test_input_context_is_not_mutated(self) -> None:
        calc = RSICalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bar_0 = make_bar(base_time, Decimal("100.00"), inst_id)
        bar_1 = make_bar(base_time + timedelta(days=1), Decimal("104.00"), inst_id)
        bar_2 = make_bar(base_time + timedelta(days=2), Decimal("102.00"), inst_id)

        original_tuple = (bar_2, bar_0, bar_1)
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 3),
            market_bars=original_tuple,
        )
        definition = FeatureDefinition(
            name="rsi",
            category=FeatureCategory.TECHNICAL,
            description="2-period RSI",
            parameters={"window": 2},
        )
        calc.calculate(definition, context)

        # Context tuple unchanged and in original order
        assert context.market_bars == original_tuple

    @pytest.mark.parametrize("invalid_window", [0, -1, -14])
    def test_invalid_numeric_window_raises_value_error(self, invalid_window: int) -> None:
        calc = RSICalculator()
        definition = FeatureDefinition(
            name="rsi",
            category=FeatureCategory.TECHNICAL,
            description="RSI",
            parameters={"window": invalid_window},
        )
        context = FeatureCalculationContext(
            instrument_id=uuid4(),
            as_of_date=date(2026, 9, 1),
        )
        with pytest.raises(ValueError, match="must be a positive integer"):
            calc.calculate(definition, context)

    @pytest.mark.parametrize("invalid_type_window", [14.0, "14", True, False, None, [14]])
    def test_invalid_type_window_raises_type_error(self, invalid_type_window: object) -> None:
        calc = RSICalculator()
        definition = FeatureDefinition(
            name="rsi",
            category=FeatureCategory.TECHNICAL,
            description="RSI",
            parameters={"window": invalid_type_window},
        )
        context = FeatureCalculationContext(
            instrument_id=uuid4(),
            as_of_date=date(2026, 9, 1),
        )
        with pytest.raises(TypeError, match="must be an integer"):
            calc.calculate(definition, context)

    def test_output_strictly_bounded_within_0_to_100(self) -> None:
        calc = RSICalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # Various arbitrary prices
        prices = [
            Decimal("100"), Decimal("105"), Decimal("102"), Decimal("98"),
            Decimal("103"), Decimal("107"), Decimal("106"), Decimal("101"),
            Decimal("104"), Decimal("108"), Decimal("110"), Decimal("109"),
            Decimal("112"), Decimal("115"), Decimal("111"), Decimal("114"),
        ]
        bars = [
            make_bar(base_time + timedelta(days=i), prices[i], inst_id)
            for i in range(len(prices))
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 16),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="rsi",
            category=FeatureCategory.TECHNICAL,
            description="RSI",
            parameters={"window": 14},
        )
        result = calc.calculate(definition, context)
        assert isinstance(result, Decimal)
        assert Decimal("0") <= result <= Decimal("100")
