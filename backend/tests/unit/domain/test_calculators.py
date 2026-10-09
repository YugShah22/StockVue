from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from app.domain.entities.market_bar import MarketBar
from app.domain.features.calculators.return_calculator import ReturnCalculator
from app.domain.features.calculators.sma_calculator import SMACalculator
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


# ===========================================================================
# ReturnCalculator Unit Tests
# ===========================================================================


class TestReturnCalculator:
    def test_feature_name_is_return(self) -> None:
        calc = ReturnCalculator()
        assert calc.feature_name == "return"

    def test_default_window_is_one(self) -> None:
        calc = ReturnCalculator()
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
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="1-period return",
            parameters={},  # window omitted
        )

        result = calc.calculate(definition, context)
        assert result == Decimal("0.05")

    def test_configurable_windows(self) -> None:
        calc = ReturnCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # 6 bars: prices 100, 102, 104, 106, 108, 110
        bars = [
            make_bar(base_time + timedelta(days=i), Decimal(str(100 + i * 2)), inst_id)
            for i in range(6)
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 6),
            market_bars=bars,
        )

        # 5-period return: latest is 110 (index 5), 5 periods ago is 100 (index 0)
        def_5 = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="5-period return",
            parameters={"window": 5},
        )
        assert calc.calculate(def_5, context) == Decimal("0.10")

        # 2-period return: latest is 110 (index 5), 2 periods ago is 106 (index 3)
        # (110 / 106) - 1
        def_2 = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="2-period return",
            parameters={"window": 2},
        )
        expected = (Decimal("110") / Decimal("106")) - Decimal("1")
        assert calc.calculate(def_2, context) == expected

    def test_positive_negative_and_zero_returns(self) -> None:
        calc = ReturnCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        def_1 = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="1-period return",
            parameters={"window": 1},
        )

        # Positive return: 100 -> 115
        ctx_pos = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 2),
            market_bars=[
                make_bar(base_time, Decimal("100.00"), inst_id),
                make_bar(base_time + timedelta(days=1), Decimal("115.00"), inst_id),
            ],
        )
        assert calc.calculate(def_1, ctx_pos) == Decimal("0.15")

        # Negative return: 100 -> 80
        ctx_neg = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 2),
            market_bars=[
                make_bar(base_time, Decimal("100.00"), inst_id),
                make_bar(base_time + timedelta(days=1), Decimal("80.00"), inst_id),
            ],
        )
        assert calc.calculate(def_1, ctx_neg) == Decimal("-0.20")

        # Zero return: 100 -> 100
        ctx_zero = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 2),
            market_bars=[
                make_bar(base_time, Decimal("100.00"), inst_id),
                make_bar(base_time + timedelta(days=1), Decimal("100.00"), inst_id),
            ],
        )
        assert calc.calculate(def_1, ctx_zero) == Decimal("0")

    def test_insufficient_history_returns_none(self) -> None:
        calc = ReturnCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # 0 bars
        ctx_empty = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 1),
            market_bars=[],
        )
        def_1 = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="1-period return",
            parameters={"window": 1},
        )
        assert calc.calculate(def_1, ctx_empty) is None

        # 1 bar for window=1 (needs 2 bars)
        ctx_1 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 1),
            market_bars=[make_bar(base_time, Decimal("100.00"), inst_id)],
        )
        assert calc.calculate(def_1, ctx_1) is None

        # 5 bars for window=5 (needs 6 bars)
        ctx_5 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 5),
            market_bars=[
                make_bar(base_time + timedelta(days=i), Decimal("100.00"), inst_id)
                for i in range(5)
            ],
        )
        def_5 = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="5-period return",
            parameters={"window": 5},
        )
        assert calc.calculate(def_5, ctx_5) is None

    def test_zero_or_negative_comparison_price_returns_none(self) -> None:
        calc = ReturnCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
        def_1 = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="1-period return",
            parameters={"window": 1},
        )

        # Zero comparison price
        ctx_zero = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 2),
            market_bars=[
                make_bar(base_time, Decimal("0.00"), inst_id),
                make_bar(base_time + timedelta(days=1), Decimal("100.00"), inst_id),
            ],
        )
        assert calc.calculate(def_1, ctx_zero) is None

        # Negative comparison price (if encountered)
        # Note: MarketBar rejects negative in __post_init__, but we bypass via object.__setattr__ to test calculator defense
        bar_neg = make_bar(base_time, Decimal("10.00"), inst_id)
        object.__setattr__(bar_neg, "close", Decimal("-10.00"))
        ctx_neg = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 2),
            market_bars=[
                bar_neg,
                make_bar(base_time + timedelta(days=1), Decimal("100.00"), inst_id),
            ],
        )
        assert calc.calculate(def_1, ctx_neg) is None

    @pytest.mark.parametrize("invalid_window", [0, -1, -20])
    def test_invalid_numeric_window_raises_value_error(self, invalid_window: int) -> None:
        calc = ReturnCalculator()
        definition = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="Return",
            parameters={"window": invalid_window},
        )
        context = FeatureCalculationContext(
            instrument_id=uuid4(),
            as_of_date=date(2026, 9, 1),
        )
        with pytest.raises(ValueError, match="must be a positive integer"):
            calc.calculate(definition, context)

    @pytest.mark.parametrize("invalid_type_window", [5.5, "5", True, False, None, [5]])
    def test_invalid_type_window_raises_type_error(self, invalid_type_window: object) -> None:
        calc = ReturnCalculator()
        definition = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="Return",
            parameters={"window": invalid_type_window},
        )
        context = FeatureCalculationContext(
            instrument_id=uuid4(),
            as_of_date=date(2026, 9, 1),
        )
        with pytest.raises(TypeError, match="must be an integer"):
            calc.calculate(definition, context)

    def test_unsorted_market_bars_produce_correct_result(self) -> None:
        calc = ReturnCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bar_0 = make_bar(base_time, Decimal("100.00"), inst_id)
        bar_1 = make_bar(base_time + timedelta(days=1), Decimal("102.00"), inst_id)
        bar_2 = make_bar(base_time + timedelta(days=2), Decimal("110.00"), inst_id)

        # Supplied in reverse order [bar_2, bar_0, bar_1]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 3),
            market_bars=[bar_2, bar_0, bar_1],
        )

        # 1-period return: latest is bar_2 (110), 1 period ago is bar_1 (102)
        def_1 = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="1-period return",
            parameters={"window": 1},
        )
        expected = (Decimal("110.00") / Decimal("102.00")) - Decimal("1")
        assert calc.calculate(def_1, context) == expected

        # 2-period return: latest is bar_2 (110), 2 periods ago is bar_0 (100)
        def_2 = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="2-period return",
            parameters={"window": 2},
        )
        assert calc.calculate(def_2, context) == Decimal("0.10")

    def test_decimal_precision_is_preserved(self) -> None:
        calc = ReturnCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        p0 = Decimal("100.123456")
        p1 = Decimal("105.789101")
        bars = [
            make_bar(base_time, p0, inst_id),
            make_bar(base_time + timedelta(days=1), p1, inst_id),
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 2),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="Return",
            parameters={"window": 1},
        )
        result = calc.calculate(definition, context)
        assert isinstance(result, Decimal)
        expected = (p1 / p0) - Decimal("1")
        assert result == expected


# ===========================================================================
# SMACalculator Unit Tests
# ===========================================================================


class TestSMACalculator:
    def test_feature_name_is_sma(self) -> None:
        calc = SMACalculator()
        assert calc.feature_name == "sma"

    def test_default_window_is_twenty(self) -> None:
        calc = SMACalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # 20 bars all with close 150.00
        bars = [
            make_bar(base_time + timedelta(days=i), Decimal("150.00"), inst_id)
            for i in range(20)
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 20),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="20-day SMA",
            parameters={},  # window omitted
        )

        assert calc.calculate(definition, context) == Decimal("150.00")

    def test_configurable_windows(self) -> None:
        calc = SMACalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        prices = [Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40"), Decimal("50")]
        bars = [
            make_bar(base_time + timedelta(days=i), prices[i], inst_id)
            for i in range(len(prices))
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 5),
            market_bars=bars,
        )

        # 5-period SMA: (10 + 20 + 30 + 40 + 50) / 5 = 30
        def_5 = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="5-day SMA",
            parameters={"window": 5},
        )
        assert calc.calculate(def_5, context) == Decimal("30")

        # 3-period SMA: last 3 bars (30 + 40 + 50) / 3 = 40
        def_3 = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="3-day SMA",
            parameters={"window": 3},
        )
        assert calc.calculate(def_3, context) == Decimal("40")

    def test_older_bars_outside_window_do_not_affect_result(self) -> None:
        calc = SMACalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # 10 older bars with extreme price 1000
        older_bars = [
            make_bar(base_time + timedelta(days=i), Decimal("1000.00"), inst_id)
            for i in range(10)
        ]
        # 3 latest bars with price 10, 20, 30
        latest_bars = [
            make_bar(base_time + timedelta(days=10 + i), Decimal(str((i + 1) * 10)), inst_id)
            for i in range(3)
        ]

        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 13),
            market_bars=older_bars + latest_bars,
        )
        definition = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="3-day SMA",
            parameters={"window": 3},
        )

        # Average of (10 + 20 + 30) / 3 = 20, older 1000 prices completely excluded
        assert calc.calculate(definition, context) == Decimal("20")

    def test_insufficient_history_returns_none(self) -> None:
        calc = SMACalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # 19 bars for default window 20
        bars_19 = [
            make_bar(base_time + timedelta(days=i), Decimal("100.00"), inst_id)
            for i in range(19)
        ]
        ctx_19 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 19),
            market_bars=bars_19,
        )
        def_20 = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="20-day SMA",
            parameters={"window": 20},
        )
        assert calc.calculate(def_20, ctx_19) is None

        # 0 bars
        ctx_0 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 1),
            market_bars=[],
        )
        assert calc.calculate(def_20, ctx_0) is None

    @pytest.mark.parametrize("invalid_window", [0, -1, -50])
    def test_invalid_numeric_window_raises_value_error(self, invalid_window: int) -> None:
        calc = SMACalculator()
        definition = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="SMA",
            parameters={"window": invalid_window},
        )
        context = FeatureCalculationContext(
            instrument_id=uuid4(),
            as_of_date=date(2026, 9, 1),
        )
        with pytest.raises(ValueError, match="must be a positive integer"):
            calc.calculate(definition, context)

    @pytest.mark.parametrize("invalid_type_window", [20.0, "20", True, False, None, [20]])
    def test_invalid_type_window_raises_type_error(self, invalid_type_window: object) -> None:
        calc = SMACalculator()
        definition = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="SMA",
            parameters={"window": invalid_type_window},
        )
        context = FeatureCalculationContext(
            instrument_id=uuid4(),
            as_of_date=date(2026, 9, 1),
        )
        with pytest.raises(TypeError, match="must be an integer"):
            calc.calculate(definition, context)

    def test_unsorted_market_bars_produce_correct_result(self) -> None:
        calc = SMACalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bar_0 = make_bar(base_time, Decimal("10.00"), inst_id)
        bar_1 = make_bar(base_time + timedelta(days=1), Decimal("20.00"), inst_id)
        bar_2 = make_bar(base_time + timedelta(days=2), Decimal("30.00"), inst_id)

        # Scrambled order
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 3),
            market_bars=[bar_1, bar_2, bar_0],
        )
        def_3 = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="3-day SMA",
            parameters={"window": 3},
        )
        assert calc.calculate(def_3, context) == Decimal("20.00")

    def test_decimal_arithmetic_precision(self) -> None:
        calc = SMACalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        p1 = Decimal("10.25")
        p2 = Decimal("20.75")
        bars = [
            make_bar(base_time, p1, inst_id),
            make_bar(base_time + timedelta(days=1), p2, inst_id),
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 2),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="2-day SMA",
            parameters={"window": 2},
        )
        result = calc.calculate(definition, context)
        assert isinstance(result, Decimal)
        assert result == Decimal("15.50")
