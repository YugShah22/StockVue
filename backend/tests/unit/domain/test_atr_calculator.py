from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from app.domain.entities.market_bar import MarketBar
from app.domain.features.calculators.atr_calculator import ATRCalculator
from app.domain.features.feature_calculation_context import FeatureCalculationContext
from app.domain.features.feature_category import FeatureCategory
from app.domain.features.feature_definition import FeatureDefinition


def make_bar(
    timestamp: datetime,
    close: Decimal,
    high: Decimal | None = None,
    low: Decimal | None = None,
    open_: Decimal | None = None,
    instrument_id: UUID | None = None,
) -> MarketBar:
    high_val = high if high is not None else close
    low_val = low if low is not None else close
    open_val = open_ if open_ is not None else close
    return MarketBar(
        instrument_id=instrument_id or uuid4(),
        timestamp=timestamp,
        open=open_val,
        high=high_val,
        low=low_val,
        close=close,
        volume=Decimal("1000"),
    )


class TestATRCalculator:
    def test_canonical_feature_name(self) -> None:
        calc = ATRCalculator()
        assert calc.feature_name == "atr"

    def test_default_window_is_fourteen(self) -> None:
        calc = ATRCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # 14 bars (needs 15 for default window of 14) -> returns None
        bars_14 = [
            make_bar(base_time + timedelta(days=i), Decimal("100.00"), instrument_id=inst_id)
            for i in range(14)
        ]
        ctx_14 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 14),
            market_bars=bars_14,
        )
        definition = FeatureDefinition(
            name="atr",
            category=FeatureCategory.TECHNICAL,
            description="14-period ATR",
            parameters={},  # window omitted
        )
        assert calc.calculate(definition, ctx_14) is None

        # 15 bars -> valid result
        bars_15 = [
            make_bar(
                base_time + timedelta(days=i),
                close=Decimal("100.00"),
                high=Decimal("102.00"),
                low=Decimal("98.00"),
                instrument_id=inst_id,
            )
            for i in range(15)
        ]
        ctx_15 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 15),
            market_bars=bars_15,
        )
        result = calc.calculate(definition, ctx_15)
        assert isinstance(result, Decimal)
        # All 14 TRs are 102 - 98 = 4 -> average is 4
        assert result == Decimal("4")

    def test_custom_window_one(self) -> None:
        calc = ATRCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # 2 bars: Bar 0 close=100. Bar 1 high=105, low=99, close=104
        # TR_1 = max(105-99, |105-100|, |99-100|) = max(6, 5, 1) = 6
        bars = [
            make_bar(base_time, close=Decimal("100.00"), instrument_id=inst_id),
            make_bar(
                base_time + timedelta(days=1),
                close=Decimal("104.00"),
                high=Decimal("105.00"),
                low=Decimal("99.00"),
                instrument_id=inst_id,
            ),
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 2),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="atr",
            category=FeatureCategory.TECHNICAL,
            description="1-period ATR",
            parameters={"window": 1},
        )
        result = calc.calculate(definition, context)
        assert result == Decimal("6")

        # 3 bars with window=1: Wilder smoothing updates to latest TR
        # Bar 2: high=112, low=101, close=110. prev_close=104
        # TR_2 = max(112-101, |112-104|, |101-104|) = max(11, 8, 3) = 11
        # ATR_2 = (6 * 0 + 11) / 1 = 11
        bar_2 = make_bar(
            base_time + timedelta(days=2),
            close=Decimal("110.00"),
            high=Decimal("112.00"),
            low=Decimal("101.00"),
            instrument_id=inst_id,
        )
        ctx_3 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 3),
            market_bars=bars + [bar_2],
        )
        assert calc.calculate(definition, ctx_3) == Decimal("11")

    def test_custom_window_greater_than_one(self) -> None:
        calc = ATRCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bars = [
            make_bar(base_time, close=Decimal("100.00"), instrument_id=inst_id),
            make_bar(
                base_time + timedelta(days=1),
                close=Decimal("102.00"),
                high=Decimal("105.00"),
                low=Decimal("99.00"),
                instrument_id=inst_id,
            ),
            make_bar(
                base_time + timedelta(days=2),
                close=Decimal("104.00"),
                high=Decimal("106.00"),
                low=Decimal("100.00"),
                instrument_id=inst_id,
            ),
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 3),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="atr",
            category=FeatureCategory.TECHNICAL,
            description="2-period ATR",
            parameters={"window": 2},
        )
        # TR_1: max(6, 5, 1) = 6
        # TR_2: max(6, |106-102|=4, |100-102|=2) = 6
        # Initial ATR: (6 + 6) / 2 = 6
        result = calc.calculate(definition, context)
        assert result == Decimal("6")

    def test_ordinary_bar_intraday_range_dominant(self) -> None:
        # Intraday range (high - low) is strictly greater than gap components
        calc = ATRCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bars = [
            make_bar(base_time, close=Decimal("100.00"), instrument_id=inst_id),
            make_bar(
                base_time + timedelta(days=1),
                close=Decimal("101.00"),
                high=Decimal("110.00"),
                low=Decimal("90.00"),
                instrument_id=inst_id,
            ),
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 2),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="atr",
            category=FeatureCategory.TECHNICAL,
            description="1-period ATR",
            parameters={"window": 1},
        )
        # high - low = 110 - 90 = 20
        # |high - prev_close| = |110 - 100| = 10
        # |low - prev_close| = |90 - 100| = 10
        # TR = 20
        assert calc.calculate(definition, context) == Decimal("20")

    def test_true_range_capturing_upward_gap(self) -> None:
        # Upward gap: |high - prev_close| is the greatest component
        calc = ATRCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bars = [
            make_bar(base_time, close=Decimal("100.00"), instrument_id=inst_id),
            make_bar(
                base_time + timedelta(days=1),
                close=Decimal("108.00"),
                high=Decimal("110.00"),
                low=Decimal("105.00"),
                instrument_id=inst_id,
            ),
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 2),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="atr",
            category=FeatureCategory.TECHNICAL,
            description="1-period ATR",
            parameters={"window": 1},
        )
        # high - low = 110 - 105 = 5
        # |high - prev_close| = |110 - 100| = 10  <- greatest
        # |low - prev_close| = |105 - 100| = 5
        # TR = 10
        assert calc.calculate(definition, context) == Decimal("10")

    def test_true_range_capturing_downward_gap(self) -> None:
        # Downward gap: |low - prev_close| is the greatest component
        calc = ATRCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bars = [
            make_bar(base_time, close=Decimal("100.00"), instrument_id=inst_id),
            make_bar(
                base_time + timedelta(days=1),
                close=Decimal("88.00"),
                high=Decimal("92.00"),
                low=Decimal("85.00"),
                instrument_id=inst_id,
            ),
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 2),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="atr",
            category=FeatureCategory.TECHNICAL,
            description="1-period ATR",
            parameters={"window": 1},
        )
        # high - low = 92 - 85 = 7
        # |high - prev_close| = |92 - 100| = 8
        # |low - prev_close| = |85 - 100| = 15  <- greatest
        # TR = 15
        assert calc.calculate(definition, context) == Decimal("15")

    def test_initial_atr_arithmetic_average(self) -> None:
        # Window = 2 with exactly 3 bars
        calc = ATRCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bars = [
            make_bar(base_time, close=Decimal("100.00"), high=Decimal("102.00"), low=Decimal("98.00"), instrument_id=inst_id),
            make_bar(base_time + timedelta(days=1), close=Decimal("104.00"), high=Decimal("106.00"), low=Decimal("100.00"), instrument_id=inst_id),
            make_bar(base_time + timedelta(days=2), close=Decimal("98.00"), high=Decimal("103.00"), low=Decimal("96.00"), instrument_id=inst_id),
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 3),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="atr",
            category=FeatureCategory.TECHNICAL,
            description="2-period ATR",
            parameters={"window": 2},
        )
        # TR_1: max(6, 6, 0) = 6
        # TR_2: max(7, 1, 8) = 8
        # Initial ATR: (6 + 8) / 2 = 7
        assert calc.calculate(definition, context) == Decimal("7")

    def test_wilder_smoothing_across_subsequent_bars(self) -> None:
        # Window = 2 with 4 bars
        calc = ATRCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bars = [
            make_bar(base_time, close=Decimal("100.00"), high=Decimal("102.00"), low=Decimal("98.00"), instrument_id=inst_id),
            make_bar(base_time + timedelta(days=1), close=Decimal("104.00"), high=Decimal("106.00"), low=Decimal("100.00"), instrument_id=inst_id),
            make_bar(base_time + timedelta(days=2), close=Decimal("98.00"), high=Decimal("103.00"), low=Decimal("96.00"), instrument_id=inst_id),
            make_bar(base_time + timedelta(days=3), close=Decimal("105.00"), high=Decimal("108.00"), low=Decimal("102.00"), instrument_id=inst_id),
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 4),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="atr",
            category=FeatureCategory.TECHNICAL,
            description="2-period ATR",
            parameters={"window": 2},
        )
        # Initial ATR (from first 2 TRs: 6 and 8) = 7
        # Bar 3: TR_3 = max(108-102, |108-98|, |102-98|) = max(6, 10, 4) = 10
        # Smoothed ATR: (7 * 1 + 10) / 2 = 17 / 2 = 8.5
        assert calc.calculate(definition, context) == Decimal("8.5")

    def test_additional_history_affects_recursive_atr(self) -> None:
        calc = ATRCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bars_4 = [
            make_bar(base_time, close=Decimal("100.00"), high=Decimal("102.00"), low=Decimal("98.00"), instrument_id=inst_id),
            make_bar(base_time + timedelta(days=1), close=Decimal("104.00"), high=Decimal("106.00"), low=Decimal("100.00"), instrument_id=inst_id),
            make_bar(base_time + timedelta(days=2), close=Decimal("98.00"), high=Decimal("103.00"), low=Decimal("96.00"), instrument_id=inst_id),
            make_bar(base_time + timedelta(days=3), close=Decimal("105.00"), high=Decimal("108.00"), low=Decimal("102.00"), instrument_id=inst_id),
        ]
        bar_5 = make_bar(
            base_time + timedelta(days=4),
            close=Decimal("100.00"),
            high=Decimal("105.00"),
            low=Decimal("99.00"),
            instrument_id=inst_id,
        )
        # Bar 4: TR_4 = max(105-99, |105-105|, |99-105|) = max(6, 0, 6) = 6
        # Previous ATR was 8.5.
        # Smoothed ATR_4: (8.5 * 1 + 6) / 2 = 14.5 / 2 = 7.25
        definition = FeatureDefinition(
            name="atr",
            category=FeatureCategory.TECHNICAL,
            description="2-period ATR",
            parameters={"window": 2},
        )
        ctx_4 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 4),
            market_bars=bars_4,
        )
        ctx_5 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 5),
            market_bars=bars_4 + [bar_5],
        )

        res_4 = calc.calculate(definition, ctx_4)
        res_5 = calc.calculate(definition, ctx_5)

        assert res_4 == Decimal("8.5")
        assert res_5 == Decimal("7.25")
        assert res_4 != res_5

    def test_insufficient_history_returns_none(self) -> None:
        calc = ATRCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        def_14 = FeatureDefinition(
            name="atr",
            category=FeatureCategory.TECHNICAL,
            description="14-period ATR",
            parameters={"window": 14},
        )

        # 0 bars
        ctx_0 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 1),
            market_bars=(),
        )
        assert calc.calculate(def_14, ctx_0) is None

        # 1 bar
        ctx_1 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 1),
            market_bars=[make_bar(base_time, Decimal("100.00"), instrument_id=inst_id)],
        )
        assert calc.calculate(def_14, ctx_1) is None

        # 14 bars for window=14 (needs 15 bars)
        bars_14 = [
            make_bar(base_time + timedelta(days=i), Decimal("100.00"), instrument_id=inst_id)
            for i in range(14)
        ]
        ctx_14 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 14),
            market_bars=bars_14,
        )
        assert calc.calculate(def_14, ctx_14) is None

    def test_exactly_sufficient_history_produces_result(self) -> None:
        calc = ATRCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # Exactly 15 bars for window 14
        bars_15 = [
            make_bar(
                base_time + timedelta(days=i),
                close=Decimal("100.00"),
                high=Decimal("105.00"),
                low=Decimal("95.00"),
                instrument_id=inst_id,
            )
            for i in range(15)
        ]
        ctx_15 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 15),
            market_bars=bars_15,
        )
        definition = FeatureDefinition(
            name="atr",
            category=FeatureCategory.TECHNICAL,
            description="14-period ATR",
            parameters={"window": 14},
        )
        result = calc.calculate(definition, ctx_15)
        assert result == Decimal("10")

    def test_completely_flat_prices_returns_zero(self) -> None:
        calc = ATRCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bars = [
            make_bar(
                base_time + timedelta(days=i),
                close=Decimal("100.00"),
                high=Decimal("100.00"),
                low=Decimal("100.00"),
                instrument_id=inst_id,
            )
            for i in range(15)
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 15),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="atr",
            category=FeatureCategory.TECHNICAL,
            description="ATR",
            parameters={"window": 14},
        )
        result = calc.calculate(definition, context)
        assert result == Decimal("0")

    def test_unsorted_bars_produce_same_result(self) -> None:
        calc = ATRCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        b0 = make_bar(base_time, close=Decimal("100.00"), high=Decimal("102.00"), low=Decimal("98.00"), instrument_id=inst_id)
        b1 = make_bar(base_time + timedelta(days=1), close=Decimal("104.00"), high=Decimal("106.00"), low=Decimal("100.00"), instrument_id=inst_id)
        b2 = make_bar(base_time + timedelta(days=2), close=Decimal("98.00"), high=Decimal("103.00"), low=Decimal("96.00"), instrument_id=inst_id)
        b3 = make_bar(base_time + timedelta(days=3), close=Decimal("105.00"), high=Decimal("108.00"), low=Decimal("102.00"), instrument_id=inst_id)

        definition = FeatureDefinition(
            name="atr",
            category=FeatureCategory.TECHNICAL,
            description="2-period ATR",
            parameters={"window": 2},
        )

        ctx_sorted = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 4),
            market_bars=[b0, b1, b2, b3],
        )
        # Scrambled order [b2, b0, b3, b1]
        ctx_unsorted = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 4),
            market_bars=[b2, b0, b3, b1],
        )

        assert calc.calculate(definition, ctx_unsorted) == calc.calculate(definition, ctx_sorted)
        assert calc.calculate(definition, ctx_unsorted) == Decimal("8.5")

    def test_input_context_and_bars_remain_unmodified(self) -> None:
        calc = ATRCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        b0 = make_bar(base_time, close=Decimal("100.00"), high=Decimal("102.00"), low=Decimal("98.00"), instrument_id=inst_id)
        b1 = make_bar(base_time + timedelta(days=1), close=Decimal("104.00"), high=Decimal("106.00"), low=Decimal("100.00"), instrument_id=inst_id)
        b2 = make_bar(base_time + timedelta(days=2), close=Decimal("98.00"), high=Decimal("103.00"), low=Decimal("96.00"), instrument_id=inst_id)

        original_tuple = (b2, b0, b1)
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 3),
            market_bars=original_tuple,
        )
        definition = FeatureDefinition(
            name="atr",
            category=FeatureCategory.TECHNICAL,
            description="2-period ATR",
            parameters={"window": 2},
        )
        calc.calculate(definition, context)

        # Context collection unchanged and in original order
        assert context.market_bars == original_tuple
        # Individual bar attributes unchanged
        assert b0.high == Decimal("102.00")
        assert b0.low == Decimal("98.00")
        assert b0.close == Decimal("100.00")

    @pytest.mark.parametrize("invalid_window", [0, -1, -14])
    def test_invalid_numeric_window_raises_value_error(self, invalid_window: int) -> None:
        calc = ATRCalculator()
        definition = FeatureDefinition(
            name="atr",
            category=FeatureCategory.TECHNICAL,
            description="ATR",
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
        calc = ATRCalculator()
        definition = FeatureDefinition(
            name="atr",
            category=FeatureCategory.TECHNICAL,
            description="ATR",
            parameters={"window": invalid_type_window},
        )
        context = FeatureCalculationContext(
            instrument_id=uuid4(),
            as_of_date=date(2026, 9, 1),
        )
        with pytest.raises(TypeError, match="must be an integer"):
            calc.calculate(definition, context)

    def test_output_is_decimal_and_non_negative(self) -> None:
        calc = ATRCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bars = [
            make_bar(
                base_time + timedelta(days=i),
                close=Decimal("100.00"),
                high=Decimal("105.00"),
                low=Decimal("95.00"),
                instrument_id=inst_id,
            )
            for i in range(20)
        ]
        context = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 20),
            market_bars=bars,
        )
        definition = FeatureDefinition(
            name="atr",
            category=FeatureCategory.TECHNICAL,
            description="14-period ATR",
            parameters={"window": 14},
        )
        result = calc.calculate(definition, context)
        assert isinstance(result, Decimal)
        assert result >= Decimal("0")
