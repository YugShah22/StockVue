from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from app.domain.entities.market_bar import MarketBar
from app.domain.features.calculators.bollinger_bands_calculator import BollingerBandsCalculator
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


class TestBollingerBandsCalculator:
    def test_canonical_feature_name(self) -> None:
        calc = BollingerBandsCalculator()
        assert calc.feature_name == "bollinger_bands"

    def test_default_parameters(self) -> None:
        calc = BollingerBandsCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # 19 bars -> returns None with default window of 20
        bars_19 = [
            make_bar(base_time + timedelta(days=i), Decimal(str(100 + i)), instrument_id=inst_id)
            for i in range(19)
        ]
        ctx_19 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 19),
            market_bars=bars_19,
        )
        def_empty = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="Default Bollinger Bands",
            parameters={},  # window, k, and band omitted
        )
        assert calc.calculate(def_empty, ctx_19) is None

        # 20 bars: closes 101 to 120 -> mean is (101 + 120) / 2 = 110.5
        bars_20 = [
            make_bar(base_time + timedelta(days=i), Decimal(str(101 + i)), instrument_id=inst_id)
            for i in range(20)
        ]
        ctx_20 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 20),
            market_bars=bars_20,
        )
        result = calc.calculate(def_empty, ctx_20)
        assert isinstance(result, Decimal)
        # Default band is "middle", which is the SMA
        expected_mean = sum(Decimal(str(101 + i)) for i in range(20)) / Decimal(20)
        assert result == expected_mean
        assert result == Decimal("110.5")

    def test_middle_upper_and_lower_bands(self) -> None:
        calc = BollingerBandsCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # 4 bars: closes [10, 20, 10, 20]
        # mean = 15
        # squared diffs = [25, 25, 25, 25] -> sum = 100
        # population variance = 100 / 4 = 25
        # std_dev = 5
        # k = 2
        # upper = 15 + 2*5 = 25
        # middle = 15
        # lower = 15 - 2*5 = 5
        closes = [Decimal("10"), Decimal("20"), Decimal("10"), Decimal("20")]
        bars = [
            make_bar(base_time + timedelta(days=i), closes[i], instrument_id=inst_id)
            for i in range(4)
        ]
        ctx = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 4),
            market_bars=bars,
        )

        def_mid = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="BB Middle",
            parameters={"window": 4, "k": 2, "band": "middle"},
        )
        def_upper = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="BB Upper",
            parameters={"window": 4, "k": 2, "band": "upper"},
        )
        def_lower = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="BB Lower",
            parameters={"window": 4, "k": 2, "band": "lower"},
        )

        assert calc.calculate(def_mid, ctx) == Decimal("15")
        assert calc.calculate(def_upper, ctx) == Decimal("25")
        assert calc.calculate(def_lower, ctx) == Decimal("5")

    def test_custom_window(self) -> None:
        calc = BollingerBandsCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # 6 bars: first 2 are old bars [999, 999], trailing 4 are [10, 20, 10, 20]
        closes = [Decimal("999"), Decimal("999"), Decimal("10"), Decimal("20"), Decimal("10"), Decimal("20")]
        bars = [
            make_bar(base_time + timedelta(days=i), closes[i], instrument_id=inst_id)
            for i in range(6)
        ]
        ctx = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 6),
            market_bars=bars,
        )

        def_bb = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="BB window 4",
            parameters={"window": 4, "k": Decimal("2"), "band": "upper"},
        )
        assert calc.calculate(def_bb, ctx) == Decimal("25")

    def test_custom_k_including_fractional_decimal(self) -> None:
        calc = BollingerBandsCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # 4 bars: closes [10, 20, 10, 20] -> mean=15, std_dev=5
        closes = [Decimal("10"), Decimal("20"), Decimal("10"), Decimal("20")]
        bars = [
            make_bar(base_time + timedelta(days=i), closes[i], instrument_id=inst_id)
            for i in range(4)
        ]
        ctx = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 4),
            market_bars=bars,
        )

        # Custom integer k = 3 -> upper = 15 + 3*5 = 30, lower = 15 - 3*5 = 0
        def_k3_upper = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="BB k=3",
            parameters={"window": 4, "k": 3, "band": "upper"},
        )
        def_k3_lower = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="BB k=3",
            parameters={"window": 4, "k": 3, "band": "lower"},
        )
        assert calc.calculate(def_k3_upper, ctx) == Decimal("30")
        assert calc.calculate(def_k3_lower, ctx) == Decimal("0")

        # Fractional Decimal k = Decimal("1.5") -> upper = 15 + 1.5*5 = 22.5, lower = 15 - 1.5*5 = 7.5
        def_k_frac_upper = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="BB k=1.5",
            parameters={"window": 4, "k": Decimal("1.5"), "band": "upper"},
        )
        def_k_frac_lower = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="BB k=1.5",
            parameters={"window": 4, "k": Decimal("1.5"), "band": "lower"},
        )
        assert calc.calculate(def_k_frac_upper, ctx) == Decimal("22.5")
        assert calc.calculate(def_k_frac_lower, ctx) == Decimal("7.5")

    def test_population_standard_deviation_distinguished_from_sample(self) -> None:
        calc = BollingerBandsCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # 4 bars: [10, 20, 10, 20]
        # Sum of squared deviations = (10-15)^2 + (20-15)^2 + (10-15)^2 + (20-15)^2 = 100
        # Population variance: 100 / 4 = 25 -> std_dev = sqrt(25) = 5
        # Sample variance:     100 / (4 - 1) = 100 / 3 = 33.333... -> sample std_dev = sqrt(100/3) ≈ 5.77350269...
        closes = [Decimal("10"), Decimal("20"), Decimal("10"), Decimal("20")]
        bars = [
            make_bar(base_time + timedelta(days=i), closes[i], instrument_id=inst_id)
            for i in range(4)
        ]
        ctx = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 4),
            market_bars=bars,
        )

        def_upper = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="Upper band",
            parameters={"window": 4, "k": 2, "band": "upper"},
        )

        result = calc.calculate(def_upper, ctx)
        assert result is not None

        # Population result: exactly 25
        assert result == Decimal("25")

        # Explicitly verify that the result does NOT match sample standard deviation
        sample_variance = Decimal("100") / Decimal("3")
        sample_std_dev = sample_variance.sqrt()
        sample_upper = Decimal("15") + (Decimal("2") * sample_std_dev)
        assert result != sample_upper

    def test_window_one_all_bands_equal_single_close(self) -> None:
        calc = BollingerBandsCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # Single bar
        single_bar = make_bar(base_time, Decimal("152.75"), instrument_id=inst_id)
        ctx = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 1),
            market_bars=[single_bar],
        )

        for band_name in ("middle", "upper", "lower"):
            defn = FeatureDefinition(
                name="bollinger_bands",
                category=FeatureCategory.TECHNICAL,
                description=f"BB 1d {band_name}",
                parameters={"window": 1, "k": 2, "band": band_name},
            )
            assert calc.calculate(defn, ctx) == Decimal("152.75")

        # Multiple bars with window=1: trailing bar close is 200.00
        bars = [
            make_bar(base_time, Decimal("100.00"), instrument_id=inst_id),
            make_bar(base_time + timedelta(days=1), Decimal("200.00"), instrument_id=inst_id),
        ]
        ctx_multi = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 2),
            market_bars=bars,
        )
        for band_name in ("middle", "upper", "lower"):
            defn = FeatureDefinition(
                name="bollinger_bands",
                category=FeatureCategory.TECHNICAL,
                description=f"BB 1d {band_name}",
                parameters={"window": 1, "k": Decimal("3"), "band": band_name},
            )
            assert calc.calculate(defn, ctx_multi) == Decimal("200.00")

    def test_constant_prices_produce_identical_bands(self) -> None:
        calc = BollingerBandsCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        constant_price = Decimal("88.50")
        bars = [
            make_bar(base_time + timedelta(days=i), constant_price, instrument_id=inst_id)
            for i in range(10)
        ]
        ctx = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 10),
            market_bars=bars,
        )

        for band_name in ("middle", "upper", "lower"):
            defn = FeatureDefinition(
                name="bollinger_bands",
                category=FeatureCategory.TECHNICAL,
                description=f"BB flat {band_name}",
                parameters={"window": 10, "k": Decimal("2"), "band": band_name},
            )
            assert calc.calculate(defn, ctx) == constant_price

    def test_insufficient_history_returns_none(self) -> None:
        calc = BollingerBandsCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # Window 5, 4 bars -> None
        bars_4 = [
            make_bar(base_time + timedelta(days=i), Decimal("100"), instrument_id=inst_id)
            for i in range(4)
        ]
        ctx_4 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 4),
            market_bars=bars_4,
        )
        defn_5 = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="5d BB",
            parameters={"window": 5},
        )
        assert calc.calculate(defn_5, ctx_4) is None

        # Window 1, 0 bars -> None
        ctx_0 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 1),
            market_bars=[],
        )
        defn_1 = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="1d BB",
            parameters={"window": 1},
        )
        assert calc.calculate(defn_1, ctx_0) is None

    def test_exact_history_returns_result(self) -> None:
        calc = BollingerBandsCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # Window 5, exactly 5 bars
        bars_5 = [
            make_bar(base_time + timedelta(days=i), Decimal("100"), instrument_id=inst_id)
            for i in range(5)
        ]
        ctx_5 = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 5),
            market_bars=bars_5,
        )
        defn_5 = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="5d BB",
            parameters={"window": 5},
        )
        assert calc.calculate(defn_5, ctx_5) == Decimal("100")

    def test_extra_bars_uses_trailing_window_only(self) -> None:
        calc = BollingerBandsCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # 10 bars: first 7 have close=1000, last 3 are [10, 20, 30]
        closes = [Decimal("1000")] * 7 + [Decimal("10"), Decimal("20"), Decimal("30")]
        bars = [
            make_bar(base_time + timedelta(days=i), closes[i], instrument_id=inst_id)
            for i in range(10)
        ]
        ctx = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 10),
            market_bars=bars,
        )

        # window 3 -> trailing [10, 20, 30], mean = 20
        defn_mid = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="3d BB mid",
            parameters={"window": 3, "band": "middle"},
        )
        assert calc.calculate(defn_mid, ctx) == Decimal("20")

    def test_unsorted_bars_sorted_chronologically(self) -> None:
        calc = BollingerBandsCalculator()
        inst_id = uuid4()
        t0 = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        # 5 bars with distinct timestamps:
        # Chronological closes: [100, 10, 20, 30, 40]
        bar0 = make_bar(t0, Decimal("100"), instrument_id=inst_id)
        bar1 = make_bar(t0 + timedelta(days=1), Decimal("10"), instrument_id=inst_id)
        bar2 = make_bar(t0 + timedelta(days=2), Decimal("20"), instrument_id=inst_id)
        bar3 = make_bar(t0 + timedelta(days=3), Decimal("30"), instrument_id=inst_id)
        bar4 = make_bar(t0 + timedelta(days=4), Decimal("40"), instrument_id=inst_id)

        # Pass in shuffled order where the last 3 elements are [bar0, bar1, bar2]
        # (closes 100, 10, 20), which would yield mean 130/3 ≈ 43.33 if selected without sorting.
        shuffled_bars = [bar4, bar3, bar0, bar1, bar2]
        shuffled_bars_copy = list(shuffled_bars)

        ctx = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 5),
            market_bars=shuffled_bars,
        )

        # Correct trailing window of 3 bars chronologically is [bar2, bar3, bar4] -> closes [20, 30, 40]
        # Mean = (20 + 30 + 40) / 3 = 30
        defn_mid = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="3d BB Middle",
            parameters={"window": 3, "k": 2, "band": "middle"},
        )
        result_mid = calc.calculate(defn_mid, ctx)
        assert result_mid == Decimal("30")

        # Upper band: mean 30, variance = ((20-30)^2 + (30-30)^2 + (40-30)^2)/3 = 200/3
        expected_variance = Decimal("200") / Decimal("3")
        expected_std_dev = expected_variance.sqrt()
        expected_upper = Decimal("30") + (Decimal("2") * expected_std_dev)

        defn_upper = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="3d BB Upper",
            parameters={"window": 3, "k": 2, "band": "upper"},
        )
        result_upper = calc.calculate(defn_upper, ctx)
        assert result_upper == expected_upper

        # Verify that the result differs from what naive unsorted trailing selection would yield
        unsorted_mean = (Decimal("100") + Decimal("10") + Decimal("20")) / Decimal("3")
        assert result_mid != unsorted_mean

        # Input order and context must remain completely unchanged
        assert shuffled_bars == shuffled_bars_copy
        assert ctx.market_bars == tuple(shuffled_bars_copy)
        assert ctx.market_bars[0] is bar4
        assert ctx.market_bars[-1] is bar2

    def test_input_sequence_and_context_unmodified(self) -> None:
        calc = BollingerBandsCalculator()
        inst_id = uuid4()
        t0 = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

        bar1 = make_bar(t0 + timedelta(days=2), Decimal("10"), instrument_id=inst_id)
        bar2 = make_bar(t0, Decimal("20"), instrument_id=inst_id)
        original_bars = [bar1, bar2]
        original_bars_copy = list(original_bars)

        ctx = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 3),
            market_bars=original_bars,
        )

        defn = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="2d BB",
            parameters={"window": 2},
        )
        calc.calculate(defn, ctx)

        # Input list and context tuple must remain completely untouched
        assert original_bars == original_bars_copy
        assert ctx.market_bars == tuple(original_bars_copy)
        assert ctx.market_bars[0] is bar1
        assert ctx.market_bars[1] is bar2

    def test_window_validation(self) -> None:
        calc = BollingerBandsCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
        bars = [make_bar(base_time, Decimal("100"), instrument_id=inst_id)]
        ctx = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 1),
            market_bars=bars,
        )

        # Zero or negative window raises ValueError
        for non_positive_window in (0, -1, -20):
            defn = FeatureDefinition(
                name="bollinger_bands",
                category=FeatureCategory.TECHNICAL,
                description="Invalid window",
                parameters={"window": non_positive_window},
            )
            with pytest.raises(ValueError, match="positive integer"):
                calc.calculate(defn, ctx)

        # Non-integer / boolean raises TypeError
        for non_int_window in (True, False, 20.0, "20", None, [20]):
            defn = FeatureDefinition(
                name="bollinger_bands",
                category=FeatureCategory.TECHNICAL,
                description="Invalid window type",
                parameters={"window": non_int_window},
            )
            with pytest.raises(TypeError, match="must be an integer"):
                calc.calculate(defn, ctx)

    def test_k_validation(self) -> None:
        calc = BollingerBandsCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
        bars = [make_bar(base_time, Decimal("100"), instrument_id=inst_id)]
        ctx = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 1),
            market_bars=bars,
        )

        # Zero or negative integer raises ValueError
        for non_positive_k_int in (0, -1, -5):
            defn = FeatureDefinition(
                name="bollinger_bands",
                category=FeatureCategory.TECHNICAL,
                description="Invalid k int",
                parameters={"k": non_positive_k_int},
            )
            with pytest.raises(ValueError, match="must be positive"):
                calc.calculate(defn, ctx)

        # Zero or negative Decimal raises ValueError
        for non_positive_k_dec in (Decimal("0"), Decimal("-1"), Decimal("-0.001")):
            defn = FeatureDefinition(
                name="bollinger_bands",
                category=FeatureCategory.TECHNICAL,
                description="Invalid k Decimal",
                parameters={"k": non_positive_k_dec},
            )
            with pytest.raises(ValueError, match="must be positive"):
                calc.calculate(defn, ctx)

        # Non-finite Decimal raises ValueError
        for non_finite_k in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
            defn = FeatureDefinition(
                name="bollinger_bands",
                category=FeatureCategory.TECHNICAL,
                description="Non-finite k Decimal",
                parameters={"k": non_finite_k},
            )
            with pytest.raises(ValueError, match="finite Decimal"):
                calc.calculate(defn, ctx)

        # Unsupported types raise TypeError
        for invalid_k_type in (True, False, 2.0, "2", None, [2]):
            defn = FeatureDefinition(
                name="bollinger_bands",
                category=FeatureCategory.TECHNICAL,
                description="Invalid k type",
                parameters={"k": invalid_k_type},
            )
            with pytest.raises(TypeError, match="integer or Decimal"):
                calc.calculate(defn, ctx)

    def test_band_validation(self) -> None:
        calc = BollingerBandsCalculator()
        inst_id = uuid4()
        base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
        bars = [make_bar(base_time, Decimal("100"), instrument_id=inst_id)]
        ctx = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=date(2026, 9, 1),
            market_bars=bars,
        )

        # Unsupported types raise TypeError
        for invalid_type in (1, True, False, 2.0, None, ["upper"]):
            defn = FeatureDefinition(
                name="bollinger_bands",
                category=FeatureCategory.TECHNICAL,
                description="Invalid band type",
                parameters={"band": invalid_type},
            )
            with pytest.raises(TypeError, match="must be a string"):
                calc.calculate(defn, ctx)

        # Invalid string values raise ValueError
        for invalid_band in ("UPPER", "Middle", "LOWER", "sma", "", "other", " upper "):
            defn = FeatureDefinition(
                name="bollinger_bands",
                category=FeatureCategory.TECHNICAL,
                description="Invalid band value",
                parameters={"band": invalid_band},
            )
            with pytest.raises(ValueError, match="must be one of"):
                calc.calculate(defn, ctx)
