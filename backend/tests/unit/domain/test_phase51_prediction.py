"""Unit tests for Phase 5.1 — TargetDefinition, TargetValue, and ForwardReturnCalculator."""

from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import pytest

from app.domain.entities.market_bar import MarketBar
from app.domain.prediction.calculators.forward_return_calculator import ForwardReturnCalculator
from app.domain.prediction.target_definition import TargetDefinition
from app.domain.prediction.target_value import TargetValue

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_bar(
    timestamp: datetime,
    close: Decimal,
    instrument_id: UUID | None = None,
) -> MarketBar:
    """Create a MarketBar with open = high = low = close for simplicity."""
    return MarketBar(
        instrument_id=instrument_id or uuid4(),
        timestamp=timestamp,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=Decimal("1000"),
    )


BASE_TIME = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)


def make_bars(
    closes: list[str],
    instrument_id: UUID | None = None,
    start: datetime | None = None,
    day_step: int = 1,
) -> list[MarketBar]:
    """Build a list of daily bars from a list of close-price strings."""
    inst = instrument_id or uuid4()
    t0 = start or BASE_TIME
    return [
        make_bar(t0 + timedelta(days=i * day_step), Decimal(c), instrument_id=inst)
        for i, c in enumerate(closes)
    ]


# ===========================================================================
# TargetDefinition Tests
# ===========================================================================


class TestTargetDefinition:
    def test_valid_creation_with_required_fields(self) -> None:
        defn = TargetDefinition(name="forward_return", horizon_bars=5)
        assert defn.name == "forward_return"
        assert defn.horizon_bars == 5
        assert defn.description == ""
        assert defn.parameters == {}

    def test_name_is_stripped(self) -> None:
        defn = TargetDefinition(name="  forward_return  ", horizon_bars=5)
        assert defn.name == "forward_return"

    def test_name_empty_after_strip_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            TargetDefinition(name="   ", horizon_bars=5)

    def test_name_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            TargetDefinition(name="", horizon_bars=5)

    def test_name_non_str_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="must be a str"):
            TargetDefinition(name=42, horizon_bars=5)  # type: ignore[arg-type]

    def test_valid_horizon_one(self) -> None:
        defn = TargetDefinition(name="forward_return", horizon_bars=1)
        assert defn.horizon_bars == 1

    def test_valid_horizon_five(self) -> None:
        defn = TargetDefinition(name="forward_return", horizon_bars=5)
        assert defn.horizon_bars == 5

    def test_valid_horizon_twenty(self) -> None:
        defn = TargetDefinition(name="forward_return", horizon_bars=20)
        assert defn.horizon_bars == 20

    def test_horizon_zero_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            TargetDefinition(name="forward_return", horizon_bars=0)

    def test_horizon_negative_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            TargetDefinition(name="forward_return", horizon_bars=-5)

    def test_horizon_bool_true_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="got bool"):
            TargetDefinition(name="forward_return", horizon_bars=True)

    def test_horizon_bool_false_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="got bool"):
            TargetDefinition(name="forward_return", horizon_bars=False)

    def test_horizon_float_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="must be an integer"):
            TargetDefinition(name="forward_return", horizon_bars=5.0)  # type: ignore[arg-type]

    def test_horizon_string_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="must be an integer"):
            TargetDefinition(name="forward_return", horizon_bars="5")  # type: ignore[arg-type]

    def test_description_optional(self) -> None:
        defn = TargetDefinition(name="forward_return", horizon_bars=5, description="5-day return")
        assert defn.description == "5-day return"

    def test_parameters_optional(self) -> None:
        defn = TargetDefinition(
            name="forward_return", horizon_bars=5, parameters={"adjusted": True}
        )
        assert defn.parameters == {"adjusted": True}

    def test_parameters_defaults_to_empty_dict(self) -> None:
        defn = TargetDefinition(name="forward_return", horizon_bars=5)
        assert defn.parameters == {}

    def test_parameters_defensively_copied(self) -> None:
        original = {"key": "val"}
        defn = TargetDefinition(name="forward_return", horizon_bars=5, parameters=original)
        original["extra"] = "mutated"
        assert "extra" not in defn.parameters

    def test_parameters_non_dict_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="must be a dict"):
            TargetDefinition(name="forward_return", horizon_bars=5, parameters="bad")  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        defn = TargetDefinition(name="forward_return", horizon_bars=5)
        with pytest.raises(FrozenInstanceError):
            defn.horizon_bars = 10  # type: ignore[misc]


# ===========================================================================
# TargetValue Tests
# ===========================================================================


class TestTargetValue:
    def _make_defn(self) -> TargetDefinition:
        return TargetDefinition(name="forward_return", horizon_bars=5)

    def test_valid_creation(self) -> None:
        defn = self._make_defn()
        inst_id = uuid4()
        tv = TargetValue(
            target=defn,
            instrument_id=inst_id,
            observation_date=date(2026, 1, 7),
            value=Decimal("0.0345"),
            realization_date=date(2026, 1, 14),
        )
        assert tv.target is defn
        assert tv.instrument_id == inst_id
        assert tv.observation_date == date(2026, 1, 7)
        assert tv.value == Decimal("0.0345")
        assert tv.realization_date == date(2026, 1, 14)

    def test_target_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError, match="TargetDefinition instance"):
            TargetValue(
                target="not_a_definition",  # type: ignore[arg-type]
                instrument_id=uuid4(),
                observation_date=date(2026, 1, 7),
                value=Decimal("0.05"),
                realization_date=date(2026, 1, 14),
            )

    def test_instrument_id_wrong_type_raises(self) -> None:
        defn = self._make_defn()
        with pytest.raises(TypeError, match="UUID instance"):
            TargetValue(
                target=defn,
                instrument_id="not-a-uuid",  # type: ignore[arg-type]
                observation_date=date(2026, 1, 7),
                value=Decimal("0.05"),
                realization_date=date(2026, 1, 14),
            )

    def test_observation_date_as_datetime_raises(self) -> None:
        defn = self._make_defn()
        with pytest.raises(TypeError, match="must be a date, not a datetime"):
            TargetValue(
                target=defn,
                instrument_id=uuid4(),
                observation_date=datetime(2026, 1, 7, tzinfo=UTC),
                value=Decimal("0.05"),
                realization_date=date(2026, 1, 14),
            )

    def test_value_wrong_type_raises(self) -> None:
        defn = self._make_defn()
        with pytest.raises(TypeError, match="Decimal instance"):
            TargetValue(
                target=defn,
                instrument_id=uuid4(),
                observation_date=date(2026, 1, 7),
                value=0.05,  # type: ignore[arg-type]
                realization_date=date(2026, 1, 14),
            )

    def test_realization_date_as_datetime_raises(self) -> None:
        defn = self._make_defn()
        with pytest.raises(TypeError, match="must be a date, not a datetime"):
            TargetValue(
                target=defn,
                instrument_id=uuid4(),
                observation_date=date(2026, 1, 7),
                value=Decimal("0.05"),
                realization_date=datetime(2026, 1, 14, tzinfo=UTC),
            )

    def test_realization_date_same_as_observation_raises(self) -> None:
        defn = self._make_defn()
        with pytest.raises(ValueError, match="strictly after"):
            TargetValue(
                target=defn,
                instrument_id=uuid4(),
                observation_date=date(2026, 1, 7),
                value=Decimal("0.05"),
                realization_date=date(2026, 1, 7),
            )

    def test_realization_date_before_observation_raises(self) -> None:
        defn = self._make_defn()
        with pytest.raises(ValueError, match="strictly after"):
            TargetValue(
                target=defn,
                instrument_id=uuid4(),
                observation_date=date(2026, 1, 7),
                value=Decimal("0.05"),
                realization_date=date(2026, 1, 6),
            )

    def test_is_frozen(self) -> None:
        defn = self._make_defn()
        tv = TargetValue(
            target=defn,
            instrument_id=uuid4(),
            observation_date=date(2026, 1, 7),
            value=Decimal("0.05"),
            realization_date=date(2026, 1, 14),
        )
        with pytest.raises(FrozenInstanceError):
            tv.value = Decimal("0.1")  # type: ignore[misc]


# ===========================================================================
# ForwardReturnCalculator Tests
# ===========================================================================


class TestForwardReturnCalculator:
    """All tests use UTC timestamps and hand-calculated expected values."""

    def _make_defn(self, horizon: int = 5) -> TargetDefinition:
        return TargetDefinition(name="forward_return", horizon_bars=horizon)

    # -----------------------------------------------------------------------
    # 1. Correct forward return using known close prices
    # -----------------------------------------------------------------------

    def test_correct_return_value_h5(self) -> None:
        """Bars [100, 102, 104, 106, 108, 110]: obs=bar0(100), future=bar5(110).
        R = 110/100 - 1 = 0.10 exactly.
        """
        inst_id = uuid4()
        bars = make_bars(
            ["100", "102", "104", "106", "108", "110"],
            instrument_id=inst_id,
        )
        defn = self._make_defn(horizon=5)
        calc = ForwardReturnCalculator()
        obs_date = bars[0].timestamp.date()

        result = calc.calculate(defn, inst_id, bars, obs_date)

        assert result is not None
        assert result.value == Decimal("110") / Decimal("100") - Decimal("1")
        assert result.value == Decimal("0.10")

    def test_correct_return_value_h1(self) -> None:
        """Bars [200, 210]: obs=bar0(200), future=bar1(210). R = 0.05."""
        inst_id = uuid4()
        bars = make_bars(["200", "210"], instrument_id=inst_id)
        defn = self._make_defn(horizon=1)
        calc = ForwardReturnCalculator()
        obs_date = bars[0].timestamp.date()

        result = calc.calculate(defn, inst_id, bars, obs_date)

        assert result is not None
        assert result.value == Decimal("210") / Decimal("200") - Decimal("1")
        assert result.value == Decimal("0.05")

    def test_correct_return_value_h3(self) -> None:
        """Bars [100, 105, 110, 120]: obs=bar0(100), future=bar3(120). R = 0.20."""
        inst_id = uuid4()
        bars = make_bars(["100", "105", "110", "120"], instrument_id=inst_id)
        defn = self._make_defn(horizon=3)
        calc = ForwardReturnCalculator()
        obs_date = bars[0].timestamp.date()

        result = calc.calculate(defn, inst_id, bars, obs_date)

        assert result is not None
        assert result.value == Decimal("120") / Decimal("100") - Decimal("1")

    def test_correct_return_negative(self) -> None:
        """Negative return: obs=150, future=120. R = 120/150 - 1 = -0.2."""
        inst_id = uuid4()
        bars = make_bars(["150", "140", "130", "125", "122", "120"], instrument_id=inst_id)
        defn = self._make_defn(horizon=5)
        calc = ForwardReturnCalculator()
        obs_date = bars[0].timestamp.date()

        result = calc.calculate(defn, inst_id, bars, obs_date)

        assert result is not None
        assert result.value == Decimal("120") / Decimal("150") - Decimal("1")

    # -----------------------------------------------------------------------
    # 2. Correct realization date
    # -----------------------------------------------------------------------

    def test_realization_date_is_date_of_future_bar(self) -> None:
        """Realization date must be the date of bar at obs_idx + h."""
        inst_id = uuid4()
        bars = make_bars(
            ["100", "101", "102", "103", "104", "105"],
            instrument_id=inst_id,
            start=datetime(2026, 1, 5, 10, 0, tzinfo=UTC),
        )
        defn = self._make_defn(horizon=5)
        calc = ForwardReturnCalculator()
        obs_date = bars[0].timestamp.date()  # 2026-01-05

        result = calc.calculate(defn, inst_id, bars, obs_date)

        assert result is not None
        assert result.realization_date == bars[5].timestamp.date()  # 2026-01-10

    # -----------------------------------------------------------------------
    # 3. Correct instrument_id and observation_date in TargetValue
    # -----------------------------------------------------------------------

    def test_target_value_fields(self) -> None:
        inst_id = uuid4()
        bars = make_bars(["50", "55", "60", "65", "70", "75"], instrument_id=inst_id)
        defn = self._make_defn(horizon=5)
        calc = ForwardReturnCalculator()
        obs_date = bars[0].timestamp.date()

        result = calc.calculate(defn, inst_id, bars, obs_date)

        assert result is not None
        assert result.target is defn
        assert result.instrument_id == inst_id
        assert result.observation_date == obs_date
        assert result.realization_date > result.observation_date

    # -----------------------------------------------------------------------
    # 4. Insufficient future history → None
    # -----------------------------------------------------------------------

    def test_insufficient_future_bars_returns_none(self) -> None:
        """h=5 but only 5 bars total; obs is bar0 → only 4 future bars, need 5."""
        inst_id = uuid4()
        bars = make_bars(["100", "101", "102", "103", "104"], instrument_id=inst_id)
        defn = self._make_defn(horizon=5)
        calc = ForwardReturnCalculator()

        result = calc.calculate(defn, inst_id, bars, bars[0].timestamp.date())

        assert result is None

    def test_exactly_enough_future_bars(self) -> None:
        """h=5, obs=bar0, exactly 6 bars total → bar5 is the realization bar → succeeds."""
        inst_id = uuid4()
        bars = make_bars(["100", "101", "102", "103", "104", "105"], instrument_id=inst_id)
        defn = self._make_defn(horizon=5)
        calc = ForwardReturnCalculator()

        result = calc.calculate(defn, inst_id, bars, bars[0].timestamp.date())

        assert result is not None
        assert result.value == Decimal("105") / Decimal("100") - Decimal("1")

    def test_observation_at_end_has_no_future_bars(self) -> None:
        """Observation date is the last available bar → insufficient future history."""
        inst_id = uuid4()
        bars = make_bars(["100", "101", "102", "103", "104", "105"], instrument_id=inst_id)
        defn = self._make_defn(horizon=5)
        calc = ForwardReturnCalculator()

        result = calc.calculate(defn, inst_id, bars, bars[-1].timestamp.date())

        assert result is None

    # -----------------------------------------------------------------------
    # 5. Missing observation date → None
    # -----------------------------------------------------------------------

    def test_missing_observation_date_returns_none(self) -> None:
        inst_id = uuid4()
        bars = make_bars(["100", "101", "102", "103", "104", "105"], instrument_id=inst_id)
        calc = ForwardReturnCalculator()
        defn = self._make_defn(horizon=5)

        missing_date = date(2025, 6, 15)
        result = calc.calculate(defn, inst_id, bars, missing_date)

        assert result is None

    # -----------------------------------------------------------------------
    # 6. Weekend / calendar gaps don't change trading-bar count
    # -----------------------------------------------------------------------

    def test_calendar_gaps_do_not_affect_trading_bar_count(self) -> None:
        """Bars spaced 7 days apart (weekly). h=3 means 3 *supplied* bars forward."""
        inst_id = uuid4()
        start = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)  # Monday
        # Timestamps: Jan 5, Jan 12, Jan 19, Jan 26 (weekly intervals)
        bars = [
            make_bar(start + timedelta(weeks=i), Decimal(str(100 + i * 10)), instrument_id=inst_id)
            for i in range(5)
        ]
        defn = self._make_defn(horizon=3)
        calc = ForwardReturnCalculator()
        obs_date = bars[0].timestamp.date()  # Jan 5

        result = calc.calculate(defn, inst_id, bars, obs_date)

        # h=3 means bar at obs_idx + 3 = bar[3] (Jan 26, close=130)
        assert result is not None
        assert result.value == Decimal("130") / Decimal("100") - Decimal("1")
        assert result.realization_date == bars[3].timestamp.date()

    # -----------------------------------------------------------------------
    # 7. Unsorted input produces correct result
    # -----------------------------------------------------------------------

    def test_unsorted_input_produces_correct_result(self) -> None:
        """Pass bars in reversed order; result must use correct chronological ordering."""
        inst_id = uuid4()
        ordered_bars = make_bars(
            ["100", "105", "110", "115", "120", "130"],
            instrument_id=inst_id,
        )
        # Observation = bar0 (Jan 1, close=100), h=5 → bar5 (Jan 6, close=130)
        expected_obs_date = ordered_bars[0].timestamp.date()
        defn = self._make_defn(horizon=5)
        calc = ForwardReturnCalculator()

        # Shuffle the input: bars in reversed chronological order
        shuffled = list(reversed(ordered_bars))
        shuffled_copy = list(shuffled)

        result = calc.calculate(defn, inst_id, shuffled, expected_obs_date)

        # Correct result regardless of input order
        assert result is not None
        assert result.value == Decimal("130") / Decimal("100") - Decimal("1")
        # Input sequence must not be mutated
        assert shuffled == shuffled_copy

    # -----------------------------------------------------------------------
    # 8. Input sequence and its ordering remain unchanged
    # -----------------------------------------------------------------------

    def test_input_sequence_unchanged_after_calculate(self) -> None:
        inst_id = uuid4()
        bars = make_bars(["100", "105", "110", "115", "120", "130"], instrument_id=inst_id)
        original_order = [b.timestamp for b in bars]
        calc = ForwardReturnCalculator()
        defn = self._make_defn(horizon=5)

        calc.calculate(defn, inst_id, bars, bars[0].timestamp.date())

        assert [b.timestamp for b in bars] == original_order

    # -----------------------------------------------------------------------
    # 9. Zero or negative observation close → None
    # -----------------------------------------------------------------------

    def test_zero_observation_close_returns_none(self) -> None:
        """MarketBar validates close >= 0, so we patch via allowed zero price path."""
        # MarketBar raises for close < 0 but allows close == 0 when open/high/low = 0
        inst_id = uuid4()
        zero_bar = MarketBar(
            instrument_id=inst_id,
            timestamp=BASE_TIME,
            open=Decimal("0"),
            high=Decimal("0"),
            low=Decimal("0"),
            close=Decimal("0"),
            volume=Decimal("0"),
        )
        future_bars = make_bars(
            ["100", "101", "102", "103", "104"],
            instrument_id=inst_id,
            start=BASE_TIME + timedelta(days=1),
        )
        bars = [zero_bar] + future_bars
        calc = ForwardReturnCalculator()
        defn = self._make_defn(horizon=5)

        result = calc.calculate(defn, inst_id, bars, zero_bar.timestamp.date())

        assert result is None

    # -----------------------------------------------------------------------
    # 10. A future bar cannot accidentally become the observation bar
    # -----------------------------------------------------------------------

    def test_future_bar_does_not_become_observation_bar(self) -> None:
        """Ask for observation_date that is in the future relative to available bars."""
        inst_id = uuid4()
        bars = make_bars(
            ["100", "105", "110", "115", "120", "130"],
            instrument_id=inst_id,
            start=datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
        )
        # The bars only cover Jan 1 to Jan 6; requesting Jan 10 should return None
        future_obs_date = date(2026, 1, 10)
        calc = ForwardReturnCalculator()
        defn = self._make_defn(horizon=5)

        result = calc.calculate(defn, inst_id, bars, future_obs_date)

        assert result is None

    # -----------------------------------------------------------------------
    # 11. Duplicate / ambiguous observation records raise ValueError
    # -----------------------------------------------------------------------

    def test_duplicate_observation_date_raises_value_error(self) -> None:
        """Two bars with the same date but different times must raise ValueError."""
        inst_id = uuid4()
        obs_day = datetime(2026, 1, 1, tzinfo=UTC)
        bar_a = make_bar(obs_day.replace(hour=9), Decimal("100"), instrument_id=inst_id)
        bar_b = make_bar(obs_day.replace(hour=15), Decimal("105"), instrument_id=inst_id)
        future_bars = make_bars(
            ["110", "111", "112", "113", "114"],
            instrument_id=inst_id,
            start=datetime(2026, 1, 2, 10, 0, tzinfo=UTC),
        )
        bars = [bar_a, bar_b] + future_bars
        calc = ForwardReturnCalculator()
        defn = self._make_defn(horizon=5)

        with pytest.raises(ValueError, match="2 bars found for observation_date"):
            calc.calculate(defn, inst_id, bars, obs_day.date())

    # -----------------------------------------------------------------------
    # 12. Decimal arithmetic — no float intermediaries
    # -----------------------------------------------------------------------

    def test_decimal_arithmetic_no_float(self) -> None:
        """Verify result uses pure Decimal arithmetic (no float intermediaries).

        301/300 - 1 is non-terminating; any float path would produce a different
        bit pattern.  We verify the result equals the pure Decimal calculation.
        """
        inst_id = uuid4()
        close_obs = Decimal("300")
        close_future = Decimal("301")
        bars = make_bars(
            [str(close_obs)] + ["300"] * 4 + [str(close_future)],
            instrument_id=inst_id,
        )
        calc = ForwardReturnCalculator()
        defn = self._make_defn(horizon=5)

        result = calc.calculate(defn, inst_id, bars, bars[0].timestamp.date())

        assert result is not None
        assert isinstance(result.value, Decimal)
        expected = close_future / close_obs - Decimal("1")
        assert result.value == expected
        # Verify it does NOT equal the float-rounded version
        float_approx = Decimal(str(float(close_future) / float(close_obs) - 1.0))
        assert result.value != float_approx

    # -----------------------------------------------------------------------
    # 13. Middle observation in a longer series
    # -----------------------------------------------------------------------

    def test_observation_in_middle_of_series(self) -> None:
        """Use the 3rd bar as observation; check h bars forward from there."""
        inst_id = uuid4()
        # 10 bars: closes 100..109
        bars = make_bars([str(100 + i) for i in range(10)], instrument_id=inst_id)
        defn = self._make_defn(horizon=3)
        calc = ForwardReturnCalculator()
        obs_date = bars[2].timestamp.date()  # bar index 2, close=102

        result = calc.calculate(defn, inst_id, bars, obs_date)

        assert result is not None
        # h=3: future bar is at index 2+3=5, close=105
        assert result.value == Decimal("105") / Decimal("102") - Decimal("1")
        assert result.realization_date == bars[5].timestamp.date()

    # -----------------------------------------------------------------------
    # 14. ForwardReturnCalculator produces correct TargetValue type
    # -----------------------------------------------------------------------

    def test_result_is_target_value_instance(self) -> None:
        inst_id = uuid4()
        bars = make_bars(["100", "101", "102", "103", "104", "105"], instrument_id=inst_id)
        calc = ForwardReturnCalculator()
        defn = self._make_defn(horizon=5)

        result = calc.calculate(defn, inst_id, bars, bars[0].timestamp.date())

        assert isinstance(result, TargetValue)

    # -----------------------------------------------------------------------
    # 15. Horizon h=5 vs h=1 on same data produces different results
    # -----------------------------------------------------------------------

    def test_different_horizons_produce_different_results(self) -> None:
        inst_id = uuid4()
        bars = make_bars(
            ["100", "102", "104", "106", "108", "110"],
            instrument_id=inst_id,
        )
        calc = ForwardReturnCalculator()
        obs_date = bars[0].timestamp.date()

        result_h1 = calc.calculate(self._make_defn(horizon=1), inst_id, bars, obs_date)
        result_h5 = calc.calculate(self._make_defn(horizon=5), inst_id, bars, obs_date)

        assert result_h1 is not None
        assert result_h5 is not None
        # h=1: 102/100-1 = 0.02;  h=5: 110/100-1 = 0.10
        assert result_h1.value == Decimal("0.02")
        assert result_h5.value == Decimal("0.10")

    # -----------------------------------------------------------------------
    # 16. Realization date is strictly after observation date (TargetValue invariant)
    # -----------------------------------------------------------------------

    def test_realization_date_strictly_after_observation_date(self) -> None:
        inst_id = uuid4()
        bars = make_bars(["100", "101", "102", "103", "104", "105"], instrument_id=inst_id)
        calc = ForwardReturnCalculator()
        defn = self._make_defn(horizon=5)

        result = calc.calculate(defn, inst_id, bars, bars[0].timestamp.date())

        assert result is not None
        assert result.realization_date > result.observation_date

    # -----------------------------------------------------------------------
    # 17. Cross-instrument isolation regression tests
    # -----------------------------------------------------------------------

    def test_interleaved_instruments_uses_only_target_instrument_bars(self) -> None:
        """Interleave bars from two instruments; prove calculation for target
        instrument uses strictly its own bars and ignores the other instrument.
        """
        inst_a = uuid4()
        inst_b = uuid4()
        calc = ForwardReturnCalculator()
        defn = self._make_defn(horizon=2)

        # inst_a: Day 1 (100), Day 3 (110), Day 5 (120) -> h=2 forward from Day 1 is Day 5 (120)
        # Expected return: 120/100 - 1 = +0.20
        # inst_b: Day 2 (500), Day 4 (600)
        t_base = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        bar_a1 = make_bar(t_base, Decimal("100"), instrument_id=inst_a)
        bar_b1 = make_bar(t_base + timedelta(days=1), Decimal("500"), instrument_id=inst_b)
        bar_a2 = make_bar(t_base + timedelta(days=2), Decimal("110"), instrument_id=inst_a)
        bar_b2 = make_bar(t_base + timedelta(days=3), Decimal("600"), instrument_id=inst_b)
        bar_a3 = make_bar(t_base + timedelta(days=4), Decimal("120"), instrument_id=inst_a)

        # Interleaved input
        interleaved = [bar_a1, bar_b1, bar_a2, bar_b2, bar_a3]

        result = calc.calculate(defn, inst_a, interleaved, bar_a1.timestamp.date())

        assert result is not None
        assert result.instrument_id == inst_a
        assert result.observation_date == bar_a1.timestamp.date()
        assert result.realization_date == bar_a3.timestamp.date()
        assert result.value == Decimal("0.20")

    def test_other_instrument_on_observation_date_does_not_trigger_duplicate_error(
        self,
    ) -> None:
        """Another instrument having a bar on the same observation date must not
        trigger a duplicate-date ValueError for the target instrument.
        """
        inst_a = uuid4()
        inst_b = uuid4()
        calc = ForwardReturnCalculator()
        defn = self._make_defn(horizon=1)

        t_obs = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
        bar_a_obs = make_bar(t_obs, Decimal("100"), instrument_id=inst_a)
        bar_b_obs = make_bar(t_obs, Decimal("999"), instrument_id=inst_b)  # Same date, other inst
        bar_a_fut = make_bar(t_obs + timedelta(days=1), Decimal("110"), instrument_id=inst_a)

        bars = [bar_a_obs, bar_b_obs, bar_a_fut]

        result = calc.calculate(defn, inst_a, bars, t_obs.date())

        assert result is not None
        assert result.instrument_id == inst_a
        assert result.value == Decimal("0.10")

    def test_horizon_counted_using_only_target_instrument_bars(self) -> None:
        """Target instrument has 3 bars; unrelated instrument has 10 bars.
        For horizon=5, target instrument has insufficient bars -> must return None
        even though combined list has 13 bars total.
        """
        inst_a = uuid4()
        inst_b = uuid4()
        calc = ForwardReturnCalculator()
        defn = self._make_defn(horizon=5)

        # inst_a has only 3 bars total
        bars_a = make_bars(["100", "105", "110"], instrument_id=inst_a)
        # inst_b has 10 bars
        bars_b = make_bars([str(200 + i) for i in range(10)], instrument_id=inst_b)

        combined = bars_a + bars_b

        result = calc.calculate(defn, inst_a, combined, bars_a[0].timestamp.date())

        assert result is None

    def test_interleaved_input_ordering_remains_unchanged(self) -> None:
        """Caller's interleaved sequence order and contents are preserved."""
        inst_a = uuid4()
        inst_b = uuid4()
        calc = ForwardReturnCalculator()
        defn = self._make_defn(horizon=2)

        bars = [
            make_bar(datetime(2026, 1, 3, tzinfo=UTC), Decimal("100"), instrument_id=inst_a),
            make_bar(datetime(2026, 1, 1, tzinfo=UTC), Decimal("500"), instrument_id=inst_b),
            make_bar(datetime(2026, 1, 2, tzinfo=UTC), Decimal("110"), instrument_id=inst_a),
        ]
        original_ids = [b.market_bar_id for b in bars]

        calc.calculate(defn, inst_a, bars, date(2026, 1, 2))

        assert [b.market_bar_id for b in bars] == original_ids

    def test_no_bars_for_requested_instrument_returns_none(self) -> None:
        """When supplied bars contain zero bars for the requested instrument, return None."""
        inst_a = uuid4()
        inst_b = uuid4()
        calc = ForwardReturnCalculator()
        defn = self._make_defn(horizon=1)

        bars_b = make_bars(["100", "105", "110"], instrument_id=inst_b)

        result = calc.calculate(defn, inst_a, bars_b, bars_b[0].timestamp.date())

        assert result is None

    # -----------------------------------------------------------------------
    # 18. Market-date & timezone conventions regression tests
    # -----------------------------------------------------------------------

    def test_non_utc_timezone_matches_local_market_observation_date(self) -> None:
        """Exchange-local daily timestamps (e.g. IST = UTC+05:30) must match their
        local calendar date directly without being shifted backwards by UTC conversion.

        2026-01-05 00:00:00+05:30 in UTC is 2026-01-04 18:30:00 UTC.
        The market date is 2026-01-05 (Monday). Under pure bar.timestamp.date(),
        this correctly matches observation_date = date(2026, 1, 5).
        """
        ist = ZoneInfo("Asia/Kolkata")
        inst_id = uuid4()
        calc = ForwardReturnCalculator()
        defn = self._make_defn(horizon=1)

        # Midnight IST timestamps
        t_obs = datetime(2026, 1, 5, 0, 0, tzinfo=ist)
        t_fut = datetime(2026, 1, 6, 0, 0, tzinfo=ist)

        bar_obs = make_bar(t_obs, Decimal("100"), instrument_id=inst_id)
        bar_fut = make_bar(t_fut, Decimal("105"), instrument_id=inst_id)

        result = calc.calculate(defn, inst_id, [bar_obs, bar_fut], date(2026, 1, 5))

        assert result is not None
        assert result.observation_date == date(2026, 1, 5)
        assert result.realization_date == date(2026, 1, 6)
        assert result.value == Decimal("0.05")

    def test_equivalent_instant_timestamps_match_local_market_date(self) -> None:
        """Verify the interaction between equivalent instants, timezones, and calendar dates.

        1. Daytime case: 10:00 UTC == 15:30 IST (same instant AND same calendar date 2026-01-05).
           Bars created in either timezone both match observation_date = date(2026, 1, 5)
           because both share the same calendar date.

        2. Midnight boundary case: 2026-01-05 00:00:00+05:30 == 2026-01-04 18:30:00 UTC.
           These represent the exact same instant, but their calendar dates differ
           (2026-01-05 vs 2026-01-04). ForwardReturnCalculator uses bar.timestamp.date()
           directly without timezone conversion, so:
           - The IST bar matches date(2026, 1, 5).
           - The UTC bar matches date(2026, 1, 4), and returns None for date(2026, 1, 5).
        """
        ist = ZoneInfo("Asia/Kolkata")
        inst_id = uuid4()
        calc = ForwardReturnCalculator()
        defn = self._make_defn(horizon=1)

        # 1. Daytime market close: same instant AND same calendar date
        t_utc = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
        t_ist = datetime(2026, 1, 5, 15, 30, tzinfo=ist)
        assert t_utc == t_ist
        assert t_utc.date() == t_ist.date() == date(2026, 1, 5)

        # Both IST and UTC daytime bars match date(2026, 1, 5)
        bars_ist = [
            make_bar(t_ist, Decimal("200"), instrument_id=inst_id),
            make_bar(t_ist + timedelta(days=1), Decimal("210"), instrument_id=inst_id),
        ]
        res_ist = calc.calculate(defn, inst_id, bars_ist, date(2026, 1, 5))
        assert res_ist is not None
        assert res_ist.observation_date == date(2026, 1, 5)
        assert res_ist.realization_date == date(2026, 1, 6)
        assert res_ist.value == Decimal("0.05")

        bars_utc = [
            make_bar(t_utc, Decimal("200"), instrument_id=inst_id),
            make_bar(t_utc + timedelta(days=1), Decimal("210"), instrument_id=inst_id),
        ]
        res_utc = calc.calculate(defn, inst_id, bars_utc, date(2026, 1, 5))
        assert res_utc is not None
        assert res_utc.observation_date == date(2026, 1, 5)
        assert res_utc.realization_date == date(2026, 1, 6)
        assert res_utc.value == Decimal("0.05")

        # 2. Midnight boundary case: same instant BUT different calendar dates
        t_ist_midnight = datetime(2026, 1, 5, 0, 0, tzinfo=ist)
        t_utc_equivalent = datetime(2026, 1, 4, 18, 30, tzinfo=UTC)
        assert t_ist_midnight == t_utc_equivalent
        assert t_ist_midnight.date() == date(2026, 1, 5)
        assert t_utc_equivalent.date() == date(2026, 1, 4)

        # IST midnight bar matches Jan 5 observation date
        res_ist_mid = calc.calculate(
            defn,
            inst_id,
            [
                make_bar(t_ist_midnight, Decimal("100"), instrument_id=inst_id),
                make_bar(t_ist_midnight + timedelta(days=1), Decimal("105"), instrument_id=inst_id),
            ],
            date(2026, 1, 5),
        )
        assert res_ist_mid is not None
        assert res_ist_mid.observation_date == date(2026, 1, 5)

        # UTC equivalent bar matches Jan 4, NOT Jan 5 (returns None because Jan 5 has no bar)
        res_utc_mid = calc.calculate(
            defn,
            inst_id,
            [
                make_bar(t_utc_equivalent, Decimal("100"), instrument_id=inst_id),
                make_bar(
                    t_utc_equivalent + timedelta(days=1), Decimal("105"), instrument_id=inst_id
                ),
            ],
            date(2026, 1, 5),
        )
        assert res_utc_mid is None

        # But it DOES match when querying date(2026, 1, 4)
        res_utc_jan4 = calc.calculate(
            defn,
            inst_id,
            [
                make_bar(t_utc_equivalent, Decimal("100"), instrument_id=inst_id),
                make_bar(
                    t_utc_equivalent + timedelta(days=1), Decimal("105"), instrument_id=inst_id
                ),
            ],
            date(2026, 1, 4),
        )
        assert res_utc_jan4 is not None
        assert res_utc_jan4.observation_date == date(2026, 1, 4)

    def test_naive_timestamps_support(self) -> None:
        """When input market bars carry naive datetimes (e.g. from fixtures or test beds),
        bar.timestamp.date() resolves without error and matches observation_date.
        """
        inst_id = uuid4()
        calc = ForwardReturnCalculator()
        defn = self._make_defn(horizon=1)

        t_obs = datetime(2026, 1, 5, 10, 0)  # naive
        t_fut = datetime(2026, 1, 6, 10, 0)  # naive

        bar_obs = make_bar(t_obs, Decimal("100"), instrument_id=inst_id)
        bar_fut = make_bar(t_fut, Decimal("108"), instrument_id=inst_id)

        result = calc.calculate(defn, inst_id, [bar_obs, bar_fut], date(2026, 1, 5))

        assert result is not None
        assert result.observation_date == date(2026, 1, 5)
        assert result.realization_date == date(2026, 1, 6)
        assert result.value == Decimal("0.08")

    def test_database_persisted_utc_bars_workflow(self) -> None:
        """Bars persisted in PostgreSQL TIMESTAMPTZ always return in UTC.
        A trading session occurring during market hours (e.g. NSE 15:30 IST = 10:00 UTC)
        has matching calendar dates in both UTC and local market time (2026-01-05).
        ForwardReturnCalculator matches the observation date and produces realization date
        in the consistent UTC convention.
        """
        inst_id = uuid4()
        calc = ForwardReturnCalculator()
        defn = self._make_defn(horizon=2)

        # 3 bars at 10:00 UTC (15:30 IST market close) on Jan 5, Jan 6, Jan 7
        t_base = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
        bars = [
            make_bar(t_base, Decimal("100"), instrument_id=inst_id),
            make_bar(t_base + timedelta(days=1), Decimal("105"), instrument_id=inst_id),
            make_bar(t_base + timedelta(days=2), Decimal("110"), instrument_id=inst_id),
        ]

        result = calc.calculate(defn, inst_id, bars, date(2026, 1, 5))

        assert result is not None
        assert result.observation_date == date(2026, 1, 5)
        assert result.realization_date == date(2026, 1, 7)
        assert result.value == Decimal("0.10")
