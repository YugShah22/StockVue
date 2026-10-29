"""Unit tests for TotalReturnCalculator.

Verifies split-aware and dividend-aware forward total return calculations,
including dividend reinvestment conventions, scale invariance for dividends
followed by splits, half-open interval boundaries, and raw-bar immutability.
"""

from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from app.domain.entities.corporate_action import CorporateAction
from app.domain.entities.market_bar import MarketBar
from app.domain.enums.corporate_action_type import CorporateActionType
from app.domain.prediction.calculators.total_return_calculator import (
    TotalReturnCalculator,
)
from app.domain.prediction.target_definition import TargetDefinition


def _make_bar(
    timestamp: datetime,
    close: Decimal,
    instrument_id: UUID | None = None,
    open_: Decimal | None = None,
    high: Decimal | None = None,
    low: Decimal | None = None,
    volume: Decimal = Decimal("1000"),
) -> MarketBar:
    inst_id = instrument_id or uuid4()
    c = close
    o = open_ if open_ is not None else c
    hi = high if high is not None else max(o, c)
    lo = low if low is not None else min(o, c)
    return MarketBar(
        instrument_id=inst_id,
        timestamp=timestamp,
        open=o,
        high=hi,
        low=lo,
        close=c,
        volume=volume,
    )


def _make_target_defn(name: str = "total_return", horizon: int = 1) -> TargetDefinition:
    return TargetDefinition(name=name, horizon_bars=horizon)


class TestTotalReturnCalculator:
    """Test suite for TotalReturnCalculator."""

    def test_no_corporate_actions_matches_raw_forward_return(self) -> None:
        inst_id = uuid4()
        t0 = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id)
        t1 = _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("105"), inst_id)

        calc = TotalReturnCalculator()
        result = calc.calculate(_make_target_defn(horizon=1), inst_id, [t0, t1], date(2026, 1, 5))

        assert result is not None
        assert result.value == Decimal("0.05")
        assert result.observation_date == date(2026, 1, 5)
        assert result.realization_date == date(2026, 1, 6)

    def test_two_for_one_split_does_not_create_false_loss(self) -> None:
        inst_id = uuid4()
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 6),
            value=Decimal("2"),  # 2-for-1
        )
        t0 = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("200"), inst_id)
        t1 = _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("100"), inst_id)

        calc = TotalReturnCalculator(corporate_actions=[split])
        result = calc.calculate(_make_target_defn(horizon=1), inst_id, [t0, t1], date(2026, 1, 5))

        assert result is not None
        # Economic value unchanged -> return is 0.00, NOT -0.50
        assert result.value == Decimal("0")
        assert result.value != Decimal("-0.5")

    def test_reverse_split_does_not_create_false_gain(self) -> None:
        inst_id = uuid4()
        # 1-for-2 reverse split: price doubles on ex-date
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 6),
            value=Decimal("0.5"),
        )
        t0 = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("50"), inst_id)
        t1 = _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("100"), inst_id)

        calc = TotalReturnCalculator()
        result = calc.calculate(
            _make_target_defn(horizon=1),
            inst_id,
            [t0, t1],
            date(2026, 1, 5),
            corporate_actions=[split],
        )

        assert result is not None
        # Economic value unchanged -> return is 0.00, NOT +1.00
        assert result.value == Decimal("0")

    def test_single_dividend_reinvestment_at_ex_dividend_close(self) -> None:
        inst_id = uuid4()
        dividend = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.DIVIDEND,
            execution_date=date(2026, 1, 6),
            value=Decimal("3.00"),
            currency="INR",
        )
        t0 = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id)
        t1 = _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("97"), inst_id)
        t2 = _make_bar(datetime(2026, 1, 7, 10, tzinfo=UTC), Decimal("97"), inst_id)

        calc = TotalReturnCalculator(corporate_actions=[dividend])
        result = calc.calculate(
            _make_target_defn(horizon=2), inst_id, [t0, t1, t2], date(2026, 1, 5)
        )

        assert result is not None
        # Reinvestment factor = 1 + 3/97 = 100/97. Total return = (97 * 100/97)/100 - 1 = 0.00
        assert result.value == Decimal("0")

    def test_multiple_dividends_compound_reinvestment(self) -> None:
        inst_id = uuid4()
        # Div 1 at t1: $2 on $98 close -> factor 1 + 2/98 = 100/98
        d1 = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.DIVIDEND,
            execution_date=date(2026, 1, 6),
            value=Decimal("2.00"),
        )
        # Div 2 at t2: $3 on $97 close -> factor 1 + 3/97 = 100/97
        d2 = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.DIVIDEND,
            execution_date=date(2026, 1, 7),
            value=Decimal("3.00"),
        )
        bars = [
            _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id),  # t0
            _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("98"), inst_id),  # t1
            _make_bar(datetime(2026, 1, 7, 10, tzinfo=UTC), Decimal("97"), inst_id),  # t2
            _make_bar(datetime(2026, 1, 8, 10, tzinfo=UTC), Decimal("97"), inst_id),  # t3
        ]

        calc = TotalReturnCalculator(corporate_actions=[d1, d2])
        result = calc.calculate(_make_target_defn(horizon=3), inst_id, bars, date(2026, 1, 5))

        assert result is not None
        # Expected: (97 * (100/98) * (100/97)) / 100 - 1 = 100/98 - 1 = 2/98
        expected = (Decimal("100") / Decimal("98")) - Decimal("1")
        assert abs(result.value - expected) < Decimal("1e-25")

    def test_dividend_followed_by_later_split_invariant_to_share_basis(
        self,
    ) -> None:
        """Numerical regression test: Dividend followed by a later split within horizon.

        Timeline:
            t0: obs_date, raw close = 200
            t1: dividend ex-date, raw close = 196 (drop by ~4), dividend = 4.00
            t2: 2-for-1 split ex-date, raw close = 98 (halved from 196)
            t3: realization_date, raw close = 98

        Portfolio simulation:
            Start at t0: 1 share @ 200 (V0 = 200)
            At t1: receive 4.00 on 1 share. Reinvest at raw close 196:
                   shares = 1 + 4/196 = 200/196
            At t2: 2-for-1 split doubles shares:
                   shares = (200/196) * 2 = 400/196
            At t3: realization close = 98.
                   V3 = (400/196) * 98 = 200.00
            True economic return = (200 - 200) / 200 = 0.00

        Consistency Requirement:
            The dividend (4.00) and its ex-date price (196) must be evaluated on
            the same share basis. Dividing 4.00 by a back-adjusted price (98)
            without adjusting the dividend would give false 2% gain.
            Both Method 1 (raw ex-date basis) and Method 2 (realization-date basis)
            must produce exact 0.00.
        """
        inst_id = uuid4()
        dividend = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.DIVIDEND,
            execution_date=date(2026, 1, 6),
            value=Decimal("4.00"),
        )
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 7),
            value=Decimal("2"),  # 2-for-1 split AFTER dividend
        )
        bars = [
            _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("200"), inst_id),  # t0
            _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("196"), inst_id),  # t1 div
            _make_bar(datetime(2026, 1, 7, 10, tzinfo=UTC), Decimal("98"), inst_id),  # t2 split
            _make_bar(
                datetime(2026, 1, 8, 10, tzinfo=UTC), Decimal("98"), inst_id
            ),  # t3 realization
        ]

        calc = TotalReturnCalculator(corporate_actions=[dividend, split])
        result = calc.calculate(_make_target_defn(horizon=3), inst_id, bars, date(2026, 1, 5))

        assert result is not None
        # Strict equality check: must be exactly 0.00, NOT +0.02
        assert result.value == Decimal("0")

    def test_split_followed_by_later_dividend_consistent(self) -> None:
        """Split followed by a later dividend within horizon."""
        inst_id = uuid4()
        # Split on t1: 2-for-1, price drops 200 -> 100
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 6),
            value=Decimal("2"),
        )
        # Dividend on t2: 2.00 on post-split shares, price drops 100 -> 98
        dividend = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.DIVIDEND,
            execution_date=date(2026, 1, 7),
            value=Decimal("2.00"),
        )
        bars = [
            _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("200"), inst_id),  # t0
            _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("100"), inst_id),  # t1 split
            _make_bar(datetime(2026, 1, 7, 10, tzinfo=UTC), Decimal("98"), inst_id),  # t2 div
            _make_bar(
                datetime(2026, 1, 8, 10, tzinfo=UTC), Decimal("98"), inst_id
            ),  # t3 realization
        ]

        calc = TotalReturnCalculator(corporate_actions=[split, dividend])
        result = calc.calculate(_make_target_defn(horizon=3), inst_id, bars, date(2026, 1, 5))

        assert result is not None
        assert result.value == Decimal("0")

    def test_action_on_observation_date_excluded(self) -> None:
        """Actions on observation date itself are EXCLUDED from (t0, th]."""
        inst_id = uuid4()
        # Dividend ex-date is t0 (2026-01-05). Investor buying at t0 close does NOT get it.
        dividend = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.DIVIDEND,
            execution_date=date(2026, 1, 5),
            value=Decimal("5.00"),
        )
        t0 = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id)
        t1 = _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("100"), inst_id)

        calc = TotalReturnCalculator(corporate_actions=[dividend])
        result = calc.calculate(_make_target_defn(horizon=1), inst_id, [t0, t1], date(2026, 1, 5))

        assert result is not None
        # Dividend on t0 is excluded, so return is 100/100 - 1 = 0.00 (not 5%)
        assert result.value == Decimal("0")

    def test_action_on_realization_date_included(self) -> None:
        """Actions on realization date ARE INCLUDED in (t0, th]."""
        inst_id = uuid4()
        # Dividend ex-date is realization date t1 (2026-01-06). Investor holding through t1 gets it.
        dividend = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.DIVIDEND,
            execution_date=date(2026, 1, 6),
            value=Decimal("5.00"),
        )
        t0 = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id)
        t1 = _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("95"), inst_id)

        calc = TotalReturnCalculator(corporate_actions=[dividend])
        result = calc.calculate(_make_target_defn(horizon=1), inst_id, [t0, t1], date(2026, 1, 5))

        assert result is not None
        # Price dropped 100 -> 95 (-5%), offset by 5 dividend -> total return 0.00
        # Reinvestment factor: 1 + 5/95 = 100/95. Total return: (95 * 100/95)/100 - 1 = 0
        assert result.value == Decimal("0")

    def test_action_outside_horizon_excluded(self) -> None:
        inst_id = uuid4()
        # Split occurs after realization date
        future_split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 10),
            value=Decimal("2"),
        )
        t0 = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id)
        t1 = _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("105"), inst_id)

        calc = TotalReturnCalculator(corporate_actions=[future_split])
        result = calc.calculate(_make_target_defn(horizon=1), inst_id, [t0, t1], date(2026, 1, 5))

        assert result is not None
        # Future split outside horizon has zero effect on this target
        assert result.value == Decimal("0.05")

    def test_missing_ex_date_bar_returns_none(self) -> None:
        inst_id = uuid4()
        # Dividend ex-date is 2026-01-06, but we omit the bar for 2026-01-06
        dividend = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.DIVIDEND,
            execution_date=date(2026, 1, 6),
            value=Decimal("2.00"),
        )
        t0 = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id)
        # Next bar jumps to 2026-01-07 (missing Jan 6)
        t2 = _make_bar(datetime(2026, 1, 7, 10, tzinfo=UTC), Decimal("100"), inst_id)

        calc = TotalReturnCalculator(corporate_actions=[dividend])
        result = calc.calculate(_make_target_defn(horizon=1), inst_id, [t0, t2], date(2026, 1, 5))

        # Missing ex-date bar means ex-div close is unknown -> returns None
        assert result is None

    def test_non_positive_ex_date_close_returns_none(self) -> None:
        inst_id = uuid4()
        dividend = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.DIVIDEND,
            execution_date=date(2026, 1, 6),
            value=Decimal("2.00"),
        )
        t0 = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id)
        t1 = _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("0"), inst_id)  # 0 close

        calc = TotalReturnCalculator(corporate_actions=[dividend])
        result = calc.calculate(_make_target_defn(horizon=1), inst_id, [t0, t1], date(2026, 1, 5))

        assert result is None

    def test_raw_market_bars_immutability(self) -> None:
        inst_id = uuid4()
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 6),
            value=Decimal("2"),
        )
        t0 = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("200"), inst_id)
        t1 = _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("100"), inst_id)

        orig_t0_close = t0.close
        orig_t1_close = t1.close

        calc = TotalReturnCalculator(corporate_actions=[split])
        calc.calculate(_make_target_defn(horizon=1), inst_id, [t0, t1], date(2026, 1, 5))

        # Original bars are completely untouched
        assert t0.close == orig_t0_close == Decimal("200")
        assert t1.close == orig_t1_close == Decimal("100")

    def test_multi_instrument_actions_isolated(self) -> None:
        inst_a = uuid4()
        inst_b = uuid4()
        # Split is for instrument B only
        split_b = CorporateAction(
            instrument_id=inst_b,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 6),
            value=Decimal("2"),
        )
        t0 = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("200"), inst_a)
        t1 = _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("100"), inst_a)

        calc = TotalReturnCalculator(corporate_actions=[split_b])
        result = calc.calculate(_make_target_defn(horizon=1), inst_a, [t0, t1], date(2026, 1, 5))

        assert result is not None
        # Split for B does NOT apply to A -> A has raw return -0.50
        assert result.value == Decimal("-0.5")

    def test_method_argument_overrides_constructor_actions(self) -> None:
        inst_id = uuid4()
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 6),
            value=Decimal("2"),
        )
        t0 = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("200"), inst_id)
        t1 = _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("100"), inst_id)

        # Constructor has empty actions
        calc = TotalReturnCalculator(corporate_actions=[])
        # Pass split directly in calculate call
        result = calc.calculate(
            _make_target_defn(horizon=1),
            inst_id,
            [t0, t1],
            date(2026, 1, 5),
            corporate_actions=[split],
        )

        assert result is not None
        assert result.value == Decimal("0")

    def test_split_on_observation_date_not_applied_to_horizon(self) -> None:
        """A split on observation date is NOT effective for the future horizon (t0, th].

        On observation_date, the ex-date has already arrived; the bar's closing price
        is traded on the post-split basis. An investor buying at t0 close does not experience
        a forward share split. Therefore, the split on t0 must not back-adjust t0 close.

        Scenario:
            t0: 2026-01-05, close = 100. Split 2-for-1 executes on 2026-01-05.
            t1: 2026-01-06, close = 100.
        Expected return: (100 / 100) - 1 = 0.00 (NOT 100 / 50 - 1 = +1.00).
        """
        inst_id = uuid4()
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 5),  # Split executes ON observation date
            value=Decimal("2"),
        )
        t0 = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id)
        t1 = _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("100"), inst_id)

        calc = TotalReturnCalculator(corporate_actions=[split])
        result = calc.calculate(_make_target_defn(horizon=1), inst_id, [t0, t1], date(2026, 1, 5))

        assert result is not None
        assert result.value == Decimal("0")
        assert result.value != Decimal("1.0")

    def test_mixed_instrument_input_bars_produces_identical_result(self) -> None:
        """Supplying bars from multiple instruments produces identical results to single-instrument input."""
        inst_a = uuid4()
        inst_b = uuid4()

        split_b = CorporateAction(
            instrument_id=inst_b,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 6),
            value=Decimal("2"),
        )

        bars_a = [
            _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_a),
            _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("110"), inst_a),
        ]
        bars_b = [
            _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("200"), inst_b),
            _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("100"), inst_b),
        ]

        # Interleave bars of instrument A and B
        combined_bars = [bars_a[0], bars_b[0], bars_a[1], bars_b[1]]

        calc = TotalReturnCalculator(corporate_actions=[split_b])

        # Calculating with combined bars vs isolated bars for inst_a:
        result_combined = calc.calculate(
            _make_target_defn(horizon=1), inst_a, combined_bars, date(2026, 1, 5)
        )
        result_isolated = calc.calculate(
            _make_target_defn(horizon=1), inst_a, bars_a, date(2026, 1, 5)
        )

        assert result_combined is not None
        assert result_isolated is not None
        assert result_combined.value == result_isolated.value == Decimal("0.1")
        assert result_combined.observation_date == date(2026, 1, 5)
        assert result_combined.realization_date == date(2026, 1, 6)

    def test_missing_observation_bar_returns_none(self) -> None:
        """Returns None safely when the requested observation bar is missing from input bars."""
        inst_id = uuid4()
        bars = [
            _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("100"), inst_id),
            _make_bar(datetime(2026, 1, 7, 10, tzinfo=UTC), Decimal("105"), inst_id),
        ]

        calc = TotalReturnCalculator()
        # Request observation on 2026-01-05, which is not present in bars
        result = calc.calculate(_make_target_defn(horizon=1), inst_id, bars, date(2026, 1, 5))

        assert result is None

    def test_insufficient_future_bars_returns_none(self) -> None:
        """Returns None safely when fewer than horizon_bars follow the observation bar."""
        inst_id = uuid4()
        bars = [
            _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id),
            _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("105"), inst_id),
        ]

        calc = TotalReturnCalculator()
        # Request horizon_bars=5, but only 1 future bar is available
        result = calc.calculate(_make_target_defn(horizon=5), inst_id, bars, date(2026, 1, 5))

        assert result is None
