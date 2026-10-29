"""Unit tests for AdjustedPriceService.

Tests split-adjusted OHLCV price and volume transformations, point-in-time
isolation, deterministic chronological ordering, and raw-bar immutability.
"""

from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from app.domain.entities.corporate_action import CorporateAction
from app.domain.entities.market_bar import MarketBar
from app.domain.enums.corporate_action_type import CorporateActionType
from app.domain.services.adjusted_price_service import AdjustedPriceService


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


class TestAdjustedPriceService:
    """Test suite for pure AdjustedPriceService."""

    def test_empty_bars_returns_empty_list(self) -> None:
        service = AdjustedPriceService()
        result = service.adjust([], [])
        assert result == []

    def test_no_corporate_actions_returns_unadjusted_new_instances(self) -> None:
        inst_id = uuid4()
        bar = _make_bar(
            datetime(2026, 1, 5, 10, tzinfo=UTC),
            Decimal("100"),
            instrument_id=inst_id,
            open_=Decimal("98"),
            high=Decimal("102"),
            low=Decimal("97"),
            volume=Decimal("500"),
        )
        service = AdjustedPriceService()
        result = service.adjust([bar], [])

        assert len(result) == 1
        adj = result[0]
        # Immutability: new instance returned
        assert adj is not bar
        assert adj.market_bar_id == bar.market_bar_id
        assert adj.instrument_id == inst_id
        assert adj.timestamp == bar.timestamp
        assert adj.open == Decimal("98")
        assert adj.high == Decimal("102")
        assert adj.low == Decimal("97")
        assert adj.close == Decimal("100")
        assert adj.volume == Decimal("500")

    def test_two_for_one_split_adjusts_historical_bars(self) -> None:
        inst_id = uuid4()
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 6),
            value=Decimal("2"),  # 2-for-1
        )
        t0 = _make_bar(
            datetime(2026, 1, 5, 10, tzinfo=UTC),
            Decimal("200"),
            instrument_id=inst_id,
            open_=Decimal("190"),
            high=Decimal("210"),
            low=Decimal("180"),
            volume=Decimal("1000"),
        )
        t1 = _make_bar(
            datetime(2026, 1, 6, 10, tzinfo=UTC),
            Decimal("105"),
            instrument_id=inst_id,
            open_=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("98"),
            volume=Decimal("2200"),
        )
        service = AdjustedPriceService()
        result = service.adjust([t0, t1], [split])

        assert len(result) == 2
        adj_t0, adj_t1 = result

        # Pre-split bar t0 is back-adjusted: prices divided by 2, volume multiplied by 2
        assert adj_t0.open == Decimal("95")
        assert adj_t0.high == Decimal("105")
        assert adj_t0.low == Decimal("90")
        assert adj_t0.close == Decimal("100")
        assert adj_t0.volume == Decimal("2000")

        # Post-split bar t1 (on execution date) is already traded on split basis -> unadjusted
        assert adj_t1.open == Decimal("100")
        assert adj_t1.high == Decimal("110")
        assert adj_t1.low == Decimal("98")
        assert adj_t1.close == Decimal("105")
        assert adj_t1.volume == Decimal("2200")

    def test_reverse_split_adjusts_historical_bars(self) -> None:
        inst_id = uuid4()
        # 1-for-2 reverse split: 0.5 new shares for 1 old share
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 6),
            value=Decimal("0.5"),
        )
        t0 = _make_bar(
            datetime(2026, 1, 5, 10, tzinfo=UTC),
            Decimal("50"),
            instrument_id=inst_id,
            open_=Decimal("48"),
            high=Decimal("52"),
            low=Decimal("47"),
            volume=Decimal("2000"),
        )
        service = AdjustedPriceService()
        result = service.adjust([t0], [split])

        adj_t0 = result[0]
        # Prices divided by 0.5 (i.e. doubled), volume multiplied by 0.5 (i.e. halved)
        assert adj_t0.open == Decimal("96")
        assert adj_t0.high == Decimal("104")
        assert adj_t0.low == Decimal("94")
        assert adj_t0.close == Decimal("100")
        assert adj_t0.volume == Decimal("1000")

    def test_consistent_ohlc_adjustment_preserves_relationships(self) -> None:
        inst_id = uuid4()
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 6),
            value=Decimal("2"),
        )
        bar = MarketBar(
            instrument_id=inst_id,
            timestamp=datetime(2026, 1, 5, 10, tzinfo=UTC),
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("95"),
            close=Decimal("105"),
            volume=Decimal("1000"),
        )
        service = AdjustedPriceService()
        result = service.adjust([bar], [split])

        adj = result[0]
        assert adj.open == Decimal("50")
        assert adj.high == Decimal("55")
        assert adj.low == Decimal("47.5")
        assert adj.close == Decimal("52.5")
        assert adj.volume == Decimal("2000")
        assert adj.low <= adj.open <= adj.high
        assert adj.low <= adj.close <= adj.high

    def test_multiple_splits_combine_multiplicatively(self) -> None:
        inst_id = uuid4()
        # Split 1: 2-for-1 on Feb 1
        s1 = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 2, 1),
            value=Decimal("2"),
        )
        # Split 2: 3-for-1 on May 1
        s2 = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 5, 1),
            value=Decimal("3"),
        )
        # Bar 1 (Jan 15): before both splits -> factor = 2 * 3 = 6
        b_jan = _make_bar(
            datetime(2026, 1, 15, 10, tzinfo=UTC),
            Decimal("600"),
            instrument_id=inst_id,
            volume=Decimal("100"),
        )
        # Bar 2 (Mar 15): after s1, before s2 -> factor = 3
        b_mar = _make_bar(
            datetime(2026, 3, 15, 10, tzinfo=UTC),
            Decimal("300"),
            instrument_id=inst_id,
            volume=Decimal("200"),
        )
        # Bar 3 (Jun 15): after both splits -> factor = 1
        b_jun = _make_bar(
            datetime(2026, 6, 15, 10, tzinfo=UTC),
            Decimal("110"),
            instrument_id=inst_id,
            volume=Decimal("600"),
        )

        service = AdjustedPriceService()
        result = service.adjust([b_jan, b_mar, b_jun], [s1, s2])

        assert len(result) == 3
        adj_jan, adj_mar, adj_jun = result

        # Jan: 600 / 6 = 100, volume 100 * 6 = 600
        assert adj_jan.close == Decimal("100")
        assert adj_jan.volume == Decimal("600")

        # Mar: 300 / 3 = 100, volume 200 * 3 = 600
        assert adj_mar.close == Decimal("100")
        assert adj_mar.volume == Decimal("600")

        # Jun: 110 / 1 = 110, volume 600 * 1 = 600
        assert adj_jun.close == Decimal("110")
        assert adj_jun.volume == Decimal("600")

    def test_raw_bar_immutability(self) -> None:
        inst_id = uuid4()
        raw_bar = _make_bar(
            datetime(2026, 1, 5, 10, tzinfo=UTC),
            Decimal("200"),
            instrument_id=inst_id,
            open_=Decimal("195"),
            high=Decimal("205"),
            low=Decimal("190"),
            volume=Decimal("1000"),
        )
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 6),
            value=Decimal("2"),
        )
        service = AdjustedPriceService()
        result = service.adjust([raw_bar], [split])

        adj_bar = result[0]
        # Raw bar is completely unmutated
        assert raw_bar.open == Decimal("195")
        assert raw_bar.high == Decimal("205")
        assert raw_bar.low == Decimal("190")
        assert raw_bar.close == Decimal("200")
        assert raw_bar.volume == Decimal("1000")
        assert adj_bar is not raw_bar
        assert adj_bar.close == Decimal("100")

    def test_chronological_ordering_deterministic_output(self) -> None:
        inst_id = uuid4()
        b1 = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id)
        b2 = _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("101"), inst_id)
        b3 = _make_bar(datetime(2026, 1, 7, 10, tzinfo=UTC), Decimal("102"), inst_id)

        service = AdjustedPriceService()
        # Pass in reverse order [b3, b1, b2]
        result = service.adjust([b3, b1, b2], [])

        # Must be returned in ascending chronological order [b1, b2, b3]
        assert [b.timestamp for b in result] == [
            b1.timestamp,
            b2.timestamp,
            b3.timestamp,
        ]

    def test_point_in_time_protection_future_splits_ignored(self) -> None:
        inst_id = uuid4()
        t0 = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("200"), inst_id)
        # Split executes on Jan 20 (future relative to observation date Jan 5)
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 20),
            value=Decimal("2"),
        )
        service = AdjustedPriceService()

        # View as of Jan 5 (obs_date): split at Jan 20 is not yet effective
        result_pit = service.adjust([t0], [split], as_of_date=date(2026, 1, 5))
        assert result_pit[0].close == Decimal("200")  # NOT divided by 2

        # View as of Jan 25 (post-split): split is now effective, t0 is back-adjusted
        result_post = service.adjust([t0], [split], as_of_date=date(2026, 1, 25))
        assert result_post[0].close == Decimal("100")  # Divided by 2

    def test_dividends_ignored_in_ohlcv_adjustment(self) -> None:
        inst_id = uuid4()
        dividend = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.DIVIDEND,
            execution_date=date(2026, 1, 6),
            value=Decimal("5.00"),
            currency="INR",
        )
        bar = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id)
        service = AdjustedPriceService()
        result = service.adjust([bar], [dividend])

        # Dividend does not affect split-adjusted OHLCV
        assert result[0].close == Decimal("100")
        assert result[0].volume == Decimal("1000")

    def test_multi_instrument_splits_isolated(self) -> None:
        inst_a = uuid4()
        inst_b = uuid4()
        split_a = CorporateAction(
            instrument_id=inst_a,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 6),
            value=Decimal("2"),
        )
        bar_a = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("200"), inst_a)
        bar_b = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("200"), inst_b)

        service = AdjustedPriceService()
        result = service.adjust([bar_a, bar_b], [split_a])

        # Find adjusted bars
        adj_a = next(b for b in result if b.instrument_id == inst_a)
        adj_b = next(b for b in result if b.instrument_id == inst_b)

        assert adj_a.close == Decimal("100")  # Split applies to A
        assert adj_b.close == Decimal("200")  # Split does NOT apply to B

    def test_static_and_instance_invocation(self) -> None:
        inst_id = uuid4()
        bar = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id)
        # Calling via class staticmethod
        r1 = AdjustedPriceService.adjust([bar], [])
        # Calling via instance method
        r2 = AdjustedPriceService().adjust([bar], [])
        assert r1[0].close == r2[0].close == Decimal("100")

    def test_invalid_split_ratio_rejected(self) -> None:
        inst_id = uuid4()
        bar = _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id)
        # Construct action and bypass __post_init__ to test defensive check in adjust
        action = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 6),
            value=Decimal("2"),
        )
        object.__setattr__(action, "value", Decimal("0"))
        with pytest.raises(ValueError, match="invalid split ratio"):
            AdjustedPriceService.adjust([bar], [action])
