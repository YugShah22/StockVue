"""Tests for MarketBarValidator and FundamentalRecordValidator."""

from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from app.core.exceptions import DataQualityError
from app.domain.entities.fundamental_record import FundamentalRecord
from app.domain.entities.market_bar import MarketBar
from app.domain.services.fundamental_record_validator import (
    FundamentalRecordValidator,
)
from app.domain.services.market_bar_validator import MarketBarValidator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_bar(
    instrument_id: UUID | None = None,
    timestamp: datetime | None = None,
    *,
    close: Decimal = Decimal("105.0"),
) -> MarketBar:
    return MarketBar(
        instrument_id=instrument_id or uuid4(),
        timestamp=timestamp or datetime(2026, 9, 1, tzinfo=UTC),
        open=close,
        high=close + Decimal("5"),
        low=close - Decimal("5"),
        close=close,
        volume=Decimal("1000"),
    )


def make_record(
    instrument_id: UUID | None = None,
    period_end: date = date(2026, 6, 30),
    metric_name: str = "revenue",
    fiscal_year: int | None = 2026,
    fiscal_quarter: int | None = 2,
) -> FundamentalRecord:
    return FundamentalRecord(
        instrument_id=instrument_id or uuid4(),
        period_end=period_end,
        metric_name=metric_name,
        value=Decimal("100000.0"),
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
    )


# ===========================================================================
# MarketBarValidator tests
# ===========================================================================


class TestMarketBarValidatorEmptyBatch:
    def test_empty_batch_does_not_raise(self) -> None:
        instrument_id = uuid4()
        start = datetime(2026, 9, 1, tzinfo=UTC)
        end = datetime(2026, 9, 5, tzinfo=UTC)
        MarketBarValidator.validate_batch([], instrument_id, start, end)


class TestMarketBarValidatorWrongInstrument:
    def test_wrong_instrument_raises(self) -> None:
        instrument_id = uuid4()
        wrong_id = uuid4()
        bar = make_bar(instrument_id=wrong_id)
        start = datetime(2026, 9, 1, tzinfo=UTC)
        end = datetime(2026, 9, 5, tzinfo=UTC)

        with pytest.raises(DataQualityError, match="expected"):
            MarketBarValidator.validate_batch([bar], instrument_id, start, end)


class TestMarketBarValidatorNaiveTimestamp:
    def test_naive_timestamp_raises(self) -> None:
        instrument_id = uuid4()
        bar = make_bar(
            instrument_id=instrument_id,
            timestamp=datetime(2026, 9, 1),  # naive
        )
        start = datetime(2026, 9, 1, tzinfo=UTC)
        end = datetime(2026, 9, 5, tzinfo=UTC)

        with pytest.raises(DataQualityError, match="timezone-naive"):
            MarketBarValidator.validate_batch([bar], instrument_id, start, end)


class TestMarketBarValidatorOutOfRange:
    def test_bar_before_start_raises(self) -> None:
        instrument_id = uuid4()
        bar = make_bar(
            instrument_id=instrument_id,
            timestamp=datetime(2026, 8, 31, tzinfo=UTC),
        )
        start = datetime(2026, 9, 1, tzinfo=UTC)
        end = datetime(2026, 9, 5, tzinfo=UTC)

        with pytest.raises(DataQualityError, match="outside requested window"):
            MarketBarValidator.validate_batch([bar], instrument_id, start, end)

    def test_bar_after_end_raises(self) -> None:
        instrument_id = uuid4()
        bar = make_bar(
            instrument_id=instrument_id,
            timestamp=datetime(2026, 9, 6, tzinfo=UTC),
        )
        start = datetime(2026, 9, 1, tzinfo=UTC)
        end = datetime(2026, 9, 5, tzinfo=UTC)

        with pytest.raises(DataQualityError, match="outside requested window"):
            MarketBarValidator.validate_batch([bar], instrument_id, start, end)

    def test_bar_at_start_is_valid(self) -> None:
        instrument_id = uuid4()
        bar = make_bar(
            instrument_id=instrument_id,
            timestamp=datetime(2026, 9, 1, tzinfo=UTC),
        )
        start = datetime(2026, 9, 1, tzinfo=UTC)
        end = datetime(2026, 9, 5, tzinfo=UTC)
        MarketBarValidator.validate_batch([bar], instrument_id, start, end)

    def test_bar_at_end_is_valid(self) -> None:
        instrument_id = uuid4()
        bar = make_bar(
            instrument_id=instrument_id,
            timestamp=datetime(2026, 9, 5, tzinfo=UTC),
        )
        start = datetime(2026, 9, 1, tzinfo=UTC)
        end = datetime(2026, 9, 5, tzinfo=UTC)
        MarketBarValidator.validate_batch([bar], instrument_id, start, end)


class TestMarketBarValidatorDuplicateTimestamps:
    def test_duplicate_timestamp_raises(self) -> None:
        instrument_id = uuid4()
        ts = datetime(2026, 9, 1, tzinfo=UTC)
        bar1 = make_bar(instrument_id=instrument_id, timestamp=ts)
        bar2 = make_bar(instrument_id=instrument_id, timestamp=ts)
        start = datetime(2026, 9, 1, tzinfo=UTC)
        end = datetime(2026, 9, 5, tzinfo=UTC)

        with pytest.raises(DataQualityError, match="Duplicate timestamp"):
            MarketBarValidator.validate_batch(
                [bar1, bar2], instrument_id, start, end
            )


class TestMarketBarValidatorNonMonotonicOrder:
    def test_non_monotonic_order_raises(self) -> None:
        instrument_id = uuid4()
        t1 = datetime(2026, 9, 2, tzinfo=UTC)
        t2 = datetime(2026, 9, 1, tzinfo=UTC)
        bar1 = make_bar(instrument_id=instrument_id, timestamp=t1)
        bar2 = make_bar(instrument_id=instrument_id, timestamp=t2)
        start = datetime(2026, 9, 1, tzinfo=UTC)
        end = datetime(2026, 9, 5, tzinfo=UTC)

        with pytest.raises(DataQualityError, match="ascending order"):
            MarketBarValidator.validate_batch(
                [bar1, bar2], instrument_id, start, end
            )

    def test_monotonic_order_is_valid(self) -> None:
        instrument_id = uuid4()
        t1 = datetime(2026, 9, 1, tzinfo=UTC)
        t2 = datetime(2026, 9, 2, tzinfo=UTC)
        t3 = datetime(2026, 9, 3, tzinfo=UTC)
        bars = [
            make_bar(instrument_id=instrument_id, timestamp=t1),
            make_bar(instrument_id=instrument_id, timestamp=t2),
            make_bar(instrument_id=instrument_id, timestamp=t3),
        ]
        start = datetime(2026, 9, 1, tzinfo=UTC)
        end = datetime(2026, 9, 5, tzinfo=UTC)
        MarketBarValidator.validate_batch(bars, instrument_id, start, end)


# ===========================================================================
# FundamentalRecordValidator tests
# ===========================================================================


class TestFundamentalRecordValidatorEmptyBatch:
    def test_empty_batch_does_not_raise(self) -> None:
        instrument_id = uuid4()
        FundamentalRecordValidator.validate_batch(
            [],
            instrument_id,
            start_period=date(2026, 1, 1),
            end_period=date(2026, 12, 31),
        )


class TestFundamentalRecordValidatorWrongInstrument:
    def test_wrong_instrument_raises(self) -> None:
        instrument_id = uuid4()
        wrong_id = uuid4()
        record = make_record(instrument_id=wrong_id)

        with pytest.raises(DataQualityError, match="expected"):
            FundamentalRecordValidator.validate_batch(
                [record],
                instrument_id,
                start_period=date(2026, 1, 1),
                end_period=date(2026, 12, 31),
            )


class TestFundamentalRecordValidatorOutOfRange:
    def test_period_before_start_raises(self) -> None:
        instrument_id = uuid4()
        record = make_record(
            instrument_id=instrument_id, period_end=date(2025, 12, 31)
        )

        with pytest.raises(DataQualityError, match="outside requested window"):
            FundamentalRecordValidator.validate_batch(
                [record],
                instrument_id,
                start_period=date(2026, 1, 1),
                end_period=date(2026, 12, 31),
            )

    def test_period_after_end_raises(self) -> None:
        instrument_id = uuid4()
        record = make_record(
            instrument_id=instrument_id, period_end=date(2027, 1, 1)
        )

        with pytest.raises(DataQualityError, match="outside requested window"):
            FundamentalRecordValidator.validate_batch(
                [record],
                instrument_id,
                start_period=date(2026, 1, 1),
                end_period=date(2026, 12, 31),
            )

    def test_period_at_boundary_is_valid(self) -> None:
        instrument_id = uuid4()
        record = make_record(
            instrument_id=instrument_id, period_end=date(2026, 1, 1)
        )
        FundamentalRecordValidator.validate_batch(
            [record],
            instrument_id,
            start_period=date(2026, 1, 1),
            end_period=date(2026, 12, 31),
        )


class TestFundamentalRecordValidatorNaiveTimestamps:
    def test_naive_available_at_raises(self) -> None:
        instrument_id = uuid4()
        record = make_record(instrument_id=instrument_id)
        record.available_at = datetime(2026, 9, 1)  # naive

        with pytest.raises(DataQualityError, match="timezone-naive"):
            FundamentalRecordValidator.validate_batch(
                [record],
                instrument_id,
                start_period=date(2026, 1, 1),
                end_period=date(2026, 12, 31),
            )

    def test_naive_published_at_raises(self) -> None:
        instrument_id = uuid4()
        record = make_record(instrument_id=instrument_id)
        record.published_at = datetime(2026, 9, 1)  # naive

        with pytest.raises(DataQualityError, match="timezone-naive"):
            FundamentalRecordValidator.validate_batch(
                [record],
                instrument_id,
                start_period=date(2026, 1, 1),
                end_period=date(2026, 12, 31),
            )


class TestFundamentalRecordValidatorDuplicates:
    def test_duplicate_key_raises(self) -> None:
        instrument_id = uuid4()
        record1 = make_record(
            instrument_id=instrument_id,
            period_end=date(2026, 6, 30),
            metric_name="revenue",
            fiscal_year=2026,
            fiscal_quarter=2,
        )
        record2 = make_record(
            instrument_id=instrument_id,
            period_end=date(2026, 6, 30),
            metric_name="revenue",
            fiscal_year=2026,
            fiscal_quarter=2,
        )

        with pytest.raises(DataQualityError, match="Duplicate"):
            FundamentalRecordValidator.validate_batch(
                [record1, record2],
                instrument_id,
                start_period=date(2026, 1, 1),
                end_period=date(2026, 12, 31),
            )

    def test_different_metric_same_period_is_valid(self) -> None:
        instrument_id = uuid4()
        record1 = make_record(
            instrument_id=instrument_id,
            metric_name="revenue",
        )
        record2 = make_record(
            instrument_id=instrument_id,
            metric_name="net_income",
        )

        FundamentalRecordValidator.validate_batch(
            [record1, record2],
            instrument_id,
            start_period=date(2026, 1, 1),
            end_period=date(2026, 12, 31),
        )
