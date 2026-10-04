from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from app.domain.entities.fundamental_record import FundamentalRecord


def test_creates_fundamental_record() -> None:
    instrument_id = uuid4()

    record = FundamentalRecord(
        instrument_id=instrument_id,
        period_end=date(2026, 6, 30),
        metric_name="Revenue",
        value=Decimal("250000"),
        fiscal_year=2026,
        fiscal_quarter=1,
        currency="inr",
    )

    assert record.instrument_id == instrument_id
    assert record.period_end == date(2026, 6, 30)
    assert record.metric_name == "revenue"
    assert record.value == Decimal("250000")
    assert record.fiscal_year == 2026
    assert record.fiscal_quarter == 1
    assert record.currency == "INR"
    assert record.fundamental_record_id is not None


def test_metric_name_is_normalized() -> None:
    record = FundamentalRecord(
        instrument_id=uuid4(),
        period_end=date(2026, 6, 30),
        metric_name="  Net Income  ",
        value=Decimal("18000"),
    )

    assert record.metric_name == "net income"


def test_currency_is_normalized() -> None:
    record = FundamentalRecord(
        instrument_id=uuid4(),
        period_end=date(2026, 6, 30),
        metric_name="revenue",
        value=Decimal("1000"),
        currency=" inr ",
    )

    assert record.currency == "INR"


def test_empty_metric_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Metric name cannot be empty"):
        FundamentalRecord(
            instrument_id=uuid4(),
            period_end=date(2026, 6, 30),
            metric_name="   ",
            value=Decimal("1000"),
        )


def test_empty_currency_is_rejected() -> None:
    with pytest.raises(ValueError, match="Currency cannot be empty"):
        FundamentalRecord(
            instrument_id=uuid4(),
            period_end=date(2026, 6, 30),
            metric_name="revenue",
            value=Decimal("1000"),
            currency="   ",
        )


def test_invalid_fiscal_year_is_rejected() -> None:
    with pytest.raises(ValueError, match="Fiscal year must be valid"):
        FundamentalRecord(
            instrument_id=uuid4(),
            period_end=date(2026, 6, 30),
            metric_name="revenue",
            value=Decimal("1000"),
            fiscal_year=1899,
        )


@pytest.mark.parametrize("quarter", [0, 5, -1])
def test_invalid_fiscal_quarter_is_rejected(
    quarter: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Fiscal quarter must be between 1 and 4",
    ):
        FundamentalRecord(
            instrument_id=uuid4(),
            period_end=date(2026, 6, 30),
            metric_name="revenue",
            value=Decimal("1000"),
            fiscal_quarter=quarter,
        )


def test_available_at_cannot_be_before_published_at() -> None:
    published_at = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)
    available_at = datetime(2026, 7, 25, 9, 0, tzinfo=UTC)

    with pytest.raises(
        ValueError,
        match="Available timestamp cannot be earlier than published timestamp",
    ):
        FundamentalRecord(
            instrument_id=uuid4(),
            period_end=date(2026, 6, 30),
            metric_name="revenue",
            value=Decimal("1000"),
            published_at=published_at,
            available_at=available_at,
        )


def test_each_record_gets_unique_id() -> None:
    first = FundamentalRecord(
        instrument_id=uuid4(),
        period_end=date(2026, 6, 30),
        metric_name="revenue",
        value=Decimal("1000"),
    )

    second = FundamentalRecord(
        instrument_id=uuid4(),
        period_end=date(2026, 6, 30),
        metric_name="revenue",
        value=Decimal("2000"),
    )

    assert first.fundamental_record_id != second.fundamental_record_id
