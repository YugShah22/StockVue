from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest

from app.domain.entities.fundamental_record import FundamentalRecord


def test_fundamental_value_preserves_decimal_precision() -> None:
    record = FundamentalRecord(
        instrument_id=uuid4(),
        period_end=date(2026, 3, 31),
        metric_name="revenue",
        value=Decimal("123456789.123456"),
        fiscal_year=2026,
        fiscal_quarter=4,
        currency="INR",
    )

    assert record.value == Decimal("123456789.123456")
    assert isinstance(record.value, Decimal)


def test_fiscal_quarter_must_be_between_one_and_four() -> None:
    for quarter in (0, 5, -1):
        with pytest.raises(ValueError):
            FundamentalRecord(
                instrument_id=uuid4(),
                period_end=date(2026, 3, 31),
                metric_name="revenue",
                value=Decimal("100000"),
                fiscal_year=2026,
                fiscal_quarter=quarter,
                currency="INR",
            )


def test_available_at_cannot_precede_published_at() -> None:
    from datetime import UTC, datetime

    with pytest.raises(ValueError):
        FundamentalRecord(
            instrument_id=uuid4(),
            period_end=date(2026, 3, 31),
            metric_name="revenue",
            value=Decimal("100000"),
            published_at=datetime(2026, 5, 1, tzinfo=UTC),
            available_at=datetime(2026, 4, 30, tzinfo=UTC),
        )


def test_fundamental_period_is_preserved() -> None:
    record = FundamentalRecord(
        instrument_id=uuid4(),
        period_end=date(2026, 3, 31),
        metric_name="net income",
        value=Decimal("250000"),
        fiscal_year=2026,
        fiscal_quarter=4,
        currency="INR",
    )

    assert record.period_end == date(2026, 3, 31)
    assert record.fiscal_year == 2026
    assert record.fiscal_quarter == 4
