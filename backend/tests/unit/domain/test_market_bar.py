from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from app.domain.entities.market_bar import MarketBar


def create_market_bar() -> MarketBar:
    return MarketBar(
        instrument_id=uuid4(),
        timestamp=datetime.now(UTC),
        open=Decimal("100.0"),
        high=Decimal("110.0"),
        low=Decimal("95.0"),
        close=Decimal("105.0"),
        volume=Decimal("100000.0"),
    )


def test_market_bar_creation() -> None:
    bar = create_market_bar()

    assert bar.open == Decimal("100.0")
    assert bar.high == Decimal("110.0")
    assert bar.low == Decimal("95.0")
    assert bar.close == Decimal("105.0")
    assert bar.volume == Decimal("100000.0")


@pytest.mark.parametrize(
    "field",
    ["open", "high", "low", "close", "volume"],
)
def test_market_bar_rejects_negative_values(field: str) -> None:
    values = {
        "open": Decimal("100.0"),
        "high": Decimal("110.0"),
        "low": Decimal("95.0"),
        "close": Decimal("105.0"),
        "volume": Decimal("100000.0"),
    }

    values[field] = Decimal("-1.0")

    with pytest.raises(ValueError):
        MarketBar(
            instrument_id=uuid4(),
            timestamp=datetime.now(UTC),
            **values,
        )


def test_market_bar_rejects_high_below_low() -> None:
    with pytest.raises(ValueError):
        MarketBar(
            instrument_id=uuid4(),
            timestamp=datetime.now(UTC),
            open=Decimal("100.0"),
            high=Decimal("90.0"),
            low=Decimal("95.0"),
            close=Decimal("100.0"),
            volume=Decimal("100000.0"),
        )


def test_market_bar_rejects_open_outside_range() -> None:
    with pytest.raises(ValueError):
        MarketBar(
            instrument_id=uuid4(),
            timestamp=datetime.now(UTC),
            open=Decimal("120.0"),
            high=Decimal("110.0"),
            low=Decimal("95.0"),
            close=Decimal("105.0"),
            volume=Decimal("100000.0"),
        )


def test_market_bar_rejects_close_outside_range() -> None:
    with pytest.raises(ValueError):
        MarketBar(
            instrument_id=uuid4(),
            timestamp=datetime.now(UTC),
            open=Decimal("100.0"),
            high=Decimal("110.0"),
            low=Decimal("95.0"),
            close=Decimal("120.0"),
            volume=Decimal("100000.0"),
        )

