from datetime import date
from decimal import Decimal
from uuid import UUID, uuid4

from app.domain.entities.fundamental_record import FundamentalRecord
from app.domain.providers.fundamentals import FundamentalsProvider


class FakeFundamentalsProvider(FundamentalsProvider):
    def get_fundamentals(
        self,
        instrument_id: UUID,
        start_period: date,
        end_period: date,
    ) -> list[FundamentalRecord]:
        return [
            FundamentalRecord(
                instrument_id=instrument_id,
                period_end=date(2026, 3, 31),
                metric_name="revenue",
                value=Decimal("1000000"),
                fiscal_year=2026,
                fiscal_quarter=4,
                currency="INR",
            )
        ]


def test_fake_fundamentals_provider_returns_records() -> None:
    provider = FakeFundamentalsProvider()
    instrument_id = uuid4()

    records = provider.get_fundamentals(
        instrument_id=instrument_id,
        start_period=date(2026, 1, 1),
        end_period=date(2026, 12, 31),
    )

    assert len(records) == 1
    assert records[0].instrument_id == instrument_id
    assert records[0].metric_name == "revenue"
    assert records[0].value == Decimal("1000000")
    assert records[0].currency == "INR"
