from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.domain.entities.market_bar import MarketBar
from app.domain.providers import MarketDataProvider, MarketQuote


class FakeMarketDataProvider:
    def __init__(self) -> None:
        self.instrument_id = uuid4()
        self.bar = MarketBar(
            instrument_id=self.instrument_id,
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0,
            volume=1000.0,
        )

    def get_bars(
        self,
        instrument_id: UUID,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> list[MarketBar]:
        return [self.bar]

    def get_latest_bar(self, instrument_id: UUID, interval: str) -> MarketBar:
        return self.bar

    def get_quote(self, instrument_id: UUID) -> MarketQuote:
        return MarketQuote(
            instrument_id=instrument_id,
            timestamp=self.bar.timestamp,
            price=self.bar.close,
            volume=self.bar.volume,
        )


def test_provider_contract_returns_market_bars() -> None:
    fake_provider = FakeMarketDataProvider()
    provider: MarketDataProvider = fake_provider

    bars = provider.get_bars(
        fake_provider.instrument_id,
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
        "1d",
    )

    assert len(bars) == 1
    assert bars[0].close == 105.0


def test_provider_contract_returns_latest_bar() -> None:
    fake_provider = FakeMarketDataProvider()
    provider: MarketDataProvider = fake_provider

    bar = provider.get_latest_bar(
        fake_provider.instrument_id,
        "1d",
    )

    assert bar.close == 105.0


def test_provider_contract_returns_quote() -> None:
    fake_provider = FakeMarketDataProvider()
    provider: MarketDataProvider = fake_provider

    quote = provider.get_quote(fake_provider.instrument_id)

    assert quote.price == 105.0
    assert quote.volume == 1000.0
