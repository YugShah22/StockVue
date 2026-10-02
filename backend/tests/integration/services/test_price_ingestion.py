from datetime import UTC, datetime
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.domain.entities.company import Company
from app.domain.entities.exchange import Exchange
from app.domain.entities.instrument import Instrument
from app.domain.entities.market_bar import MarketBar
from app.domain.enums.asset_type import AssetType
from app.domain.enums.instrument_status import InstrumentStatus
from app.domain.providers.market_data import MarketDataProvider, MarketQuote
from app.domain.value_objects.exchange_code import ExchangeCode
from app.domain.value_objects.isin import ISIN
from app.domain.value_objects.symbol import Symbol
from app.infrastructure.database.repositories.company import (
    PostgresCompanyRepository,
)
from app.infrastructure.database.repositories.exchange import (
    PostgresExchangeRepository,
)
from app.infrastructure.database.repositories.instrument import (
    PostgresInstrumentRepository,
)
from app.infrastructure.database.repositories.market_bar import (
    PostgresMarketBarRepository,
)
from app.services.price_ingestion import PriceIngestionService


class FakeMarketDataProvider(MarketDataProvider):
    def get_bars(
        self,
        instrument_id: UUID,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> list[MarketBar]:
        return [
            MarketBar(
                instrument_id=instrument_id,
                timestamp=datetime(2026, 9, 1, tzinfo=UTC),
                open=100.0,
                high=110.0,
                low=95.0,
                close=105.0,
                volume=1000.0,
            ),
            MarketBar(
                instrument_id=instrument_id,
                timestamp=datetime(2026, 9, 2, tzinfo=UTC),
                open=105.0,
                high=115.0,
                low=100.0,
                close=112.0,
                volume=1200.0,
            ),
        ]

    def get_latest_bar(
        self,
        instrument_id: UUID,
        interval: str,
    ) -> MarketBar:
        raise NotImplementedError

    def get_quote(
        self,
        instrument_id: UUID,
    ) -> MarketQuote:
        raise NotImplementedError


def test_price_ingestion_persists_market_bars(
    db_session: Session,
) -> None:
    company_repository = PostgresCompanyRepository(db_session)
    exchange_repository = PostgresExchangeRepository(db_session)
    instrument_repository = PostgresInstrumentRepository(db_session)
    market_bar_repository = PostgresMarketBarRepository(db_session)

    company_id = uuid4()
    exchange_id = uuid4()
    instrument_id = uuid4()

    company = Company(
        company_id=company_id,
        name="StockVue Test Company",
        legal_name="StockVue Test Company Private Limited",
        country="India",
    )

    exchange = Exchange(
        exchange_id=exchange_id,
        code=ExchangeCode("NSE"),
        name="National Stock Exchange",
        country="India",
        timezone=ZoneInfo("Asia/Kolkata"),
    )

    instrument = Instrument(
        instrument_id=instrument_id,
        company_id=company_id,
        isin=ISIN("INE123456789"),
        symbol=Symbol("TEST"),
        asset_type=AssetType.EQUITY,
        status=InstrumentStatus.ACTIVE,
        exchange_code=ExchangeCode("NSE"),
    )

    company_repository.save(company)
    exchange_repository.save(exchange)
    instrument_repository.save(instrument)

    service = PriceIngestionService(
        instrument_repository=instrument_repository,
        market_bar_repository=market_bar_repository,
        market_data_provider=FakeMarketDataProvider(),
    )

    result = service.ingest_bars(
        instrument_id=instrument_id,
        start=datetime(2026, 9, 1, tzinfo=UTC),
        end=datetime(2026, 9, 5, tzinfo=UTC),
        interval="1d",
    )

    assert result.received_count == 2
    assert result.inserted_count == 2
    assert result.updated_count == 0
    assert result.skipped_count == 0

    first_bar = market_bar_repository.get_by_instrument_and_timestamp(
        instrument_id,
        datetime(2026, 9, 1, tzinfo=UTC),
    )

    second_bar = market_bar_repository.get_by_instrument_and_timestamp(
        instrument_id,
        datetime(2026, 9, 2, tzinfo=UTC),
    )

    assert first_bar is not None
    assert second_bar is not None

    assert first_bar.instrument_id == instrument_id
    assert first_bar.close == 105.0
    assert first_bar.volume == 1000.0

    assert second_bar.instrument_id == instrument_id
    assert second_bar.close == 112.0
    assert second_bar.volume == 1200.0
