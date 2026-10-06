from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import event
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
                open=Decimal("100.0"),
                high=Decimal("110.0"),
                low=Decimal("95.0"),
                close=Decimal("105.0"),
                volume=Decimal("1000.0"),
            ),
            MarketBar(
                instrument_id=instrument_id,
                timestamp=datetime(2026, 9, 2, tzinfo=UTC),
                open=Decimal("105.0"),
                high=Decimal("115.0"),
                low=Decimal("100.0"),
                close=Decimal("112.0"),
                volume=Decimal("1200.0"),
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
    assert first_bar.close == Decimal("105.0")
    assert first_bar.volume == Decimal("1000.0")

    assert second_bar.instrument_id == instrument_id
    assert second_bar.close == Decimal("112.0")
    assert second_bar.volume == Decimal("1200.0")


class DynamicBatchMarketDataProvider(MarketDataProvider):
    def __init__(self, bars: list[MarketBar]) -> None:
        self.bars = bars

    def get_bars(
        self,
        instrument_id: UUID,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> list[MarketBar]:
        return self.bars

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


def test_price_ingestion_bulk_500_bars_lifecycle(
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
        name="Bulk Test Company",
        legal_name="Bulk Test Company Limited",
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
        isin=ISIN("INE999A01999"),
        symbol=Symbol("BULKTEST"),
        asset_type=AssetType.EQUITY,
        status=InstrumentStatus.ACTIVE,
        exchange_code=ExchangeCode("NSE"),
    )

    company_repository.save(company)
    exchange_repository.save(exchange)
    instrument_repository.save(instrument)

    start_time = datetime(2026, 9, 1, 9, 15, tzinfo=UTC)
    end_time = start_time + timedelta(minutes=499)

    initial_bars = [
        MarketBar(
            instrument_id=instrument_id,
            timestamp=start_time + timedelta(minutes=i),
            open=Decimal(f"{100 + i}.0"),
            high=Decimal(f"{110 + i}.0"),
            low=Decimal(f"{90 + i}.0"),
            close=Decimal(f"{105 + i}.0"),
            volume=Decimal(f"{1000 + i}.0"),
        )
        for i in range(500)
    ]

    provider = DynamicBatchMarketDataProvider(initial_bars)
    service = PriceIngestionService(
        instrument_repository=instrument_repository,
        market_bar_repository=market_bar_repository,
        market_data_provider=provider,
    )

    engine = db_session.get_bind()
    statements: list[str] = []

    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", before_cursor_execute)
    try:
        result1 = service.ingest_bars(
            instrument_id=instrument_id,
            start=start_time,
            end=end_time,
            interval="1m",
        )
    finally:
        event.remove(engine, "before_cursor_execute", before_cursor_execute)

    # 1. Batch insert of 500 bars
    assert result1.received_count == 500
    assert result1.inserted_count == 500
    assert result1.updated_count == 0
    assert result1.skipped_count == 0

    # Total SQL statements executed for 500 bars:
    # 1 SELECT instrument + 1 SELECT exchange + 1 SELECT get_bars + 1 INSERT/UPSERT = 4 statements total (O(1) round trips!)
    assert len(statements) <= 5
    upsert_stmts = [s for s in statements if "INSERT INTO market_bars" in s]
    assert len(upsert_stmts) == 1
    select_bar_stmts = [s for s in statements if "FROM market_bars" in s]
    assert len(select_bar_stmts) == 1

    # 2. Re-ingesting the exact same batch: all 500 should be skipped
    statements.clear()
    event.listen(engine, "before_cursor_execute", before_cursor_execute)
    try:
        result2 = service.ingest_bars(
            instrument_id=instrument_id,
            start=start_time,
            end=end_time,
            interval="1m",
        )
    finally:
        event.remove(engine, "before_cursor_execute", before_cursor_execute)

    assert result2.received_count == 500
    assert result2.inserted_count == 0
    assert result2.updated_count == 0
    assert result2.skipped_count == 500
    # No upsert statement should be executed when all rows skipped
    assert not any("INSERT INTO market_bars" in s for s in statements)

    # 3. Ingesting batch with 200 modified bars and 300 unchanged bars
    modified_bars = []
    for i in range(500):
        if i < 200:
            modified_bars.append(
                MarketBar(
                    instrument_id=instrument_id,
                    timestamp=start_time + timedelta(minutes=i),
                    open=Decimal(f"{100 + i}.0"),
                    high=Decimal(f"{120 + i}.0"),
                    low=Decimal(f"{90 + i}.0"),
                    close=Decimal(f"{115 + i}.0"),
                    volume=Decimal(f"{1000 + i}.0"),
                )
            )
        else:
            modified_bars.append(initial_bars[i])

    provider.bars = modified_bars
    result3 = service.ingest_bars(
        instrument_id=instrument_id,
        start=start_time,
        end=end_time,
        interval="1m",
    )

    assert result3.received_count == 500
    assert result3.inserted_count == 0
    assert result3.updated_count == 200
    assert result3.skipped_count == 300

    # 4. Verify database state
    persisted = market_bar_repository.get_bars(instrument_id, start_time, end_time)
    assert len(persisted) == 500
    for i in range(200):
        assert persisted[i].close == Decimal(f"{115 + i}.0")
    for i in range(200, 500):
        assert persisted[i].close == Decimal(f"{105 + i}.0")
