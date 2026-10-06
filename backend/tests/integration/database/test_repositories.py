from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import event
from sqlalchemy.orm import Session

from app.domain.entities.company import Company
from app.domain.entities.corporate_action import CorporateAction
from app.domain.entities.exchange import Exchange
from app.domain.entities.fundamental_record import FundamentalRecord
from app.domain.entities.instrument import Instrument
from app.domain.entities.instrument_history import InstrumentHistory
from app.domain.entities.market_bar import MarketBar
from app.domain.entities.market_holiday import MarketHoliday
from app.domain.enums.asset_type import AssetType
from app.domain.enums.corporate_action_type import CorporateActionType
from app.domain.enums.instrument_status import InstrumentStatus
from app.domain.value_objects.exchange_code import ExchangeCode
from app.domain.value_objects.isin import ISIN
from app.domain.value_objects.symbol import Symbol
from app.infrastructure.database.repositories.company import (
    PostgresCompanyRepository,
)
from app.infrastructure.database.repositories.corporate_action import (
    PostgresCorporateActionRepository,
)
from app.infrastructure.database.repositories.exchange import (
    PostgresExchangeRepository,
)
from app.infrastructure.database.repositories.fundamental_record import (
    PostgresFundamentalRecordRepository,
)
from app.infrastructure.database.repositories.instrument import (
    PostgresInstrumentRepository,
)
from app.infrastructure.database.repositories.instrument_history import (
    PostgresInstrumentHistoryRepository,
)
from app.infrastructure.database.repositories.market_bar import (
    PostgresMarketBarRepository,
)
from app.infrastructure.database.repositories.market_holiday import (
    PostgresMarketHolidayRepository,
)


def create_company() -> Company:
    return Company(
        company_id=uuid4(),
        name="Reliance Industries",
        legal_name="Reliance Industries Limited",
        country="India",
    )


def create_exchange() -> Exchange:
    return Exchange(
        exchange_id=uuid4(),
        code=ExchangeCode("NSE"),
        name="National Stock Exchange of India",
        country="India",
        timezone=ZoneInfo("Asia/Kolkata"),
    )


def create_instrument(
    company: Company,
    exchange: Exchange,
) -> Instrument:
    return Instrument(
        instrument_id=uuid4(),
        company_id=company.company_id,
        isin=ISIN("INE002A01018"),
        symbol=Symbol("RELIANCE"),
        asset_type=AssetType.EQUITY,
        status=InstrumentStatus.ACTIVE,
        exchange_code=exchange.code,
    )


def test_company_repository_save_and_get(db_session: Session) -> None:
    repository = PostgresCompanyRepository(db_session)
    company = create_company()

    repository.save(company)

    result = repository.get_by_id(company.company_id)

    assert result is not None
    assert result.company_id == company.company_id
    assert result.name == company.name
    assert result.legal_name == company.legal_name
    assert result.country == company.country


def test_exchange_repository_save_and_get(db_session: Session) -> None:
    repository = PostgresExchangeRepository(db_session)
    exchange = create_exchange()

    repository.save(exchange)

    result = repository.get_by_id(exchange.exchange_id)

    assert result is not None
    assert result.exchange_id == exchange.exchange_id
    assert str(result.code) == "NSE"
    assert result.name == exchange.name
    assert result.country == exchange.country
    assert str(result.timezone) == "Asia/Kolkata"


def test_exchange_repository_get_by_code(db_session: Session) -> None:
    repository = PostgresExchangeRepository(db_session)
    exchange = create_exchange()

    repository.save(exchange)

    result = repository.get_by_code(ExchangeCode("NSE"))

    assert result is not None
    assert result.exchange_id == exchange.exchange_id


def test_instrument_repository_save_and_get_by_id(db_session: Session) -> None:
    company_repository = PostgresCompanyRepository(db_session)
    exchange_repository = PostgresExchangeRepository(db_session)
    instrument_repository = PostgresInstrumentRepository(db_session)

    company = create_company()
    exchange = create_exchange()
    instrument = create_instrument(company, exchange)

    company_repository.save(company)
    exchange_repository.save(exchange)
    instrument_repository.save(instrument)

    result = instrument_repository.get_by_id(instrument.instrument_id)

    assert result is not None
    assert result.instrument_id == instrument.instrument_id
    assert result.company_id == company.company_id
    assert str(result.symbol) == "RELIANCE"
    assert str(result.isin) == "INE002A01018"
    assert result.asset_type == AssetType.EQUITY
    assert result.status == InstrumentStatus.ACTIVE
    assert str(result.exchange_code) == "NSE"


def test_instrument_repository_get_by_symbol_and_exchange(
    db_session: Session,
) -> None:
    company_repository = PostgresCompanyRepository(db_session)
    exchange_repository = PostgresExchangeRepository(db_session)
    instrument_repository = PostgresInstrumentRepository(db_session)

    company = create_company()
    exchange = create_exchange()
    instrument = create_instrument(company, exchange)

    company_repository.save(company)
    exchange_repository.save(exchange)
    instrument_repository.save(instrument)

    result = instrument_repository.get_by_symbol(
        Symbol("RELIANCE"),
        ExchangeCode("NSE"),
    )

    assert result is not None
    assert result.instrument_id == instrument.instrument_id


def test_instrument_repository_get_by_symbol_returns_none_for_missing(
    db_session: Session,
) -> None:
    repository = PostgresInstrumentRepository(db_session)

    result = repository.get_by_symbol(
        Symbol("DOESNOTEXIST"),
        ExchangeCode("NSE"),
    )

    assert result is None


def test_instrument_repository_update(db_session: Session) -> None:
    company_repository = PostgresCompanyRepository(db_session)
    exchange_repository = PostgresExchangeRepository(db_session)
    instrument_repository = PostgresInstrumentRepository(db_session)

    company = create_company()
    exchange = create_exchange()
    instrument = create_instrument(company, exchange)

    company_repository.save(company)
    exchange_repository.save(exchange)
    instrument_repository.save(instrument)

    instrument.suspend()

    instrument_repository.save(instrument)

    result = instrument_repository.get_by_id(instrument.instrument_id)

    assert result is not None
    assert result.status == InstrumentStatus.SUSPENDED
    assert str(result.symbol) == "RELIANCE"

    instrument_repository.save(instrument)

    result = instrument_repository.get_by_id(instrument.instrument_id)

    assert result is not None
    assert result.status == InstrumentStatus.SUSPENDED
    assert str(result.symbol) == "RELIANCE"


def setup_persisted_exchange(
    db_session: Session,
    code: str = "NSE",
) -> Exchange:
    exchange_repository = PostgresExchangeRepository(db_session)
    existing_exchange = exchange_repository.get_by_code(ExchangeCode(code))
    if existing_exchange is not None:
        return existing_exchange

    exchange = Exchange(
        exchange_id=uuid4(),
        code=ExchangeCode(code),
        name=f"Exchange {code}",
        country="India",
        timezone=ZoneInfo("Asia/Kolkata"),
    )
    exchange_repository.save(exchange)
    return exchange


def setup_persisted_instrument(
    db_session: Session,
    symbol: str = "RELIANCE",
    isin: str = "INE002A01018",
    exchange_code: str = "NSE",
) -> Instrument:
    company_repository = PostgresCompanyRepository(db_session)
    instrument_repository = PostgresInstrumentRepository(db_session)

    company = Company(
        company_id=uuid4(),
        name=f"Company {symbol}",
        legal_name=f"Company {symbol} Limited",
        country="India",
    )
    target_exchange = setup_persisted_exchange(db_session, code=exchange_code)


    instrument = Instrument(
        instrument_id=uuid4(),
        company_id=company.company_id,
        isin=ISIN(isin),
        symbol=Symbol(symbol),
        asset_type=AssetType.EQUITY,
        status=InstrumentStatus.ACTIVE,
        exchange_code=target_exchange.code,
    )

    company_repository.save(company)
    instrument_repository.save(instrument)
    return instrument


def create_market_bar(
    instrument_id: UUID,
    timestamp: datetime,
    *,
    close: Decimal = Decimal("105.0"),
    open_price: Decimal | None = None,
    high: Decimal | None = None,
    low: Decimal | None = None,
    volume: Decimal = Decimal("1000.0"),
) -> MarketBar:
    return MarketBar(
        instrument_id=instrument_id,
        timestamp=timestamp,
        open=open_price if open_price is not None else close - Decimal("5.0"),
        high=high if high is not None else close + Decimal("5.0"),
        low=low if low is not None else close - Decimal("10.0"),
        close=close,
        volume=volume,
    )



def test_market_bar_repository_get_bars_chronological_order(
    db_session: Session,
) -> None:
    instrument = setup_persisted_instrument(db_session)
    repository = PostgresMarketBarRepository(db_session)

    t1 = datetime(2026, 9, 1, tzinfo=UTC)
    t2 = datetime(2026, 9, 2, tzinfo=UTC)
    t3 = datetime(2026, 9, 3, tzinfo=UTC)

    repository.save(create_market_bar(instrument.instrument_id, t3))
    repository.save(create_market_bar(instrument.instrument_id, t1))
    repository.save(create_market_bar(instrument.instrument_id, t2))

    bars = repository.get_bars(
        instrument_id=instrument.instrument_id,
        start=t1,
        end=t3,
    )

    assert len(bars) == 3
    assert [bar.timestamp for bar in bars] == [t1, t2, t3]


def test_market_bar_repository_get_bars_filters_by_window_inclusive(
    db_session: Session,
) -> None:
    instrument = setup_persisted_instrument(db_session)
    repository = PostgresMarketBarRepository(db_session)

    t_before = datetime(2026, 9, 1, tzinfo=UTC)
    t_start = datetime(2026, 9, 2, tzinfo=UTC)
    t_mid = datetime(2026, 9, 3, tzinfo=UTC)
    t_end = datetime(2026, 9, 4, tzinfo=UTC)
    t_after = datetime(2026, 9, 5, tzinfo=UTC)

    for ts in (t_before, t_start, t_mid, t_end, t_after):
        repository.save(create_market_bar(instrument.instrument_id, ts))

    bars = repository.get_bars(
        instrument_id=instrument.instrument_id,
        start=t_start,
        end=t_end,
    )

    assert len(bars) == 3
    assert bars[0].timestamp == t_start
    assert bars[1].timestamp == t_mid
    assert bars[2].timestamp == t_end
    timestamps = {bar.timestamp for bar in bars}
    assert t_before not in timestamps
    assert t_after not in timestamps


def test_market_bar_repository_get_bars_isolates_by_instrument(
    db_session: Session,
) -> None:
    inst1 = setup_persisted_instrument(db_session, symbol="INSTONE", isin="INE001A01011")
    inst2 = setup_persisted_instrument(db_session, symbol="INSTTWO", isin="INE002B02022")
    repository = PostgresMarketBarRepository(db_session)

    t1 = datetime(2026, 9, 1, tzinfo=UTC)
    t2 = datetime(2026, 9, 2, tzinfo=UTC)

    repository.save(create_market_bar(inst1.instrument_id, t1, close=Decimal("100.0")))
    repository.save(create_market_bar(inst1.instrument_id, t2, close=Decimal("101.0")))
    repository.save(create_market_bar(inst2.instrument_id, t1, close=Decimal("200.0")))
    repository.save(create_market_bar(inst2.instrument_id, t2, close=Decimal("201.0")))

    bars1 = repository.get_bars(inst1.instrument_id, t1, t2)
    assert len(bars1) == 2
    assert all(bar.instrument_id == inst1.instrument_id for bar in bars1)
    assert [bar.close for bar in bars1] == [Decimal("100.0"), Decimal("101.0")]

    bars2 = repository.get_bars(inst2.instrument_id, t1, t2)
    assert len(bars2) == 2
    assert all(bar.instrument_id == inst2.instrument_id for bar in bars2)
    assert [bar.close for bar in bars2] == [Decimal("200.0"), Decimal("201.0")]


def test_market_bar_repository_get_bars_empty_range(
    db_session: Session,
) -> None:
    instrument = setup_persisted_instrument(db_session)
    repository = PostgresMarketBarRepository(db_session)

    repository.save(create_market_bar(instrument.instrument_id, datetime(2026, 9, 1, tzinfo=UTC)))

    empty_bars = repository.get_bars(
        instrument_id=instrument.instrument_id,
        start=datetime(2026, 10, 1, tzinfo=UTC),
        end=datetime(2026, 10, 5, tzinfo=UTC),
    )
    assert empty_bars == []

    nonexistent_bars = repository.get_bars(
        instrument_id=uuid4(),
        start=datetime(2026, 9, 1, tzinfo=UTC),
        end=datetime(2026, 9, 5, tzinfo=UTC),
    )
    assert nonexistent_bars == []


def test_market_bar_repository_get_latest_bar(
    db_session: Session,
) -> None:
    instrument = setup_persisted_instrument(db_session)
    repository = PostgresMarketBarRepository(db_session)

    t1 = datetime(2026, 9, 1, tzinfo=UTC)
    t2 = datetime(2026, 9, 5, tzinfo=UTC)
    t3 = datetime(2026, 9, 3, tzinfo=UTC)

    repository.save(create_market_bar(instrument.instrument_id, t1, close=Decimal("101.0")))
    repository.save(create_market_bar(instrument.instrument_id, t2, close=Decimal("105.0")))
    repository.save(create_market_bar(instrument.instrument_id, t3, close=Decimal("103.0")))

    latest = repository.get_latest_bar(instrument.instrument_id)

    assert latest is not None
    assert latest.instrument_id == instrument.instrument_id
    assert latest.timestamp == t2
    assert latest.close == Decimal("105.0")


def test_market_bar_repository_get_latest_bar_returns_none_when_empty(
    db_session: Session,
) -> None:
    instrument = setup_persisted_instrument(db_session)
    repository = PostgresMarketBarRepository(db_session)

    assert repository.get_latest_bar(instrument.instrument_id) is None
    assert repository.get_latest_bar(uuid4()) is None


def test_market_bar_repository_returns_decimal_values(
    db_session: Session,
) -> None:
    instrument = setup_persisted_instrument(db_session)
    repository = PostgresMarketBarRepository(db_session)

    ts = datetime(2026, 9, 1, tzinfo=UTC)
    bar = MarketBar(
        instrument_id=instrument.instrument_id,
        timestamp=ts,
        open=Decimal("100.123456"),
        high=Decimal("110.654321"),
        low=Decimal("95.000001"),
        close=Decimal("105.500000"),
        volume=Decimal("1000000.123456"),
    )
    repository.save(bar)

    bars = repository.get_bars(instrument.instrument_id, ts, ts)
    assert len(bars) == 1
    retrieved_bar = bars[0]

    assert isinstance(retrieved_bar.open, Decimal)
    assert isinstance(retrieved_bar.high, Decimal)
    assert isinstance(retrieved_bar.low, Decimal)
    assert isinstance(retrieved_bar.close, Decimal)
    assert isinstance(retrieved_bar.volume, Decimal)

    assert retrieved_bar.open == Decimal("100.123456")
    assert retrieved_bar.high == Decimal("110.654321")
    assert retrieved_bar.low == Decimal("95.000001")
    assert retrieved_bar.close == Decimal("105.500000")
    assert retrieved_bar.volume == Decimal("1000000.123456")

    latest = repository.get_latest_bar(instrument.instrument_id)
    assert latest is not None
    assert isinstance(latest.open, Decimal)
    assert isinstance(latest.high, Decimal)
    assert isinstance(latest.low, Decimal)
    assert isinstance(latest.close, Decimal)
    assert isinstance(latest.volume, Decimal)

    assert latest.open == Decimal("100.123456")
    assert latest.high == Decimal("110.654321")
    assert latest.low == Decimal("95.000001")
    assert latest.close == Decimal("105.500000")
    assert latest.volume == Decimal("1000000.123456")


def test_market_bar_repository_bulk_upsert_500_bars(
    db_session: Session,
) -> None:
    instrument = setup_persisted_instrument(db_session)
    repository = PostgresMarketBarRepository(db_session)

    base_time = datetime(2026, 1, 1, 9, 15, tzinfo=UTC)
    bars = [
        MarketBar(
            instrument_id=instrument.instrument_id,
            timestamp=base_time + timedelta(minutes=i),
            open=Decimal(f"{100 + i}.123456"),
            high=Decimal(f"{110 + i}.654321"),
            low=Decimal(f"{90 + i}.000001"),
            close=Decimal(f"{105 + i}.500000"),
            volume=Decimal(f"{1000 + i}.123456"),
        )
        for i in range(500)
    ]

    statements: list[str] = []

    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
        statements.append(statement)

    engine = db_session.get_bind()
    event.listen(engine, "before_cursor_execute", before_cursor_execute)
    try:
        repository.upsert_bars(bars)
    finally:
        event.remove(engine, "before_cursor_execute", before_cursor_execute)

    # Verify bulk execution: exactly 1 INSERT ON CONFLICT statement executed
    assert len(statements) == 1
    assert "INSERT INTO market_bars" in statements[0]
    assert "ON CONFLICT" in statements[0]

    # Verify all 500 bars persisted
    persisted_bars = repository.get_bars(
        instrument_id=instrument.instrument_id,
        start=base_time,
        end=base_time + timedelta(minutes=499),
    )
    assert len(persisted_bars) == 500

    # Verify first and last bar Decimal precision
    first = persisted_bars[0]
    assert first.timestamp == base_time
    assert first.open == Decimal("100.123456")
    assert first.high == Decimal("110.654321")
    assert first.low == Decimal("90.000001")
    assert first.close == Decimal("105.500000")
    assert first.volume == Decimal("1000.123456")

    last = persisted_bars[-1]
    assert last.timestamp == base_time + timedelta(minutes=499)
    assert last.open == Decimal("599.123456")
    assert last.high == Decimal("609.654321")
    assert last.low == Decimal("589.000001")
    assert last.close == Decimal("604.500000")
    assert last.volume == Decimal("1499.123456")


def test_market_bar_repository_bulk_upsert_idempotent_and_updates(
    db_session: Session,
) -> None:
    instrument = setup_persisted_instrument(db_session)
    repository = PostgresMarketBarRepository(db_session)

    base_time = datetime(2026, 1, 1, 9, 15, tzinfo=UTC)
    bars = [
        MarketBar(
            instrument_id=instrument.instrument_id,
            timestamp=base_time + timedelta(minutes=i),
            open=Decimal("100.0"),
            high=Decimal("110.0"),
            low=Decimal("90.0"),
            close=Decimal("105.0"),
            volume=Decimal("1000.0"),
        )
        for i in range(500)
    ]

    # 1. Initial insert of 500 bars
    repository.upsert_bars(bars)
    persisted = repository.get_bars(
        instrument_id=instrument.instrument_id,
        start=base_time,
        end=base_time + timedelta(minutes=499),
    )
    assert len(persisted) == 500
    original_pks = {b.timestamp: b.market_bar_id for b in persisted}

    # 2. Re-ingesting the exact same batch does not duplicate
    repository.upsert_bars(bars)
    persisted_reingested = repository.get_bars(
        instrument_id=instrument.instrument_id,
        start=base_time,
        end=base_time + timedelta(minutes=499),
    )
    assert len(persisted_reingested) == 500
    assert {b.timestamp: b.market_bar_id for b in persisted_reingested} == original_pks

    # 3. Updating the bars updates OHLCV and preserves primary keys
    updated_bars = [
        MarketBar(
            instrument_id=instrument.instrument_id,
            timestamp=base_time + timedelta(minutes=i),
            open=Decimal("200.0"),
            high=Decimal("210.0"),
            low=Decimal("190.0"),
            close=Decimal("205.0"),
            volume=Decimal("5000.0"),
        )
        for i in range(500)
    ]
    repository.upsert_bars(updated_bars)
    persisted_updated = repository.get_bars(
        instrument_id=instrument.instrument_id,
        start=base_time,
        end=base_time + timedelta(minutes=499),
    )
    assert len(persisted_updated) == 500
    for bar in persisted_updated:
        assert bar.open == Decimal("200.0")
        assert bar.close == Decimal("205.0")
        assert bar.volume == Decimal("5000.0")
        # Ensure PK was NOT updated unnecessarily
        assert bar.market_bar_id == original_pks[bar.timestamp]


def test_market_bar_repository_bulk_upsert_isolates_multiple_instruments(
    db_session: Session,
) -> None:
    inst1 = setup_persisted_instrument(db_session, symbol="INSTA", isin="INE001A01099")
    inst2 = setup_persisted_instrument(db_session, symbol="INSTB", isin="INE002B02099")
    repository = PostgresMarketBarRepository(db_session)

    base_time = datetime(2026, 1, 1, 9, 15, tzinfo=UTC)
    bars_inst1 = [
        MarketBar(
            instrument_id=inst1.instrument_id,
            timestamp=base_time + timedelta(minutes=i),
            open=Decimal("100.0"),
            high=Decimal("110.0"),
            low=Decimal("90.0"),
            close=Decimal("105.0"),
            volume=Decimal("1000.0"),
        )
        for i in range(500)
    ]
    bars_inst2 = [
        MarketBar(
            instrument_id=inst2.instrument_id,
            timestamp=base_time + timedelta(minutes=i),
            open=Decimal("200.0"),
            high=Decimal("210.0"),
            low=Decimal("190.0"),
            close=Decimal("205.0"),
            volume=Decimal("2000.0"),
        )
        for i in range(500)
    ]

    repository.upsert_bars(bars_inst1)
    repository.upsert_bars(bars_inst2)

    res1 = repository.get_bars(
        inst1.instrument_id,
        base_time,
        base_time + timedelta(minutes=499),
    )
    res2 = repository.get_bars(
        inst2.instrument_id,
        base_time,
        base_time + timedelta(minutes=499),
    )

    assert len(res1) == 500
    assert len(res2) == 500
    assert all(b.instrument_id == inst1.instrument_id for b in res1)
    assert all(b.instrument_id == inst2.instrument_id for b in res2)
    assert all(b.close == Decimal("105.0") for b in res1)
    assert all(b.close == Decimal("205.0") for b in res2)


def test_market_bar_repository_bulk_upsert_empty_and_in_batch_duplicates(
    db_session: Session,
) -> None:
    instrument = setup_persisted_instrument(db_session)
    repository = PostgresMarketBarRepository(db_session)

    # Empty list should not fail or execute queries
    repository.upsert_bars([])

    # In-batch duplicate timestamps: should deduplicate gracefully and keep latest
    base_time = datetime(2026, 1, 1, 9, 15, tzinfo=UTC)
    bar1 = MarketBar(
        instrument_id=instrument.instrument_id,
        timestamp=base_time,
        open=Decimal("100.0"),
        high=Decimal("110.0"),
        low=Decimal("90.0"),
        close=Decimal("105.0"),
        volume=Decimal("1000.0"),
    )
    bar1_modified = MarketBar(
        instrument_id=instrument.instrument_id,
        timestamp=base_time,
        open=Decimal("110.0"),
        high=Decimal("120.0"),
        low=Decimal("100.0"),
        close=Decimal("115.0"),
        volume=Decimal("2000.0"),
    )

    # Should not raise PostgreSQL ON CONFLICT DO UPDATE cannot affect row a second time
    repository.upsert_bars([bar1, bar1_modified])

    persisted = repository.get_bars(instrument.instrument_id, base_time, base_time)
    assert len(persisted) == 1
    assert persisted[0].close == Decimal("115.0")
    assert persisted[0].volume == Decimal("2000.0")


# ===========================================================================
# InstrumentHistoryRepository Tests
# ===========================================================================


def test_instrument_history_save_and_retrieve(
    db_session: Session,
) -> None:
    instrument = setup_persisted_instrument(db_session)
    exchange = setup_persisted_exchange(db_session)
    repository = PostgresInstrumentHistoryRepository(db_session)

    history = InstrumentHistory(
        history_id=uuid4(),
        instrument_id=instrument.instrument_id,
        symbol="OLDRELIANCE",
        exchange_id=exchange.exchange_id,
        effective_from=date(2020, 1, 1),
        effective_to=date(2022, 12, 31),
        reason="Initial listing ticker",
    )
    repository.save(history)

    results = repository.get_by_instrument_id(instrument.instrument_id)

    assert len(results) == 1
    retrieved = results[0]
    assert isinstance(retrieved, InstrumentHistory)
    assert retrieved.history_id == history.history_id
    assert retrieved.instrument_id == instrument.instrument_id
    assert retrieved.symbol == "OLDRELIANCE"
    assert retrieved.exchange_id == exchange.exchange_id
    assert retrieved.effective_from == date(2020, 1, 1)
    assert retrieved.effective_to == date(2022, 12, 31)
    assert retrieved.reason == "Initial listing ticker"
    assert retrieved.is_current is False


def test_instrument_history_filters_by_instrument(
    db_session: Session,
) -> None:
    inst1 = setup_persisted_instrument(db_session, symbol="INSTA", isin="INE001A01011")
    inst2 = setup_persisted_instrument(db_session, symbol="INSTB", isin="INE002B02022")
    exchange = setup_persisted_exchange(db_session)
    repository = PostgresInstrumentHistoryRepository(db_session)

    h1 = InstrumentHistory(
        history_id=uuid4(),
        instrument_id=inst1.instrument_id,
        symbol="INSTA",
        exchange_id=exchange.exchange_id,
        effective_from=date(2020, 1, 1),
    )
    h2 = InstrumentHistory(
        history_id=uuid4(),
        instrument_id=inst2.instrument_id,
        symbol="INSTB",
        exchange_id=exchange.exchange_id,
        effective_from=date(2020, 1, 1),
    )
    repository.save(h1)
    repository.save(h2)

    results1 = repository.get_by_instrument_id(inst1.instrument_id)
    assert len(results1) == 1
    assert results1[0].history_id == h1.history_id
    assert results1[0].instrument_id == inst1.instrument_id

    results2 = repository.get_by_instrument_id(inst2.instrument_id)
    assert len(results2) == 1
    assert results2[0].history_id == h2.history_id
    assert results2[0].instrument_id == inst2.instrument_id


def test_instrument_history_get_active_at(
    db_session: Session,
) -> None:
    instrument = setup_persisted_instrument(db_session)
    exchange = setup_persisted_exchange(db_session)
    repository = PostgresInstrumentHistoryRepository(db_session)

    h1 = InstrumentHistory(
        history_id=uuid4(),
        instrument_id=instrument.instrument_id,
        symbol="TICKERA",
        exchange_id=exchange.exchange_id,
        effective_from=date(2020, 1, 1),
        effective_to=date(2022, 12, 31),
        reason="Old ticker",
    )
    h2 = InstrumentHistory(
        history_id=uuid4(),
        instrument_id=instrument.instrument_id,
        symbol="TICKERB",
        exchange_id=exchange.exchange_id,
        effective_from=date(2023, 1, 1),
        effective_to=None,
        reason="Current ticker",
    )
    repository.save(h1)
    repository.save(h2)

    # Active during h1
    active_h1 = repository.get_active_at(instrument.instrument_id, as_of=date(2021, 6, 15))
    assert active_h1 is not None
    assert active_h1.history_id == h1.history_id
    assert active_h1.symbol == "TICKERA"

    # Exact boundary dates for h1
    active_start = repository.get_active_at(instrument.instrument_id, as_of=date(2020, 1, 1))
    assert active_start is not None
    assert active_start.history_id == h1.history_id

    active_end = repository.get_active_at(instrument.instrument_id, as_of=date(2022, 12, 31))
    assert active_end is not None
    assert active_end.history_id == h1.history_id

    # Active during h2 (open-ended)
    active_h2 = repository.get_active_at(instrument.instrument_id, as_of=date(2023, 6, 1))
    assert active_h2 is not None
    assert active_h2.history_id == h2.history_id
    assert active_h2.symbol == "TICKERB"
    assert active_h2.is_current is True

    # Date prior to any history
    assert repository.get_active_at(instrument.instrument_id, as_of=date(2019, 12, 31)) is None


def test_instrument_history_empty_result(
    db_session: Session,
) -> None:
    repository = PostgresInstrumentHistoryRepository(db_session)
    unknown_id = uuid4()

    assert repository.get_by_instrument_id(unknown_id) == []
    assert repository.get_active_at(unknown_id, as_of=date(2026, 1, 1)) is None


def test_instrument_history_deterministic_order(
    db_session: Session,
) -> None:
    instrument = setup_persisted_instrument(db_session)
    exchange = setup_persisted_exchange(db_session)
    repository = PostgresInstrumentHistoryRepository(db_session)

    h_2024 = InstrumentHistory(
        history_id=uuid4(),
        instrument_id=instrument.instrument_id,
        symbol="TICKERC",
        exchange_id=exchange.exchange_id,
        effective_from=date(2024, 1, 1),
    )
    h_2020 = InstrumentHistory(
        history_id=uuid4(),
        instrument_id=instrument.instrument_id,
        symbol="TICKERA",
        exchange_id=exchange.exchange_id,
        effective_from=date(2020, 1, 1),
        effective_to=date(2021, 12, 31),
    )
    h_2022 = InstrumentHistory(
        history_id=uuid4(),
        instrument_id=instrument.instrument_id,
        symbol="TICKERB",
        exchange_id=exchange.exchange_id,
        effective_from=date(2022, 1, 1),
        effective_to=date(2023, 12, 31),
    )

    # Save out of chronological order
    repository.save(h_2024)
    repository.save(h_2020)
    repository.save(h_2022)

    results = repository.get_by_instrument_id(instrument.instrument_id)
    assert len(results) == 3
    assert [r.effective_from for r in results] == [
        date(2020, 1, 1),
        date(2022, 1, 1),
        date(2024, 1, 1),
    ]


# ===========================================================================
# MarketHolidayRepository Tests
# ===========================================================================


def test_market_holiday_save_and_retrieve_by_date(
    db_session: Session,
) -> None:
    exchange = setup_persisted_exchange(db_session)
    repository = PostgresMarketHolidayRepository(db_session)

    holiday = MarketHoliday(
        holiday_id=uuid4(),
        exchange_id=exchange.exchange_id,
        holiday_date=date(2026, 1, 26),
        name="Republic Day",
        session_type="full_day",
    )
    repository.save(holiday)

    result = repository.get_by_date(exchange.exchange_id, date(2026, 1, 26))

    assert result is not None
    assert isinstance(result, MarketHoliday)
    assert result.holiday_id == holiday.holiday_id
    assert result.exchange_id == exchange.exchange_id
    assert result.holiday_date == date(2026, 1, 26)
    assert result.name == "Republic Day"
    assert result.session_type == "full_day"


def test_market_holiday_get_by_exchange_with_and_without_year(
    db_session: Session,
) -> None:
    exchange = setup_persisted_exchange(db_session)
    repository = PostgresMarketHolidayRepository(db_session)

    h_2025 = MarketHoliday(
        holiday_id=uuid4(),
        exchange_id=exchange.exchange_id,
        holiday_date=date(2025, 8, 15),
        name="Independence Day 2025",
    )
    h_2026_1 = MarketHoliday(
        holiday_id=uuid4(),
        exchange_id=exchange.exchange_id,
        holiday_date=date(2026, 1, 26),
        name="Republic Day 2026",
    )
    h_2026_2 = MarketHoliday(
        holiday_id=uuid4(),
        exchange_id=exchange.exchange_id,
        holiday_date=date(2026, 8, 15),
        name="Independence Day 2026",
    )
    h_2026_3 = MarketHoliday(
        holiday_id=uuid4(),
        exchange_id=exchange.exchange_id,
        holiday_date=date(2026, 10, 2),
        name="Gandhi Jayanti 2026",
    )

    # Save out of order
    for h in (h_2026_2, h_2025, h_2026_3, h_2026_1):
        repository.save(h)

    # Without year: all 4 holidays in ascending order
    all_holidays = repository.get_by_exchange(exchange.exchange_id)
    assert len(all_holidays) == 4
    assert [h.holiday_date for h in all_holidays] == [
        date(2025, 8, 15),
        date(2026, 1, 26),
        date(2026, 8, 15),
        date(2026, 10, 2),
    ]

    # Filtered by year 2026
    holidays_2026 = repository.get_by_exchange(exchange.exchange_id, year=2026)
    assert len(holidays_2026) == 3
    assert [h.holiday_date for h in holidays_2026] == [
        date(2026, 1, 26),
        date(2026, 8, 15),
        date(2026, 10, 2),
    ]

    # Filtered by year 2025
    holidays_2025 = repository.get_by_exchange(exchange.exchange_id, year=2025)
    assert len(holidays_2025) == 1
    assert holidays_2025[0].holiday_date == date(2025, 8, 15)

    # Filtered by year with no holidays
    assert repository.get_by_exchange(exchange.exchange_id, year=2099) == []


def test_market_holiday_filters_by_exchange(
    db_session: Session,
) -> None:
    exchange_a = setup_persisted_exchange(db_session, code="NSE")
    exchange_b = setup_persisted_exchange(db_session, code="BSE")
    repository = PostgresMarketHolidayRepository(db_session)

    target_date = date(2026, 1, 26)
    h_a = MarketHoliday(
        holiday_id=uuid4(),
        exchange_id=exchange_a.exchange_id,
        holiday_date=target_date,
        name="Republic Day NSE",
    )
    h_b = MarketHoliday(
        holiday_id=uuid4(),
        exchange_id=exchange_b.exchange_id,
        holiday_date=target_date,
        name="Republic Day BSE",
    )
    repository.save(h_a)
    repository.save(h_b)

    # Query for Exchange A
    holidays_a = repository.get_by_exchange(exchange_a.exchange_id)
    assert any(h.holiday_id == h_a.holiday_id for h in holidays_a)
    assert not any(h.holiday_id == h_b.holiday_id for h in holidays_a)

    result_a = repository.get_by_date(exchange_a.exchange_id, target_date)
    assert result_a is not None
    assert result_a.holiday_id == h_a.holiday_id

    # Query for Exchange B
    holidays_b = repository.get_by_exchange(exchange_b.exchange_id)
    assert any(h.holiday_id == h_b.holiday_id for h in holidays_b)
    assert not any(h.holiday_id == h_a.holiday_id for h in holidays_b)

    result_b = repository.get_by_date(exchange_b.exchange_id, target_date)
    assert result_b is not None
    assert result_b.holiday_id == h_b.holiday_id


def test_market_holiday_empty_result(
    db_session: Session,
) -> None:
    repository = PostgresMarketHolidayRepository(db_session)
    unknown_id = uuid4()

    assert repository.get_by_exchange(unknown_id) == []
    assert repository.get_by_date(unknown_id, holiday_date=date(2026, 1, 1)) is None


def test_market_holiday_update_existing(
    db_session: Session,
) -> None:
    exchange = setup_persisted_exchange(db_session)
    repository = PostgresMarketHolidayRepository(db_session)

    target_date = date(2026, 11, 1)
    holiday = MarketHoliday(
        holiday_id=uuid4(),
        exchange_id=exchange.exchange_id,
        holiday_date=target_date,
        name="Special Session",
        session_type="full_day",
    )
    repository.save(holiday)

    # Update session_type to early_close and change name
    holiday.session_type = "early_close"
    holiday.name = "Diwali Muhurat Trading"
    repository.save(holiday)

    retrieved = repository.get_by_date(exchange.exchange_id, target_date)
    assert retrieved is not None
    assert retrieved.holiday_id == holiday.holiday_id
    assert retrieved.name == "Diwali Muhurat Trading"
    assert retrieved.session_type == "early_close"


# ===========================================================================
# Fundamental Record Repository — Point-in-Time Tests
# ===========================================================================


def test_fundamental_record_point_in_time_filtering(
    db_session: Session,
) -> None:
    company_repo = PostgresCompanyRepository(db_session)
    exchange_repo = PostgresExchangeRepository(db_session)
    instrument_repo = PostgresInstrumentRepository(db_session)

    company = create_company()
    exchange = create_exchange()
    instrument = create_instrument(company, exchange)

    company_repo.save(company)
    exchange_repo.save(exchange)
    instrument_repo.save(instrument)

    repository = PostgresFundamentalRecordRepository(db_session)

    # Record 1: Q1, published at 10:00, available at 12:00
    r1 = FundamentalRecord(
        instrument_id=instrument.instrument_id,
        period_end=date(2026, 3, 31),
        metric_name="revenue",
        value=Decimal("1000000"),
        fiscal_year=2026,
        fiscal_quarter=1,
        published_at=datetime(2026, 4, 15, 10, 0, tzinfo=UTC),
        available_at=datetime(2026, 4, 15, 12, 0, tzinfo=UTC),
    )
    # Record 2: Q2, published at 10:00, available at 12:00
    r2 = FundamentalRecord(
        instrument_id=instrument.instrument_id,
        period_end=date(2026, 6, 30),
        metric_name="revenue",
        value=Decimal("1200000"),
        fiscal_year=2026,
        fiscal_quarter=2,
        published_at=datetime(2026, 7, 15, 10, 0, tzinfo=UTC),
        available_at=datetime(2026, 7, 15, 12, 0, tzinfo=UTC),
    )
    # Record 3: Q3, published at 10:00, available_at is None (unknown availability)
    r3 = FundamentalRecord(
        instrument_id=instrument.instrument_id,
        period_end=date(2026, 9, 30),
        metric_name="revenue",
        value=Decimal("1500000"),
        fiscal_year=2026,
        fiscal_quarter=3,
        published_at=datetime(2026, 10, 15, 10, 0, tzinfo=UTC),
        available_at=None,
    )
    repository.save(r1)
    repository.save(r2)
    repository.save(r3)

    # 1. Before r1 is available:
    as_of_before_r1 = datetime(2026, 4, 15, 11, 59, tzinfo=UTC)
    results = repository.get_as_of(instrument.instrument_id, as_of=as_of_before_r1)
    assert len(results) == 0

    # 2. Exactly at r1 available_at:
    as_of_at_r1 = datetime(2026, 4, 15, 12, 0, tzinfo=UTC)
    results = repository.get_as_of(instrument.instrument_id, as_of=as_of_at_r1)
    assert len(results) == 1
    assert results[0].fundamental_record_id == r1.fundamental_record_id

    # 3. After r1 available, before r2 available:
    as_of_between = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
    results = repository.get_as_of(instrument.instrument_id, as_of=as_of_between)
    assert len(results) == 1
    assert results[0].fundamental_record_id == r1.fundamental_record_id

    # 4. At r2 available_at: r1 and r2 visible, but r3 (available_at=None) excluded
    as_of_at_r2 = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
    results = repository.get_as_of(instrument.instrument_id, as_of=as_of_at_r2)
    assert len(results) == 2
    assert [r.fundamental_record_id for r in results] == [
        r1.fundamental_record_id,
        r2.fundamental_record_id,
    ]

    # 5. Far future: r3 still excluded because available_at is None
    as_of_future = datetime(2027, 1, 1, 0, 0, tzinfo=UTC)
    results = repository.get_as_of(instrument.instrument_id, as_of=as_of_future)
    assert len(results) == 2

    # 6. Filtering with start_period / end_period / metric_name
    results_filtered = repository.get_as_of(
        instrument.instrument_id,
        as_of=as_of_future,
        start_period=date(2026, 6, 1),
        end_period=date(2026, 6, 30),
    )
    assert len(results_filtered) == 1
    assert results_filtered[0].fundamental_record_id == r2.fundamental_record_id

    # 7. get_by_instrument_and_period with as_of
    assert len(repository.get_by_instrument_and_period(
        instrument.instrument_id, date(2026, 3, 31), as_of=as_of_before_r1
    )) == 0
    assert len(repository.get_by_instrument_and_period(
        instrument.instrument_id, date(2026, 3, 31), as_of=as_of_at_r1
    )) == 1

    # 8. Normal query without as_of preserves existing behavior
    normal_results = repository.get_by_instrument_and_period(
        instrument.instrument_id, date(2026, 3, 31)
    )
    assert len(normal_results) == 1
    assert normal_results[0].fundamental_record_id == r1.fundamental_record_id

    # 9. Naive as_of timestamp raises ValueError
    naive_as_of = datetime(2026, 4, 15)
    import pytest
    with pytest.raises(ValueError, match="timezone-aware"):
        repository.get_as_of(instrument.instrument_id, as_of=naive_as_of)
    with pytest.raises(ValueError, match="timezone-aware"):
        repository.get_by_instrument_and_period(
            instrument.instrument_id, date(2026, 3, 31), as_of=naive_as_of
        )


# ===========================================================================
# Corporate Action Repository Tests
# ===========================================================================


def test_corporate_action_save_and_get_by_id(
    db_session: Session,
) -> None:
    company_repo = PostgresCompanyRepository(db_session)
    exchange_repo = PostgresExchangeRepository(db_session)
    instrument_repo = PostgresInstrumentRepository(db_session)
    repo = PostgresCorporateActionRepository(db_session)

    company = create_company()
    exchange = create_exchange()
    instrument = create_instrument(company, exchange)
    company_repo.save(company)
    exchange_repo.save(exchange)
    instrument_repo.save(instrument)

    action = CorporateAction(
        instrument_id=instrument.instrument_id,
        action_type=CorporateActionType.SPLIT,
        execution_date=date(2026, 6, 1),
        value=Decimal("2.0"),
        description="2:1 Stock Split",
    )
    repo.save(action)

    retrieved = repo.get_by_id(action.corporate_action_id)
    assert retrieved is not None
    assert retrieved.corporate_action_id == action.corporate_action_id
    assert retrieved.instrument_id == instrument.instrument_id
    assert retrieved.action_type == CorporateActionType.SPLIT
    assert retrieved.execution_date == date(2026, 6, 1)
    assert retrieved.value == Decimal("2.000000")
    assert retrieved.description == "2:1 Stock Split"
    assert retrieved.currency is None


def test_corporate_action_filtering_and_isolation(
    db_session: Session,
) -> None:
    company_repo = PostgresCompanyRepository(db_session)
    exchange_repo = PostgresExchangeRepository(db_session)
    instrument_repo = PostgresInstrumentRepository(db_session)
    repo = PostgresCorporateActionRepository(db_session)

    company = create_company()
    exchange = create_exchange()
    inst_a = create_instrument(company, exchange)
    inst_b = Instrument(
        instrument_id=uuid4(),
        company_id=company.company_id,
        isin=ISIN("INE000000099"),
        symbol=Symbol("OTHER"),
        asset_type=AssetType.EQUITY,
        status=InstrumentStatus.ACTIVE,
        exchange_code=exchange.code,
    )
    company_repo.save(company)
    exchange_repo.save(exchange)
    instrument_repo.save(inst_a)
    instrument_repo.save(inst_b)

    # Actions for inst_a
    act1 = CorporateAction(
        instrument_id=inst_a.instrument_id,
        action_type=CorporateActionType.SPLIT,
        execution_date=date(2026, 3, 1),
        value=Decimal("2.0"),
    )
    act2 = CorporateAction(
        instrument_id=inst_a.instrument_id,
        action_type=CorporateActionType.DIVIDEND,
        execution_date=date(2026, 6, 15),
        value=Decimal("10.0"),
        currency="INR",
        description="Dividend",
    )
    # Action for inst_b
    act_b = CorporateAction(
        instrument_id=inst_b.instrument_id,
        action_type=CorporateActionType.DIVIDEND,
        execution_date=date(2026, 6, 15),
        value=Decimal("5.0"),
    )
    repo.save(act1)
    repo.save(act2)
    repo.save(act_b)

    # 1. Query all for inst_a
    all_a = repo.get_by_instrument(inst_a.instrument_id)
    assert len(all_a) == 2
    assert [a.corporate_action_id for a in all_a] == [
        act1.corporate_action_id,
        act2.corporate_action_id,
    ]

    # 2. Date range filtering
    ranged = repo.get_by_instrument(
        inst_a.instrument_id,
        start_date=date(2026, 6, 1),
        end_date=date(2026, 12, 31),
    )
    assert len(ranged) == 1
    assert ranged[0].corporate_action_id == act2.corporate_action_id

    # 3. Action type filtering
    splits = repo.get_by_instrument(
        inst_a.instrument_id,
        action_type=CorporateActionType.SPLIT,
    )
    assert len(splits) == 1
    assert splits[0].corporate_action_id == act1.corporate_action_id

    # 4. By date
    on_date = repo.get_by_instrument_and_date(
        inst_a.instrument_id,
        execution_date=date(2026, 6, 15),
    )
    assert len(on_date) == 1
    assert on_date[0].corporate_action_id == act2.corporate_action_id

    # 5. Isolation: inst_b does not see inst_a's actions
    all_b = repo.get_by_instrument(inst_b.instrument_id)
    assert len(all_b) == 1
    assert all_b[0].corporate_action_id == act_b.corporate_action_id


def test_corporate_action_update(
    db_session: Session,
) -> None:
    company_repo = PostgresCompanyRepository(db_session)
    exchange_repo = PostgresExchangeRepository(db_session)
    instrument_repo = PostgresInstrumentRepository(db_session)
    repo = PostgresCorporateActionRepository(db_session)

    company = create_company()
    exchange = create_exchange()
    instrument = create_instrument(company, exchange)
    company_repo.save(company)
    exchange_repo.save(exchange)
    instrument_repo.save(instrument)

    action = CorporateAction(
        instrument_id=instrument.instrument_id,
        action_type=CorporateActionType.DIVIDEND,
        execution_date=date(2026, 9, 1),
        value=Decimal("8.0"),
        currency="INR",
        description="Tentative Dividend",
    )
    repo.save(action)

    action.value = Decimal("10.0")
    action.description = "Final Dividend"
    repo.save(action)

    retrieved = repo.get_by_id(action.corporate_action_id)
    assert retrieved is not None
    assert retrieved.value == Decimal("10.000000")
    assert retrieved.description == "Final Dividend"




