from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.domain.entities.company import Company
from app.domain.entities.exchange import Exchange
from app.domain.entities.instrument import Instrument
from app.domain.entities.market_bar import MarketBar
from app.domain.enums.asset_type import AssetType
from app.domain.enums.instrument_status import InstrumentStatus
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
from app.services.market_data import MarketDataService


def test_market_data_service_historical_reads(db_session: Session) -> None:
    comp_repo = PostgresCompanyRepository(db_session)
    exch_repo = PostgresExchangeRepository(db_session)
    inst_repo = PostgresInstrumentRepository(db_session)
    bar_repo = PostgresMarketBarRepository(db_session)

    company = Company(
        company_id=uuid4(),
        name="TCS",
        legal_name="Tata Consultancy Services",
        country="India",
    )
    exchange = Exchange(
        exchange_id=uuid4(),
        code=ExchangeCode("NSE"),
        name="NSE",
        country="India",
        timezone=ZoneInfo("Asia/Kolkata"),
    )
    instrument = Instrument(
        instrument_id=uuid4(),
        company_id=company.company_id,
        symbol=Symbol("TCS"),
        isin=ISIN("INE467B01029"),
        exchange_code=exchange.code,
        asset_type=AssetType.EQUITY,
        status=InstrumentStatus.ACTIVE,
    )

    comp_repo.save(company)
    exch_repo.save(exchange)
    inst_repo.save(instrument)

    service = MarketDataService(
        market_bar_repository=bar_repo,
        instrument_repository=inst_repo,
    )

    # Initially empty
    empty_bars = service.get_historical_bars(
        instrument.instrument_id,
        datetime(2026, 9, 1, 0, 0, tzinfo=UTC),
        datetime(2026, 9, 5, 0, 0, tzinfo=UTC),
    )
    assert empty_bars == []
    assert service.get_latest_bar(instrument.instrument_id) is None

    # Populate bars out-of-order to ensure chronological ordering on query
    t1 = datetime(2026, 9, 1, 9, 15, tzinfo=UTC)
    t2 = datetime(2026, 9, 2, 9, 15, tzinfo=UTC)
    t3 = datetime(2026, 9, 3, 9, 15, tzinfo=UTC)

    bar2 = MarketBar(
        instrument_id=instrument.instrument_id,
        timestamp=t2,
        open=Decimal("3020.0"),
        high=Decimal("3050.0"),
        low=Decimal("3010.0"),
        close=Decimal("3040.0"),
        volume=Decimal("50000"),
    )
    bar1 = MarketBar(
        instrument_id=instrument.instrument_id,
        timestamp=t1,
        open=Decimal("3000.0"),
        high=Decimal("3030.0"),
        low=Decimal("2990.0"),
        close=Decimal("3020.0"),
        volume=Decimal("45000"),
    )
    bar3 = MarketBar(
        instrument_id=instrument.instrument_id,
        timestamp=t3,
        open=Decimal("3040.0"),
        high=Decimal("3070.0"),
        low=Decimal("3030.0"),
        close=Decimal("3065.0"),
        volume=Decimal("60000"),
    )

    bar_repo.save(bar2)
    bar_repo.save(bar1)
    bar_repo.save(bar3)

    # 1. Historical range query — must be sorted t1, t2, t3
    bars = service.get_historical_bars(
        instrument.instrument_id,
        datetime(2026, 9, 1, 0, 0, tzinfo=UTC),
        datetime(2026, 9, 3, 23, 59, tzinfo=UTC),
    )
    assert len(bars) == 3
    assert [b.timestamp for b in bars] == [t1, t2, t3]

    # 2. Subset range query
    subset = service.get_historical_bars(
        instrument.instrument_id,
        datetime(2026, 9, 2, 0, 0, tzinfo=UTC),
        datetime(2026, 9, 2, 23, 59, tzinfo=UTC),
    )
    assert len(subset) == 1
    assert subset[0].timestamp == t2

    # 3. Latest bar query
    latest = service.get_latest_bar(instrument.instrument_id)
    assert latest is not None
    assert latest.timestamp == t3
    assert latest.close == Decimal("3065.000000")

    # 4. Symbol query
    sym_bars = service.get_bars_by_symbol(
        "TCS",
        "NSE",
        datetime(2026, 9, 1, 0, 0, tzinfo=UTC),
        datetime(2026, 9, 3, 23, 59, tzinfo=UTC),
    )
    assert len(sym_bars) == 3
    latest_sym = service.get_latest_bar_by_symbol("TCS", "NSE")
    assert latest_sym is not None
    assert latest_sym.timestamp == t3
