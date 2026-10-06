from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.orm import Session

from app.core.exceptions import DataQualityError
from app.domain.entities.company import Company
from app.domain.entities.exchange import Exchange
from app.domain.entities.fundamental_record import FundamentalRecord
from app.domain.entities.instrument import Instrument
from app.domain.entities.market_bar import MarketBar
from app.domain.enums.asset_type import AssetType
from app.domain.enums.instrument_status import InstrumentStatus
from app.domain.providers.fundamentals import FundamentalsProvider
from app.domain.providers.market_data import MarketDataProvider
from app.domain.value_objects.exchange_code import ExchangeCode
from app.domain.value_objects.isin import ISIN
from app.domain.value_objects.symbol import Symbol
from app.infrastructure.database.repositories.company import (
    PostgresCompanyRepository,
)
from app.infrastructure.database.repositories.exchange import (
    PostgresExchangeRepository,
)
from app.infrastructure.database.repositories.fundamental_record import (
    PostgresFundamentalRecordRepository,
)
from app.infrastructure.database.repositories.ingestion_run import (
    PostgresIngestionRunRepository,
)
from app.infrastructure.database.repositories.instrument import (
    PostgresInstrumentRepository,
)
from app.infrastructure.database.repositories.market_bar import (
    PostgresMarketBarRepository,
)
from app.services.fundamentals_ingestion import FundamentalsIngestionService
from app.services.price_ingestion import PriceIngestionService


class FakeMarketDataProvider(MarketDataProvider):
    def __init__(self, bars: list[MarketBar]) -> None:
        self._bars = bars

    def get_bars(
        self,
        instrument_id: UUID,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> list[MarketBar]:
        return self._bars


class FakeFundamentalsProvider(FundamentalsProvider):
    def __init__(self, records: list[FundamentalRecord]) -> None:
        self._records = records

    def get_fundamentals(
        self,
        instrument_id: UUID,
        start_period: date,
        end_period: date,
    ) -> list[FundamentalRecord]:
        return self._records


def setup_test_instrument(db_session: Session) -> Instrument:
    comp_repo = PostgresCompanyRepository(db_session)
    exch_repo = PostgresExchangeRepository(db_session)
    inst_repo = PostgresInstrumentRepository(db_session)

    company = Company(
        company_id=uuid4(),
        name="HDFC",
        legal_name="HDFC Bank Limited",
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
        symbol=Symbol("HDFCBANK"),
        isin=ISIN("INE040A01034"),
        exchange_code=exchange.code,
        asset_type=AssetType.EQUITY,
        status=InstrumentStatus.ACTIVE,
    )

    comp_repo.save(company)
    exch_repo.save(exchange)
    inst_repo.save(instrument)
    return instrument


def test_price_ingestion_audit_logging_success(db_session: Session) -> None:
    instrument = setup_test_instrument(db_session)
    inst_repo = PostgresInstrumentRepository(db_session)
    bar_repo = PostgresMarketBarRepository(db_session)
    run_repo = PostgresIngestionRunRepository(db_session)

    start = datetime(2026, 9, 1, 9, 15, tzinfo=UTC)
    end = datetime(2026, 9, 1, 15, 30, tzinfo=UTC)
    bar = MarketBar(
        instrument_id=instrument.instrument_id,
        timestamp=start,
        open=Decimal("1500.0"),
        high=Decimal("1520.0"),
        low=Decimal("1490.0"),
        close=Decimal("1510.0"),
        volume=Decimal("10000"),
    )
    provider = FakeMarketDataProvider([bar])

    service = PriceIngestionService(
        instrument_repository=inst_repo,
        market_bar_repository=bar_repo,
        market_data_provider=provider,
        ingestion_run_repository=run_repo,
    )

    result = service.ingest_bars(
        instrument_id=instrument.instrument_id,
        start=start,
        end=end,
        interval="1d",
    )
    assert result.inserted_count == 1

    latest_run = run_repo.get_latest_run("price", instrument.instrument_id)
    assert latest_run is not None
    assert latest_run.status == "SUCCESS"
    assert latest_run.ingestion_type == "price"
    assert latest_run.instrument_id == instrument.instrument_id
    assert latest_run.received_count == 1
    assert latest_run.inserted_count == 1
    assert latest_run.completed_at is not None
    assert latest_run.completed_at >= latest_run.started_at
    assert latest_run.error_message is None


def test_price_ingestion_audit_logging_failure(db_session: Session) -> None:
    instrument = setup_test_instrument(db_session)
    inst_repo = PostgresInstrumentRepository(db_session)
    bar_repo = PostgresMarketBarRepository(db_session)
    run_repo = PostgresIngestionRunRepository(db_session)

    # Provider returns invalid bar (duplicate timestamp or wrong instrument) to trigger failure
    bad_bar = MarketBar(
        instrument_id=uuid4(),  # wrong instrument!
        timestamp=datetime(2026, 9, 1, 9, 15, tzinfo=UTC),
        open=Decimal("100.0"),
        high=Decimal("105.0"),
        low=Decimal("95.0"),
        close=Decimal("100.0"),
        volume=Decimal("100"),
    )
    provider = FakeMarketDataProvider([bad_bar])

    service = PriceIngestionService(
        instrument_repository=inst_repo,
        market_bar_repository=bar_repo,
        market_data_provider=provider,
        ingestion_run_repository=run_repo,
    )

    with pytest.raises(DataQualityError):
        service.ingest_bars(
            instrument_id=instrument.instrument_id,
            start=datetime(2026, 9, 1, 0, 0, tzinfo=UTC),
            end=datetime(2026, 9, 1, 23, 59, tzinfo=UTC),
            interval="1d",
        )

    latest_run = run_repo.get_latest_run("price", instrument.instrument_id)
    assert latest_run is not None
    assert latest_run.status == "FAILED"
    assert latest_run.error_message is not None
    assert "belongs to instrument" in latest_run.error_message


def test_fundamentals_ingestion_audit_logging_success(db_session: Session) -> None:
    instrument = setup_test_instrument(db_session)
    inst_repo = PostgresInstrumentRepository(db_session)
    fund_repo = PostgresFundamentalRecordRepository(db_session)
    run_repo = PostgresIngestionRunRepository(db_session)

    record = FundamentalRecord(
        instrument_id=instrument.instrument_id,
        period_end=date(2026, 3, 31),
        metric_name="ebitda",
        value=Decimal("5000000"),
        fiscal_year=2026,
        fiscal_quarter=4,
    )
    provider = FakeFundamentalsProvider([record])

    service = FundamentalsIngestionService(
        instrument_repository=inst_repo,
        fundamental_record_repository=fund_repo,
        fundamentals_provider=provider,
        ingestion_run_repository=run_repo,
    )

    result = service.ingest(
        instrument_id=instrument.instrument_id,
        start_period=date(2026, 1, 1),
        end_period=date(2026, 12, 31),
    )
    assert result.inserted_count == 1

    runs = run_repo.get_by_instrument(instrument.instrument_id)
    assert len(runs) >= 1
    assert runs[0].status == "SUCCESS"
    assert runs[0].ingestion_type == "fundamentals"
    assert runs[0].received_count == 1
    assert runs[0].inserted_count == 1
