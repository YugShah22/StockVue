from datetime import date
from decimal import Decimal
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.domain.entities.company import Company
from app.domain.entities.exchange import Exchange
from app.domain.entities.fundamental_record import FundamentalRecord
from app.domain.entities.instrument import Instrument
from app.domain.enums.asset_type import AssetType
from app.domain.enums.instrument_status import InstrumentStatus
from app.domain.providers.fundamentals import FundamentalsProvider
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
from app.infrastructure.database.repositories.instrument import (
    PostgresInstrumentRepository,
)
from app.services.fundamentals_ingestion import FundamentalsIngestionService


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


def test_fundamentals_ingestion_persists_records(
    db_session: Session,
) -> None:
    company_id = uuid4()
    exchange_id = uuid4()
    instrument_id = uuid4()

    company_repository = PostgresCompanyRepository(db_session)
    exchange_repository = PostgresExchangeRepository(db_session)
    instrument_repository = PostgresInstrumentRepository(db_session)
    fundamental_repository = PostgresFundamentalRecordRepository(db_session)

    company_repository.save(
        Company(
            company_id=company_id,
            name="Test Company",
            legal_name="Test Company Limited",
            country="India",
        )
    )

    exchange_repository.save(
        Exchange(
            exchange_id=exchange_id,
            code=ExchangeCode("NSE"),
            name="National Stock Exchange",
            country="India",
            timezone=ZoneInfo("Asia/Kolkata"),
        )
    )

    instrument_repository.save(
        Instrument(
            instrument_id=instrument_id,
            company_id=company_id,
            exchange_code=ExchangeCode("NSE"),
            symbol=Symbol("TEST"),
            isin=ISIN("INE000000000"),
            asset_type=AssetType.EQUITY,
            status=InstrumentStatus.ACTIVE,
        )
    )

    records = [
        FundamentalRecord(
            instrument_id=instrument_id,
            period_end=date(2026, 3, 31),
            metric_name="revenue",
            value=Decimal("1000000"),
            fiscal_year=2026,
            fiscal_quarter=4,
            currency="INR",
        ),
        FundamentalRecord(
            instrument_id=instrument_id,
            period_end=date(2026, 3, 31),
            metric_name="net income",
            value=Decimal("200000"),
            fiscal_year=2026,
            fiscal_quarter=4,
            currency="INR",
        ),
    ]

    provider = FakeFundamentalsProvider(records)

    service = FundamentalsIngestionService(
        instrument_repository=instrument_repository,
        fundamental_record_repository=fundamental_repository,
        fundamentals_provider=provider,
    )

    result = service.ingest(
        instrument_id=instrument_id,
        start_period=date(2026, 1, 1),
        end_period=date(2026, 12, 31),
    )

    assert result.received_count == 2
    assert result.inserted_count == 2
    assert result.updated_count == 0
    assert result.skipped_count == 0

    stored_records = fundamental_repository.get_by_instrument_and_period(
        instrument_id=instrument_id,
        period_end=date(2026, 3, 31),
    )

    assert len(stored_records) == 2

    stored_by_metric = {
        record.metric_name: record for record in stored_records
    }

    assert stored_by_metric["revenue"].value == Decimal("1000000")
    assert stored_by_metric["net income"].value == Decimal("200000")
