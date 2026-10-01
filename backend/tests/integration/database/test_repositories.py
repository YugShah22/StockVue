from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.domain.entities.company import Company
from app.domain.entities.exchange import Exchange
from app.domain.entities.instrument import Instrument
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
