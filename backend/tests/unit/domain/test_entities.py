from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from app.domain.entities.company import Company
from app.domain.entities.exchange import Exchange
from app.domain.entities.instrument import Instrument
from app.domain.enums.asset_type import AssetType
from app.domain.enums.instrument_status import InstrumentStatus
from app.domain.value_objects.exchange_code import ExchangeCode
from app.domain.value_objects.isin import ISIN
from app.domain.value_objects.symbol import Symbol

# -------------------------
# Company
# -------------------------


def test_company_creation() -> None:
    company_id = uuid4()

    company = Company(
        company_id=company_id,
        name="Reliance Industries",
        legal_name="Reliance Industries Limited",
        country="India",
    )

    assert company.company_id == company_id
    assert company.name == "Reliance Industries"
    assert company.legal_name == "Reliance Industries Limited"
    assert company.country == "India"


def test_company_strips_whitespace() -> None:
    company = Company(
        company_id=uuid4(),
        name="  Reliance Industries  ",
        legal_name="  Reliance Industries Limited  ",
        country="  India  ",
    )

    assert company.name == "Reliance Industries"
    assert company.legal_name == "Reliance Industries Limited"
    assert company.country == "India"


def test_company_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        Company(
            company_id=uuid4(),
            name="",
            legal_name="Reliance Industries Limited",
            country="India",
        )


# -------------------------
# Exchange
# -------------------------


def test_exchange_creation() -> None:
    exchange_id = uuid4()

    exchange = Exchange(
        exchange_id=exchange_id,
        code=ExchangeCode("NSE"),
        name="National Stock Exchange of India",
        country="India",
        timezone=ZoneInfo("Asia/Kolkata"),
    )

    assert exchange.exchange_id == exchange_id
    assert exchange.code == ExchangeCode("NSE")
    assert exchange.name == "National Stock Exchange of India"
    assert exchange.country == "India"
    assert exchange.timezone == ZoneInfo("Asia/Kolkata")


def test_exchange_strips_whitespace() -> None:
    exchange = Exchange(
        exchange_id=uuid4(),
        code=ExchangeCode("NSE"),
        name="  National Stock Exchange of India  ",
        country="  India  ",
        timezone=ZoneInfo("Asia/Kolkata"),
    )

    assert exchange.name == "National Stock Exchange of India"
    assert exchange.country == "India"


def test_exchange_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        Exchange(
            exchange_id=uuid4(),
            code=ExchangeCode("NSE"),
            name="",
            country="India",
            timezone=ZoneInfo("Asia/Kolkata"),
        )


# -------------------------
# Instrument
# -------------------------


def create_instrument() -> Instrument:
    return Instrument(
        instrument_id=uuid4(),
        company_id=uuid4(),
        symbol=Symbol("RELIANCE"),
        isin=ISIN("INE002A01018"),
        exchange_code=ExchangeCode("NSE"),
        asset_type=AssetType.EQUITY,
        status=InstrumentStatus.ACTIVE,
    )


def test_instrument_creation() -> None:
    instrument = create_instrument()

    assert instrument.symbol == Symbol("RELIANCE")
    assert instrument.isin == ISIN("INE002A01018")
    assert instrument.exchange_code == ExchangeCode("NSE")
    assert instrument.asset_type == AssetType.EQUITY
    assert instrument.status == InstrumentStatus.ACTIVE


def test_instrument_can_be_deactivated() -> None:
    instrument = create_instrument()

    instrument.deactivate()

    assert instrument.status == InstrumentStatus.INACTIVE


def test_instrument_can_be_activated() -> None:
    instrument = create_instrument()

    instrument.deactivate()
    instrument.activate()

    assert instrument.status == InstrumentStatus.ACTIVE


def test_instrument_can_be_suspended() -> None:
    instrument = create_instrument()

    instrument.suspend()

    assert instrument.status == InstrumentStatus.SUSPENDED


def test_instrument_can_be_delisted() -> None:
    instrument = create_instrument()

    instrument.delist()

    assert instrument.status == InstrumentStatus.DELISTED
