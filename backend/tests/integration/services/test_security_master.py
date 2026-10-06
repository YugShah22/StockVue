from datetime import date
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.domain.entities.exchange import Exchange
from app.domain.enums.asset_type import AssetType
from app.domain.enums.instrument_status import InstrumentStatus
from app.domain.value_objects.exchange_code import ExchangeCode
from app.infrastructure.database.repositories.company import (
    PostgresCompanyRepository,
)
from app.infrastructure.database.repositories.exchange import (
    PostgresExchangeRepository,
)
from app.infrastructure.database.repositories.instrument import (
    PostgresInstrumentRepository,
)
from app.infrastructure.database.repositories.instrument_history import (
    PostgresInstrumentHistoryRepository,
)
from app.services.security_master import SecurityMasterService


def test_security_master_onboarding_and_resolution(
    db_session: Session,
) -> None:
    comp_repo = PostgresCompanyRepository(db_session)
    exch_repo = PostgresExchangeRepository(db_session)
    inst_repo = PostgresInstrumentRepository(db_session)
    hist_repo = PostgresInstrumentHistoryRepository(db_session)

    # Pre-populate exchange
    exchange = Exchange(
        exchange_id=None,  # type: ignore[arg-type]
        code=ExchangeCode("NSE"),
        name="National Stock Exchange",
        country="India",
        timezone=ZoneInfo("Asia/Kolkata"),
    )
    from uuid import uuid4
    exchange.exchange_id = uuid4()
    exch_repo.save(exchange)

    service = SecurityMasterService(
        instrument_repository=inst_repo,
        company_repository=comp_repo,
        exchange_repository=exch_repo,
        instrument_history_repository=hist_repo,
    )

    # 1. Register company
    company = service.register_company(
        name="Wipro",
        legal_name="Wipro Limited",
        country="India",
    )
    assert company.company_id is not None

    # 2. Register instrument
    instrument = service.register_instrument(
        company_id=company.company_id,
        symbol="WIPRO",
        isin="INE075A01022",
        exchange_code="NSE",
        asset_type=AssetType.EQUITY,
        status=InstrumentStatus.ACTIVE,
        effective_date=date(2000, 1, 1),
    )
    assert instrument.instrument_id is not None

    # 3. Resolve by ID
    resolved = service.resolve_instrument(instrument.instrument_id)
    assert resolved.instrument_id == instrument.instrument_id

    # 4. Resolve by Symbol + Exchange
    by_sym = service.get_by_symbol_and_exchange("WIPRO", "NSE")
    assert by_sym is not None
    assert by_sym.instrument_id == instrument.instrument_id

    # 5. Point-in-time ticker resolution
    sym_as_of = service.resolve_symbol_as_of(
        instrument.instrument_id, date(2020, 1, 1)
    )
    assert sym_as_of == "WIPRO"

    # 6. Record symbol change
    new_hist = service.record_symbol_change(
        instrument_id=instrument.instrument_id,
        new_symbol="WIPRO_NEW",
        effective_from=date(2026, 6, 1),
        reason="Corporate action",
    )
    assert new_hist.symbol == "WIPRO_NEW"

    # Before change date: resolve as original symbol
    assert (
        service.resolve_symbol_as_of(instrument.instrument_id, date(2026, 5, 1))
        == "WIPRO"
    )

    # On or after change date: resolve as new symbol
    assert (
        service.resolve_symbol_as_of(instrument.instrument_id, date(2026, 6, 1))
        == "WIPRO_NEW"
    )

    # 7. Status transitions
    service.deactivate_instrument(instrument.instrument_id)
    assert service.resolve_instrument(instrument.instrument_id).status == InstrumentStatus.INACTIVE
