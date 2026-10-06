from datetime import date
from unittest.mock import MagicMock
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from app.core.exceptions import InstrumentNotFoundError
from app.domain.entities.company import Company
from app.domain.entities.exchange import Exchange
from app.domain.entities.instrument import Instrument
from app.domain.entities.instrument_history import InstrumentHistory
from app.domain.enums.asset_type import AssetType
from app.domain.enums.instrument_status import InstrumentStatus
from app.domain.value_objects.exchange_code import ExchangeCode
from app.domain.value_objects.isin import ISIN
from app.domain.value_objects.symbol import Symbol
from app.services.security_master import SecurityMasterService


@pytest.fixture
def mock_inst_repo() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_comp_repo() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_exch_repo() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_hist_repo() -> MagicMock:
    return MagicMock()


@pytest.fixture
def sec_master(
    mock_inst_repo: MagicMock,
    mock_comp_repo: MagicMock,
    mock_exch_repo: MagicMock,
    mock_hist_repo: MagicMock,
) -> SecurityMasterService:
    return SecurityMasterService(
        instrument_repository=mock_inst_repo,
        company_repository=mock_comp_repo,
        exchange_repository=mock_exch_repo,
        instrument_history_repository=mock_hist_repo,
    )


@pytest.fixture
def sample_company() -> Company:
    return Company(
        company_id=uuid4(),
        name="Tata Motors",
        legal_name="Tata Motors Limited",
        country="India",
    )


@pytest.fixture
def sample_exchange() -> Exchange:
    return Exchange(
        exchange_id=uuid4(),
        code=ExchangeCode("NSE"),
        name="NSE",
        country="India",
        timezone=ZoneInfo("Asia/Kolkata"),
    )


@pytest.fixture
def sample_instrument(
    sample_company: Company,
    sample_exchange: Exchange,
) -> Instrument:
    return Instrument(
        instrument_id=uuid4(),
        company_id=sample_company.company_id,
        symbol=Symbol("TATAMOTORS"),
        isin=ISIN("INE155A01022"),
        exchange_code=sample_exchange.code,
        asset_type=AssetType.EQUITY,
        status=InstrumentStatus.ACTIVE,
    )


class TestSecurityMasterResolution:
    def test_resolve_instrument_success(
        self,
        sec_master: SecurityMasterService,
        mock_inst_repo: MagicMock,
        sample_instrument: Instrument,
    ) -> None:
        mock_inst_repo.get_by_id.return_value = sample_instrument
        result = sec_master.resolve_instrument(sample_instrument.instrument_id)
        assert result == sample_instrument

    def test_resolve_instrument_not_found(
        self,
        sec_master: SecurityMasterService,
        mock_inst_repo: MagicMock,
    ) -> None:
        mock_inst_repo.get_by_id.return_value = None
        with pytest.raises(InstrumentNotFoundError):
            sec_master.resolve_instrument(uuid4())

    def test_get_by_symbol(
        self,
        sec_master: SecurityMasterService,
        mock_inst_repo: MagicMock,
        sample_instrument: Instrument,
    ) -> None:
        mock_inst_repo.get_by_symbol.return_value = sample_instrument
        result = sec_master.get_by_symbol("TATAMOTORS", "NSE")
        assert result == sample_instrument
        assert sec_master.get_by_symbol_and_exchange("TATAMOTORS", "NSE") == sample_instrument


class TestPointInTimeIdentity:
    def test_resolve_symbol_as_of_uses_history(
        self,
        sec_master: SecurityMasterService,
        mock_inst_repo: MagicMock,
        mock_hist_repo: MagicMock,
        sample_instrument: Instrument,
    ) -> None:
        mock_inst_repo.get_by_id.return_value = sample_instrument
        mock_hist_repo.get_active_at.return_value = InstrumentHistory(
            history_id=uuid4(),
            instrument_id=sample_instrument.instrument_id,
            symbol="TELCO",
            exchange_id=uuid4(),
            effective_from=date(1990, 1, 1),
            effective_to=date(2003, 7, 28),
        )

        sym = sec_master.resolve_symbol_as_of(
            sample_instrument.instrument_id,
            date(2000, 1, 1),
        )
        assert sym == "TELCO"

    def test_resolve_symbol_as_of_falls_back_to_current(
        self,
        sec_master: SecurityMasterService,
        mock_inst_repo: MagicMock,
        mock_hist_repo: MagicMock,
        sample_instrument: Instrument,
    ) -> None:
        mock_inst_repo.get_by_id.return_value = sample_instrument
        mock_hist_repo.get_active_at.return_value = None

        sym = sec_master.resolve_symbol_as_of(
            sample_instrument.instrument_id,
            date(2026, 1, 1),
        )
        assert sym == "TATAMOTORS"

    def test_record_symbol_change(
        self,
        sec_master: SecurityMasterService,
        mock_inst_repo: MagicMock,
        mock_exch_repo: MagicMock,
        mock_hist_repo: MagicMock,
        sample_instrument: Instrument,
        sample_exchange: Exchange,
    ) -> None:
        mock_inst_repo.get_by_id.return_value = sample_instrument
        mock_exch_repo.get_by_code.return_value = sample_exchange

        old_active = InstrumentHistory(
            history_id=uuid4(),
            instrument_id=sample_instrument.instrument_id,
            symbol="OLDTICKER",
            exchange_id=sample_exchange.exchange_id,
            effective_from=date(2020, 1, 1),
            effective_to=None,
        )
        mock_hist_repo.get_active_at.return_value = old_active

        new_hist = sec_master.record_symbol_change(
            instrument_id=sample_instrument.instrument_id,
            new_symbol="NEWTICKER",
            effective_from=date(2026, 6, 1),
            reason="Rebranding",
        )

        assert new_hist.symbol == "NEWTICKER"
        assert old_active.effective_to == date(2026, 5, 31)
        assert sample_instrument.symbol == Symbol("NEWTICKER")
        mock_inst_repo.save.assert_called_with(sample_instrument)
        mock_hist_repo.save.assert_called()


class TestOnboardingAndLifecycle:
    def test_register_company(
        self,
        sec_master: SecurityMasterService,
        mock_comp_repo: MagicMock,
    ) -> None:
        comp = sec_master.register_company("Infosys", "Infosys Limited", "India")
        assert comp.name == "Infosys"
        mock_comp_repo.save.assert_called_once_with(comp)

    def test_register_instrument(
        self,
        sec_master: SecurityMasterService,
        mock_comp_repo: MagicMock,
        mock_exch_repo: MagicMock,
        mock_inst_repo: MagicMock,
        mock_hist_repo: MagicMock,
        sample_company: Company,
        sample_exchange: Exchange,
    ) -> None:
        mock_comp_repo.get_by_id.return_value = sample_company
        mock_exch_repo.get_by_code.return_value = sample_exchange

        inst = sec_master.register_instrument(
            company_id=sample_company.company_id,
            symbol="INFY",
            isin="INE009A01021",
            exchange_code="NSE",
        )
        assert inst.symbol == Symbol("INFY")
        mock_inst_repo.save.assert_called_once_with(inst)
        mock_hist_repo.save.assert_called_once()

    def test_lifecycle_transitions(
        self,
        sec_master: SecurityMasterService,
        mock_inst_repo: MagicMock,
        sample_instrument: Instrument,
    ) -> None:
        mock_inst_repo.get_by_id.return_value = sample_instrument

        sec_master.deactivate_instrument(sample_instrument.instrument_id)
        assert sample_instrument.status == InstrumentStatus.INACTIVE

        sec_master.activate_instrument(sample_instrument.instrument_id)
        assert sample_instrument.status == InstrumentStatus.ACTIVE

        sec_master.suspend_instrument(sample_instrument.instrument_id)
        assert sample_instrument.status == InstrumentStatus.SUSPENDED

        sec_master.delist_instrument(sample_instrument.instrument_id)
        assert sample_instrument.status == InstrumentStatus.DELISTED
