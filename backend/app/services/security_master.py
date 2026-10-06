from datetime import date, timedelta
from uuid import UUID, uuid4

from app.core.exceptions import DataNotFoundError, InstrumentNotFoundError
from app.domain.entities.company import Company
from app.domain.entities.instrument import Instrument
from app.domain.entities.instrument_history import InstrumentHistory
from app.domain.enums.asset_type import AssetType
from app.domain.enums.instrument_status import InstrumentStatus
from app.domain.repositories.company_repository import CompanyRepository
from app.domain.repositories.exchange_repository import ExchangeRepository
from app.domain.repositories.instrument_history_repository import (
    InstrumentHistoryRepository,
)
from app.domain.repositories.instrument_repository import InstrumentRepository
from app.domain.value_objects.exchange_code import ExchangeCode
from app.domain.value_objects.isin import ISIN
from app.domain.value_objects.symbol import Symbol


class SecurityMasterService:
    """
    Service-layer facade managing instruments, companies, and historical ticker identity.
    """

    def __init__(
        self,
        instrument_repository: InstrumentRepository,
        company_repository: CompanyRepository,
        exchange_repository: ExchangeRepository,
        instrument_history_repository: InstrumentHistoryRepository | None = None,
    ) -> None:
        self._instrument_repo = instrument_repository
        self._company_repo = company_repository
        self._exchange_repo = exchange_repository
        self._history_repo = instrument_history_repository

    # -----------------------------------------------------------------------
    # Instrument Resolution
    # -----------------------------------------------------------------------

    def resolve_instrument(self, instrument_id: UUID) -> Instrument:
        """Resolve instrument by ID or raise InstrumentNotFoundError."""
        instrument = self._instrument_repo.get_by_id(instrument_id)
        if instrument is None:
            raise InstrumentNotFoundError(
                f"Instrument not found: {instrument_id}"
            )
        return instrument

    def get_by_id(self, instrument_id: UUID) -> Instrument | None:
        """Get instrument by ID, returning None if not found."""
        return self._instrument_repo.get_by_id(instrument_id)

    def get_by_symbol(
        self,
        symbol: Symbol | str,
        exchange_code: ExchangeCode | str,
    ) -> Instrument | None:
        """Look up instrument by current symbol and exchange code."""
        sym = symbol if isinstance(symbol, Symbol) else Symbol(str(symbol))
        code = (
            exchange_code
            if isinstance(exchange_code, ExchangeCode)
            else ExchangeCode(str(exchange_code))
        )
        return self._instrument_repo.get_by_symbol(sym, code)

    def get_by_symbol_and_exchange(
        self,
        symbol: Symbol | str,
        exchange_code: ExchangeCode | str,
    ) -> Instrument | None:
        """Alias for get_by_symbol."""
        return self.get_by_symbol(symbol, exchange_code)

    # -----------------------------------------------------------------------
    # Point-in-Time Historical Identity
    # -----------------------------------------------------------------------

    def resolve_symbol_as_of(
        self,
        instrument_id: UUID,
        as_of: date,
    ) -> str:
        """
        Resolve the effective ticker symbol for an instrument on a specific date.

        Uses InstrumentHistory.get_active_at(instrument_id, as_of) if history
        records exist, falling back to the instrument's current symbol.
        """
        instrument = self.resolve_instrument(instrument_id)

        if self._history_repo is not None:
            history = self._history_repo.get_active_at(instrument_id, as_of)
            if history is not None:
                return history.symbol

        return str(instrument.symbol)

    def get_history(self, instrument_id: UUID) -> list[InstrumentHistory]:
        """Return full history timeline for an instrument."""
        if self._history_repo is None:
            return []
        return self._history_repo.get_by_instrument_id(instrument_id)

    def record_symbol_change(
        self,
        instrument_id: UUID,
        new_symbol: Symbol | str,
        effective_from: date,
        reason: str | None = None,
    ) -> InstrumentHistory:
        """
        Record a ticker change: closes previous active history record and
        opens a new one with effective_from. Updates current instrument symbol.
        """
        instrument = self.resolve_instrument(instrument_id)
        exchange = self._exchange_repo.get_by_code(instrument.exchange_code)
        if exchange is None:
            raise DataNotFoundError(
                f"Exchange not found: {instrument.exchange_code}"
            )

        new_sym_str = str(new_symbol).strip().upper()

        if self._history_repo is not None:
            # Close existing active history record if present
            active = self._history_repo.get_active_at(
                instrument_id, effective_from
            )
            if active is not None and active.effective_to is None:
                active.effective_to = effective_from - timedelta(days=1)
                self._history_repo.save(active)

            new_history = InstrumentHistory(
                history_id=uuid4(),
                instrument_id=instrument_id,
                symbol=new_sym_str,
                exchange_id=exchange.exchange_id,
                effective_from=effective_from,
                effective_to=None,
                reason=reason,
            )
            self._history_repo.save(new_history)
        else:
            new_history = InstrumentHistory(
                history_id=uuid4(),
                instrument_id=instrument_id,
                symbol=new_sym_str,
                exchange_id=exchange.exchange_id,
                effective_from=effective_from,
                effective_to=None,
                reason=reason,
            )

        # Update instrument symbol
        instrument.symbol = Symbol(new_sym_str)
        self._instrument_repo.save(instrument)

        return new_history

    # -----------------------------------------------------------------------
    # Registration & Onboarding
    # -----------------------------------------------------------------------

    def register_company(
        self,
        name: str,
        legal_name: str,
        country: str,
        company_id: UUID | None = None,
    ) -> Company:
        """Create and persist a new Company."""
        company = Company(
            company_id=company_id or uuid4(),
            name=name,
            legal_name=legal_name,
            country=country,
        )
        self._company_repo.save(company)
        return company

    def register_instrument(
        self,
        company_id: UUID,
        symbol: Symbol | str,
        isin: ISIN | str,
        exchange_code: ExchangeCode | str,
        asset_type: AssetType = AssetType.EQUITY,
        status: InstrumentStatus = InstrumentStatus.ACTIVE,
        effective_date: date | None = None,
        instrument_id: UUID | None = None,
    ) -> Instrument:
        """
        Create and persist a new Instrument, optionally recording its initial history.
        """
        company = self._company_repo.get_by_id(company_id)
        if company is None:
            raise DataNotFoundError(f"Company not found: {company_id}")

        code = (
            exchange_code
            if isinstance(exchange_code, ExchangeCode)
            else ExchangeCode(str(exchange_code))
        )
        exchange = self._exchange_repo.get_by_code(code)
        if exchange is None:
            raise DataNotFoundError(f"Exchange not found: {code}")

        sym = symbol if isinstance(symbol, Symbol) else Symbol(str(symbol))
        isin_vo = isin if isinstance(isin, ISIN) else ISIN(str(isin))
        inst_id = instrument_id or uuid4()

        instrument = Instrument(
            instrument_id=inst_id,
            company_id=company_id,
            symbol=sym,
            isin=isin_vo,
            exchange_code=code,
            asset_type=asset_type,
            status=status,
        )
        self._instrument_repo.save(instrument)

        if self._history_repo is not None:
            history = InstrumentHistory(
                history_id=uuid4(),
                instrument_id=inst_id,
                symbol=str(sym),
                exchange_id=exchange.exchange_id,
                effective_from=effective_date or date.today(),
                effective_to=None,
                reason="Initial registration",
            )
            self._history_repo.save(history)

        return instrument

    # -----------------------------------------------------------------------
    # Status Lifecycle
    # -----------------------------------------------------------------------

    def activate_instrument(self, instrument_id: UUID) -> Instrument:
        instrument = self.resolve_instrument(instrument_id)
        instrument.activate()
        self._instrument_repo.save(instrument)
        return instrument

    def deactivate_instrument(self, instrument_id: UUID) -> Instrument:
        instrument = self.resolve_instrument(instrument_id)
        instrument.deactivate()
        self._instrument_repo.save(instrument)
        return instrument

    def suspend_instrument(self, instrument_id: UUID) -> Instrument:
        instrument = self.resolve_instrument(instrument_id)
        instrument.suspend()
        self._instrument_repo.save(instrument)
        return instrument

    def delist_instrument(self, instrument_id: UUID) -> Instrument:
        instrument = self.resolve_instrument(instrument_id)
        instrument.delist()
        self._instrument_repo.save(instrument)
        return instrument
