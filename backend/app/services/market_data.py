from datetime import datetime
from uuid import UUID

from app.core.exceptions import InstrumentNotFoundError
from app.domain.entities.instrument import Instrument
from app.domain.entities.market_bar import MarketBar
from app.domain.repositories.instrument_repository import InstrumentRepository
from app.domain.repositories.market_bar_repository import MarketBarRepository
from app.domain.value_objects.exchange_code import ExchangeCode
from app.domain.value_objects.symbol import Symbol


class MarketDataService:
    """Service providing historical market data query operations."""

    def __init__(
        self,
        market_bar_repository: MarketBarRepository,
        instrument_repository: InstrumentRepository,
    ) -> None:
        self._bar_repo = market_bar_repository
        self._inst_repo = instrument_repository

    def _resolve_instrument(self, instrument_id: UUID) -> Instrument:
        instrument = self._inst_repo.get_by_id(instrument_id)
        if instrument is None:
            raise InstrumentNotFoundError(
                f"Instrument not found: {instrument_id}"
            )
        return instrument

    @staticmethod
    def _validate_range(start: datetime, end: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("start timestamp must be timezone-aware")
        if end.tzinfo is None:
            raise ValueError("end timestamp must be timezone-aware")
        if start > end:
            raise ValueError(
                f"start timestamp ({start}) cannot be after end timestamp ({end})"
            )

    def get_historical_bars(
        self,
        instrument_id: UUID,
        start: datetime,
        end: datetime,
    ) -> list[MarketBar]:
        """
        Retrieve historical market bars for an instrument in [start, end].

        Returns bars in ascending chronological order. Returns empty list
        if no bars exist for the requested range.
        """
        self._resolve_instrument(instrument_id)
        self._validate_range(start, end)
        return self._bar_repo.get_bars(instrument_id, start, end)

    def get_latest_bar(
        self,
        instrument_id: UUID,
    ) -> MarketBar | None:
        """Retrieve the most recent market bar for an instrument, or None."""
        self._resolve_instrument(instrument_id)
        return self._bar_repo.get_latest_bar(instrument_id)

    def get_bars_by_symbol(
        self,
        symbol: Symbol | str,
        exchange_code: ExchangeCode | str,
        start: datetime,
        end: datetime,
    ) -> list[MarketBar]:
        """Look up instrument by symbol and exchange, then retrieve bars in [start, end]."""
        sym = symbol if isinstance(symbol, Symbol) else Symbol(str(symbol))
        code = (
            exchange_code
            if isinstance(exchange_code, ExchangeCode)
            else ExchangeCode(str(exchange_code))
        )
        instrument = self._inst_repo.get_by_symbol(sym, code)
        if instrument is None:
            raise InstrumentNotFoundError(
                f"Instrument not found: {sym} on {code}"
            )
        return self.get_historical_bars(instrument.instrument_id, start, end)

    def get_latest_bar_by_symbol(
        self,
        symbol: Symbol | str,
        exchange_code: ExchangeCode | str,
    ) -> MarketBar | None:
        """Look up instrument by symbol and exchange, then retrieve latest bar."""
        sym = symbol if isinstance(symbol, Symbol) else Symbol(str(symbol))
        code = (
            exchange_code
            if isinstance(exchange_code, ExchangeCode)
            else ExchangeCode(str(exchange_code))
        )
        instrument = self._inst_repo.get_by_symbol(sym, code)
        if instrument is None:
            raise InstrumentNotFoundError(
                f"Instrument not found: {sym} on {code}"
            )
        return self.get_latest_bar(instrument.instrument_id)
