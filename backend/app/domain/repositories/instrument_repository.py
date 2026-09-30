from abc import ABC, abstractmethod
from uuid import UUID

from app.domain.entities.instrument import Instrument
from app.domain.value_objects.exchange_code import ExchangeCode
from app.domain.value_objects.symbol import Symbol


class InstrumentRepository(ABC):
    @abstractmethod
    def get_by_id(self, instrument_id: UUID) -> Instrument | None:
        raise NotImplementedError

    @abstractmethod
    def get_by_symbol(
        self,
        symbol: Symbol,
        exchange_code: ExchangeCode,
    ) -> Instrument | None:
        raise NotImplementedError

    @abstractmethod
    def save(self, instrument: Instrument) -> None:
        raise NotImplementedError
