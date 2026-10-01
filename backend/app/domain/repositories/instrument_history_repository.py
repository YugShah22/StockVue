from abc import ABC, abstractmethod
from datetime import date
from uuid import UUID

from app.domain.entities.instrument_history import InstrumentHistory


class InstrumentHistoryRepository(ABC):
    @abstractmethod
    def get_by_instrument_id(
        self,
        instrument_id: UUID,
    ) -> list[InstrumentHistory]:
        raise NotImplementedError

    @abstractmethod
    def get_active_at(
        self,
        instrument_id: UUID,
        as_of: date,
    ) -> InstrumentHistory | None:
        raise NotImplementedError

    @abstractmethod
    def save(self, history: InstrumentHistory) -> None:
        raise NotImplementedError
