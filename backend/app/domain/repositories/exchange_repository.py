from abc import ABC, abstractmethod
from uuid import UUID

from app.domain.entities.exchange import Exchange
from app.domain.value_objects.exchange_code import ExchangeCode


class ExchangeRepository(ABC):
    @abstractmethod
    def get_by_id(self, exchange_id: UUID) -> Exchange | None:
        raise NotImplementedError

    @abstractmethod
    def get_by_code(self, code: ExchangeCode) -> Exchange | None:
        raise NotImplementedError

    @abstractmethod
    def save(self, exchange: Exchange) -> None:
        raise NotImplementedError
