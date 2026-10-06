from abc import ABC, abstractmethod
from datetime import date
from uuid import UUID

from app.domain.entities.corporate_action import CorporateAction
from app.domain.enums.corporate_action_type import CorporateActionType


class CorporateActionRepository(ABC):
    @abstractmethod
    def get_by_id(
        self,
        corporate_action_id: UUID,
    ) -> CorporateAction | None:
        raise NotImplementedError

    @abstractmethod
    def get_by_instrument(
        self,
        instrument_id: UUID,
        start_date: date | None = None,
        end_date: date | None = None,
        action_type: CorporateActionType | None = None,
    ) -> list[CorporateAction]:
        raise NotImplementedError

    @abstractmethod
    def get_by_instrument_and_date(
        self,
        instrument_id: UUID,
        execution_date: date,
        action_type: CorporateActionType | None = None,
    ) -> list[CorporateAction]:
        raise NotImplementedError

    @abstractmethod
    def save(
        self,
        action: CorporateAction,
    ) -> None:
        raise NotImplementedError
