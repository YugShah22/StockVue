from datetime import date
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.entities.corporate_action import CorporateAction
from app.domain.enums.corporate_action_type import CorporateActionType
from app.domain.repositories.corporate_action_repository import (
    CorporateActionRepository,
)
from app.infrastructure.database.models.corporate_action import (
    CorporateActionModel,
)


class PostgresCorporateActionRepository(CorporateActionRepository):
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_id(
        self,
        corporate_action_id: UUID,
    ) -> CorporateAction | None:
        model = self.session.get(
            CorporateActionModel,
            corporate_action_id,
        )

        if model is None:
            return None

        return self._to_domain(model)

    def get_by_instrument(
        self,
        instrument_id: UUID,
        start_date: date | None = None,
        end_date: date | None = None,
        action_type: CorporateActionType | None = None,
    ) -> list[CorporateAction]:
        statement = select(CorporateActionModel).where(
            CorporateActionModel.instrument_id == instrument_id
        )

        if start_date is not None:
            statement = statement.where(
                CorporateActionModel.execution_date >= start_date
            )

        if end_date is not None:
            statement = statement.where(
                CorporateActionModel.execution_date <= end_date
            )

        if action_type is not None:
            statement = statement.where(
                CorporateActionModel.action_type == action_type.value
            )

        statement = statement.order_by(
            CorporateActionModel.execution_date.asc(),
            CorporateActionModel.action_type.asc(),
        )

        models = self.session.scalars(statement).all()

        return [self._to_domain(model) for model in models]

    def get_by_instrument_and_date(
        self,
        instrument_id: UUID,
        execution_date: date,
        action_type: CorporateActionType | None = None,
    ) -> list[CorporateAction]:
        statement = select(CorporateActionModel).where(
            CorporateActionModel.instrument_id == instrument_id,
            CorporateActionModel.execution_date == execution_date,
        )

        if action_type is not None:
            statement = statement.where(
                CorporateActionModel.action_type == action_type.value
            )

        statement = statement.order_by(
            CorporateActionModel.action_type.asc(),
        )

        models = self.session.scalars(statement).all()

        return [self._to_domain(model) for model in models]

    def save(self, action: CorporateAction) -> None:
        model = self.session.get(
            CorporateActionModel,
            action.corporate_action_id,
        )

        if model is None:
            model = CorporateActionModel(
                corporate_action_id=action.corporate_action_id,
                instrument_id=action.instrument_id,
                action_type=action.action_type.value,
                execution_date=action.execution_date,
                value=action.value,
                currency=action.currency,
                description=action.description,
            )
            self.session.add(model)
        else:
            model.instrument_id = action.instrument_id
            model.action_type = action.action_type.value
            model.execution_date = action.execution_date
            model.value = action.value
            model.currency = action.currency
            model.description = action.description

        self.session.flush()

    @staticmethod
    def _to_domain(model: CorporateActionModel) -> CorporateAction:
        return CorporateAction(
            corporate_action_id=model.corporate_action_id,
            instrument_id=model.instrument_id,
            action_type=CorporateActionType(model.action_type),
            execution_date=model.execution_date,
            value=model.value,
            currency=model.currency,
            description=model.description,
        )
