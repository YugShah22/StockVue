from datetime import date
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.domain.entities.instrument_history import InstrumentHistory
from app.domain.repositories.instrument_history_repository import (
    InstrumentHistoryRepository,
)
from app.infrastructure.database.models.instrument_history import (
    InstrumentHistoryModel,
)


class PostgresInstrumentHistoryRepository(InstrumentHistoryRepository):
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_instrument_id(
        self,
        instrument_id: UUID,
    ) -> list[InstrumentHistory]:
        statement = (
            select(InstrumentHistoryModel)
            .where(
                InstrumentHistoryModel.instrument_id == instrument_id,
            )
            .order_by(
                InstrumentHistoryModel.effective_from.asc(),
            )
        )

        models = self.session.scalars(statement).all()

        return [self._to_domain(model) for model in models]

    def get_active_at(
        self,
        instrument_id: UUID,
        as_of: date,
    ) -> InstrumentHistory | None:
        statement = (
            select(InstrumentHistoryModel)
            .where(
                InstrumentHistoryModel.instrument_id == instrument_id,
                InstrumentHistoryModel.effective_from <= as_of,
                or_(
                    InstrumentHistoryModel.effective_to.is_(None),
                    InstrumentHistoryModel.effective_to >= as_of,
                ),
            )
            .order_by(
                InstrumentHistoryModel.effective_from.desc(),
            )
            .limit(1)
        )

        model = self.session.scalar(statement)

        if model is None:
            return None

        return self._to_domain(model)

    def save(self, history: InstrumentHistory) -> None:
        model = self.session.get(
            InstrumentHistoryModel,
            history.history_id,
        )

        if model is None:
            model = InstrumentHistoryModel(
                history_id=history.history_id,
                instrument_id=history.instrument_id,
                symbol=history.symbol,
                exchange_id=history.exchange_id,
                effective_from=history.effective_from,
                effective_to=history.effective_to,
                reason=history.reason,
            )
            self.session.add(model)
        else:
            model.instrument_id = history.instrument_id
            model.symbol = history.symbol
            model.exchange_id = history.exchange_id
            model.effective_from = history.effective_from
            model.effective_to = history.effective_to
            model.reason = history.reason

        self.session.flush()

    def _to_domain(
        self,
        model: InstrumentHistoryModel,
    ) -> InstrumentHistory:
        return InstrumentHistory(
            history_id=model.history_id,
            instrument_id=model.instrument_id,
            symbol=model.symbol,
            exchange_id=model.exchange_id,
            effective_from=model.effective_from,
            effective_to=model.effective_to,
            reason=model.reason,
        )
