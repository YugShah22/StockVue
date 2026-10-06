from datetime import date
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.entities.market_holiday import MarketHoliday
from app.domain.repositories.market_holiday_repository import (
    MarketHolidayRepository,
)
from app.infrastructure.database.models.market_holiday import (
    MarketHolidayModel,
)


class PostgresMarketHolidayRepository(MarketHolidayRepository):
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_exchange(
        self,
        exchange_id: UUID,
        year: int | None = None,
    ) -> list[MarketHoliday]:
        statement = select(MarketHolidayModel).where(
            MarketHolidayModel.exchange_id == exchange_id,
        )

        if year is not None:
            statement = statement.where(
                MarketHolidayModel.holiday_date >= date(year, 1, 1),
                MarketHolidayModel.holiday_date <= date(year, 12, 31),
            )

        statement = statement.order_by(
            MarketHolidayModel.holiday_date.asc(),
        )

        models = self.session.scalars(statement).all()

        return [self._to_domain(model) for model in models]

    def get_by_date(
        self,
        exchange_id: UUID,
        holiday_date: date,
    ) -> MarketHoliday | None:
        statement = select(MarketHolidayModel).where(
            MarketHolidayModel.exchange_id == exchange_id,
            MarketHolidayModel.holiday_date == holiday_date,
        )

        model = self.session.scalar(statement)

        if model is None:
            return None

        return self._to_domain(model)

    def save(self, holiday: MarketHoliday) -> None:
        model = self.session.get(
            MarketHolidayModel,
            holiday.holiday_id,
        )

        if model is None:
            statement = select(MarketHolidayModel).where(
                MarketHolidayModel.exchange_id == holiday.exchange_id,
                MarketHolidayModel.holiday_date == holiday.holiday_date,
            )
            model = self.session.scalar(statement)

        if model is None:
            model = MarketHolidayModel(
                holiday_id=holiday.holiday_id,
                exchange_id=holiday.exchange_id,
                holiday_date=holiday.holiday_date,
                name=holiday.name,
                session_type=holiday.session_type,
            )
            self.session.add(model)
        else:
            model.exchange_id = holiday.exchange_id
            model.holiday_date = holiday.holiday_date
            model.name = holiday.name
            model.session_type = holiday.session_type

        self.session.flush()

    def _to_domain(
        self,
        model: MarketHolidayModel,
    ) -> MarketHoliday:
        return MarketHoliday(
            holiday_id=model.holiday_id,
            exchange_id=model.exchange_id,
            holiday_date=model.holiday_date,
            name=model.name,
            session_type=model.session_type,
        )
