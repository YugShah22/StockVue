from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.entities.exchange import Exchange
from app.domain.repositories.exchange_repository import ExchangeRepository
from app.domain.value_objects.exchange_code import ExchangeCode
from app.infrastructure.database.models.exchange import ExchangeModel


class PostgresExchangeRepository(ExchangeRepository):
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_id(self, exchange_id: UUID) -> Exchange | None:
        model = self.session.get(ExchangeModel, exchange_id)

        if model is None:
            return None

        return Exchange(
            exchange_id=model.exchange_id,
            code=ExchangeCode(model.code),
            name=model.name,
            country=model.country,
            timezone=ZoneInfo(model.timezone),
        )

    def get_by_code(self, code: ExchangeCode) -> Exchange | None:
        statement = select(ExchangeModel).where(
            ExchangeModel.code == str(code),
        )

        model = self.session.scalar(statement)

        if model is None:
            return None

        return Exchange(
            exchange_id=model.exchange_id,
            code=ExchangeCode(model.code),
            name=model.name,
            country=model.country,
            timezone=ZoneInfo(model.timezone)
        )

    def save(self, exchange: Exchange) -> None:
        model = self.session.get(ExchangeModel, exchange.exchange_id)

        if model is None:
            model = ExchangeModel(
                exchange_id=exchange.exchange_id,
                code=str(exchange.code),
                name=exchange.name,
                country=exchange.country,
                timezone=str(exchange.timezone),
            )
            self.session.add(model)
        else:
            model.code = str(exchange.code)
            model.name = exchange.name
            model.country = exchange.country
            model.timezone = str(exchange.timezone)

        self.session.flush()
