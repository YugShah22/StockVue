from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.entities.instrument import Instrument
from app.domain.enums.asset_type import AssetType
from app.domain.enums.instrument_status import InstrumentStatus
from app.domain.repositories.instrument_repository import InstrumentRepository
from app.domain.value_objects.exchange_code import ExchangeCode
from app.domain.value_objects.isin import ISIN
from app.domain.value_objects.symbol import Symbol
from app.infrastructure.database.models.exchange import ExchangeModel
from app.infrastructure.database.models.instrument import InstrumentModel


class PostgresInstrumentRepository(InstrumentRepository):
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_id(self, instrument_id: UUID) -> Instrument | None:
        model = self.session.get(InstrumentModel, instrument_id)

        if model is None:
            return None

        return self._to_domain(model)

    def get_by_symbol(
        self,
        symbol: Symbol,
        exchange_code: ExchangeCode,
    ) -> Instrument | None:
        statement = (
            select(InstrumentModel)
            .join(
                ExchangeModel,
                InstrumentModel.exchange_id == ExchangeModel.exchange_id,
            )
            .where(
                InstrumentModel.symbol == str(symbol),
                ExchangeModel.code == str(exchange_code),
            )
        )

        model = self.session.scalar(statement)

        if model is None:
            return None

        return self._to_domain(model)

    def save(self, instrument: Instrument) -> None:
        exchange_statement = select(ExchangeModel).where(
            ExchangeModel.code == str(instrument.exchange_code),
        )

        exchange = self.session.scalar(exchange_statement)

        if exchange is None:
            raise ValueError(
                f"Exchange not found: {instrument.exchange_code}",
            )

        model = self.session.get(
            InstrumentModel,
            instrument.instrument_id,
        )

        if model is None:
            model = InstrumentModel(
                instrument_id=instrument.instrument_id,
                company_id=instrument.company_id,
                exchange_id=exchange.exchange_id,
                symbol=str(instrument.symbol),
                isin=str(instrument.isin),
                asset_type=str(instrument.asset_type),
                status=str(instrument.status),
                is_primary_listing=True,
            )
            self.session.add(model)
        else:
            model.company_id = instrument.company_id
            model.exchange_id = exchange.exchange_id
            model.symbol = str(instrument.symbol)
            model.isin = str(instrument.isin)
            model.asset_type = str(instrument.asset_type)
            model.status = str(instrument.status)

        self.session.flush()

    def _to_domain(self, model: InstrumentModel) -> Instrument:
        exchange = self.session.get(
            ExchangeModel,
            model.exchange_id,
        )

        if exchange is None:
            raise ValueError(
                f"Exchange not found: {model.exchange_id}",
            )

        return Instrument(
            instrument_id=model.instrument_id,
            company_id=model.company_id,
            isin=ISIN(model.isin),
            symbol=Symbol(model.symbol),
            asset_type=AssetType(model.asset_type),
            status=InstrumentStatus(model.status),
            exchange_code=ExchangeCode(exchange.code),
        )
