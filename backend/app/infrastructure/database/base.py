from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


from app.infrastructure.database.models.company import CompanyModel
from app.infrastructure.database.models.exchange import ExchangeModel
from app.infrastructure.database.models.instrument import InstrumentModel
from app.infrastructure.database.models.instrument_history import (
    InstrumentHistoryModel,
)
from app.infrastructure.database.models.market_holiday import (
    MarketHolidayModel,
)