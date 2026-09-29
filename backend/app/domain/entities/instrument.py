from dataclasses import dataclass
from uuid import UUID

from app.domain.enums.asset_type import AssetType
from app.domain.enums.instrument_status import InstrumentStatus
from app.domain.value_objects.exchange_code import ExchangeCode
from app.domain.value_objects.isin import ISIN
from app.domain.value_objects.symbol import Symbol


@dataclass
class Instrument:
    instrument_id: UUID
    company_id: UUID
    isin: ISIN
    symbol: Symbol
    asset_type: AssetType
    status: InstrumentStatus
    exchange_code: ExchangeCode

    def activate(self) -> None:
        self.status = InstrumentStatus.ACTIVE

    def deactivate(self) -> None:
        self.status = InstrumentStatus.INACTIVE

    def suspend(self) -> None:
        self.status = InstrumentStatus.SUSPENDED

    def delist(self) -> None:
        self.status = InstrumentStatus.DELISTED
