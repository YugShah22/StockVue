from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.entities.market_bar import MarketBar
from app.domain.repositories.market_bar_repository import MarketBarRepository
from app.infrastructure.database.models.market_bar import MarketBarModel


class PostgresMarketBarRepository(MarketBarRepository):
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_instrument_and_timestamp(
        self,
        instrument_id: UUID,
        timestamp: datetime,
    ) -> MarketBar | None:
        statement = select(MarketBarModel).where(
            MarketBarModel.instrument_id == instrument_id,
            MarketBarModel.timestamp == timestamp,
        )

        model = self.session.scalar(statement)

        if model is None:
            return None

        return self._to_domain(model)

    def save(self, market_bar: MarketBar) -> None:
        statement = select(MarketBarModel).where(
            MarketBarModel.instrument_id == market_bar.instrument_id,
            MarketBarModel.timestamp == market_bar.timestamp,
        )

        model = self.session.scalar(statement)

        if model is None:
            model = MarketBarModel(
                market_bar_id=market_bar.market_bar_id,
                instrument_id=market_bar.instrument_id,
                timestamp=market_bar.timestamp,
                open=market_bar.open,
                high=market_bar.high,
                low=market_bar.low,
                close=market_bar.close,
                volume=market_bar.volume,
            )
            self.session.add(model)
        else:
            model.open = market_bar.open
            model.high = market_bar.high
            model.low = market_bar.low
            model.close = market_bar.close
            model.volume = market_bar.volume

        self.session.flush()

    def _to_domain(self, model: MarketBarModel) -> MarketBar:
        return MarketBar(
            market_bar_id=model.market_bar_id,
            instrument_id=model.instrument_id,
            timestamp=model.timestamp,
            open=model.open,
            high=model.high,
            low=model.low,
            close=model.close,
            volume=model.volume,
        )
