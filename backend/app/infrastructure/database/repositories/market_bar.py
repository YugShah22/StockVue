from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
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

    def get_bars(
        self,
        instrument_id: UUID,
        start: datetime,
        end: datetime,
    ) -> list[MarketBar]:
        statement = (
            select(MarketBarModel)
            .where(
                MarketBarModel.instrument_id == instrument_id,
                MarketBarModel.timestamp >= start,
                MarketBarModel.timestamp <= end,
            )
            .order_by(
                MarketBarModel.timestamp.asc(),
            )
        )

        models = self.session.scalars(statement).all()

        return [self._to_domain(model) for model in models]

    def get_latest_bar(
        self,
        instrument_id: UUID,
    ) -> MarketBar | None:
        statement = (
            select(MarketBarModel)
            .where(
                MarketBarModel.instrument_id == instrument_id,
            )
            .order_by(
                MarketBarModel.timestamp.desc(),
            )
            .limit(1)
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

    def upsert_bars(self, bars: Sequence[MarketBar]) -> None:
        if not bars:
            return

        unique_bars: dict[tuple[UUID, datetime], MarketBar] = {}
        for bar in bars:
            unique_bars[(bar.instrument_id, bar.timestamp)] = bar

        values = [
            {
                "market_bar_id": bar.market_bar_id,
                "instrument_id": bar.instrument_id,
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in unique_bars.values()
        ]

        statement = pg_insert(MarketBarModel).values(values)
        statement = statement.on_conflict_do_update(
            index_elements=[MarketBarModel.instrument_id, MarketBarModel.timestamp],
            set_={
                "open": statement.excluded.open,
                "high": statement.excluded.high,
                "low": statement.excluded.low,
                "close": statement.excluded.close,
                "volume": statement.excluded.volume,
            },
        )

        self.session.execute(statement)
        self.session.flush()
        self.session.expire_all()

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
