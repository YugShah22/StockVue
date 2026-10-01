from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.base import Base


class MarketBarModel(Base):
    __tablename__ = "market_bars"

    market_bar_id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )
    instrument_id: Mapped[UUID] = mapped_column(
        ForeignKey("instruments.instrument_id"),
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "timestamp",
            name="uq_market_bar_instrument_timestamp",
        ),
    )
