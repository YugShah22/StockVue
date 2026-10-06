from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Numeric, UniqueConstraint
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

    open: Mapped[Decimal] = mapped_column(
        Numeric(18, 6),
        nullable=False,
    )

    high: Mapped[Decimal] = mapped_column(
        Numeric(18, 6),
        nullable=False,
    )

    low: Mapped[Decimal] = mapped_column(
        Numeric(18, 6),
        nullable=False,
    )

    close: Mapped[Decimal] = mapped_column(
        Numeric(18, 6),
        nullable=False,
    )

    volume: Mapped[Decimal] = mapped_column(
        Numeric(24, 6),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "timestamp",
            name="uq_market_bar_instrument_timestamp",
        ),
    )