from datetime import date
from uuid import UUID, uuid4

from sqlalchemy import Date, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.base import Base


class InstrumentHistoryModel(Base):
    __tablename__ = "instrument_history"

    history_id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )

    instrument_id: Mapped[UUID] = mapped_column(
        ForeignKey("instruments.instrument_id"),
        nullable=False,
    )

    symbol: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    exchange_id: Mapped[UUID] = mapped_column(
        ForeignKey("exchanges.exchange_id"),
        nullable=False,
    )

    effective_from: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    effective_to: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
    )

    reason: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
