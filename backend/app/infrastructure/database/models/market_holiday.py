from datetime import date
from uuid import UUID, uuid4

from sqlalchemy import Date, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.base import Base


class MarketHolidayModel(Base):
    __tablename__ = "market_holidays"

    holiday_id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )

    exchange_id: Mapped[UUID] = mapped_column(
        ForeignKey("exchanges.exchange_id"),
        nullable=False,
    )

    holiday_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    session_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="full_day",
    )

    __table_args__ = (
        UniqueConstraint(
            "exchange_id",
            "holiday_date",
            name="uq_market_holiday_exchange_date",
        ),
    )
