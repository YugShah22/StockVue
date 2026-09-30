from uuid import UUID, uuid4

from sqlalchemy import Boolean, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.base import Base


class InstrumentModel(Base):
    __tablename__ = "instruments"

    instrument_id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )

    company_id: Mapped[UUID] = mapped_column(
        ForeignKey("companies.company_id"),
        nullable=False,
    )

    exchange_id: Mapped[UUID] = mapped_column(
        ForeignKey("exchanges.exchange_id"),
        nullable=False,
    )

    symbol: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    isin: Mapped[str] = mapped_column(
        String(12),
        nullable=False,
    )

    asset_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    is_primary_listing: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )
