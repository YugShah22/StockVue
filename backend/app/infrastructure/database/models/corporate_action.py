from datetime import date
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import Date, ForeignKey, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.base import Base


class CorporateActionModel(Base):
    __tablename__ = "corporate_actions"

    corporate_action_id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )

    instrument_id: Mapped[UUID] = mapped_column(
        ForeignKey("instruments.instrument_id"),
        nullable=False,
    )

    action_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    execution_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    value: Mapped[Decimal] = mapped_column(
        Numeric(24, 6),
        nullable=False,
    )

    currency: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
    )

    description: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "execution_date",
            "action_type",
            name="uq_corporate_action_instrument_date_type",
        ),
    )
