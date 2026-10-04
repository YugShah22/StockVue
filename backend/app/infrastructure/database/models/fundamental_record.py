from datetime import date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import Date, DateTime, ForeignKey, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.base import Base


class FundamentalRecordModel(Base):
    __tablename__ = "fundamental_records"

    fundamental_record_id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )

    instrument_id: Mapped[UUID] = mapped_column(
        ForeignKey("instruments.instrument_id"),
        nullable=False,
    )

    period_end: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    metric_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    value: Mapped[Decimal] = mapped_column(
        Numeric(24, 6),
        nullable=False,
    )

    fiscal_year: Mapped[int | None] = mapped_column(
        nullable=True,
    )

    fiscal_quarter: Mapped[int | None] = mapped_column(
        nullable=True,
    )

    currency: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
    )

    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    available_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "period_end",
            "metric_name",
            "fiscal_year",
            "fiscal_quarter",
            name="uq_fundamental_record_period_metric",
        ),
    )
