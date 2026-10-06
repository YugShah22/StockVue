from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.base import Base


class IngestionRunModel(Base):
    __tablename__ = "ingestion_runs"

    run_id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )

    ingestion_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    instrument_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("instruments.instrument_id"),
        nullable=True,
    )

    provider: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    received_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    inserted_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    updated_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    skipped_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
