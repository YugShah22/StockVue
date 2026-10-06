from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4


@dataclass
class IngestionRun:
    ingestion_type: str
    started_at: datetime
    provider: str = ""
    instrument_id: UUID | None = None
    completed_at: datetime | None = None
    status: str = "RUNNING"
    received_count: int = 0
    inserted_count: int = 0
    updated_count: int = 0
    skipped_count: int = 0
    error_message: str | None = None
    run_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        self.ingestion_type = self.ingestion_type.strip().lower()
        if not self.ingestion_type:
            raise ValueError("ingestion_type cannot be empty")

        self.provider = self.provider.strip()
        self.status = self.status.strip().upper()
        allowed_statuses = {"RUNNING", "SUCCESS", "FAILED"}
        if self.status not in allowed_statuses:
            raise ValueError(
                f"Invalid status: {self.status!r}. Must be one of: {allowed_statuses}"
            )

        if self.started_at.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")

        if self.completed_at is not None:
            if self.completed_at.tzinfo is None:
                raise ValueError("completed_at must be timezone-aware")
            if self.completed_at < self.started_at:
                raise ValueError("completed_at cannot be earlier than started_at")

        for count_name, count_val in [
            ("received_count", self.received_count),
            ("inserted_count", self.inserted_count),
            ("updated_count", self.updated_count),
            ("skipped_count", self.skipped_count),
        ]:
            if count_val < 0:
                raise ValueError(f"{count_name} cannot be negative")
