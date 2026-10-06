from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.domain.entities.ingestion_run import IngestionRun


class TestIngestionRunEntity:
    def test_valid_creation(self) -> None:
        inst_id = uuid4()
        started = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
        completed = datetime(2026, 9, 1, 10, 1, tzinfo=UTC)

        run = IngestionRun(
            ingestion_type="price",
            instrument_id=inst_id,
            provider="yfinance",
            started_at=started,
            completed_at=completed,
            status="success",
            received_count=100,
            inserted_count=90,
            updated_count=5,
            skipped_count=5,
        )

        assert run.ingestion_type == "price"
        assert run.status == "SUCCESS"
        assert run.provider == "yfinance"
        assert run.received_count == 100
        assert run.run_id is not None

    def test_empty_ingestion_type_raises(self) -> None:
        with pytest.raises(ValueError, match="ingestion_type cannot be empty"):
            IngestionRun(
                ingestion_type="",
                started_at=datetime.now(UTC),
            )

    def test_invalid_status_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid status"):
            IngestionRun(
                ingestion_type="price",
                started_at=datetime.now(UTC),
                status="PENDING",
            )

    def test_naive_started_at_raises(self) -> None:
        with pytest.raises(ValueError, match="started_at must be timezone-aware"):
            IngestionRun(
                ingestion_type="price",
                started_at=datetime(2026, 9, 1),
            )

    def test_naive_completed_at_raises(self) -> None:
        with pytest.raises(ValueError, match="completed_at must be timezone-aware"):
            IngestionRun(
                ingestion_type="price",
                started_at=datetime(2026, 9, 1, tzinfo=UTC),
                completed_at=datetime(2026, 9, 1, 10, 0),
            )

    def test_completed_before_started_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be earlier than started_at"):
            IngestionRun(
                ingestion_type="price",
                started_at=datetime(2026, 9, 1, 10, 0, tzinfo=UTC),
                completed_at=datetime(2026, 9, 1, 9, 0, tzinfo=UTC),
            )

    def test_negative_counts_raise(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            IngestionRun(
                ingestion_type="price",
                started_at=datetime.now(UTC),
                received_count=-1,
            )
