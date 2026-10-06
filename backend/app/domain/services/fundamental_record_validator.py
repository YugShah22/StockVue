from collections.abc import Sequence
from datetime import date
from uuid import UUID

from app.core.exceptions import DataQualityError
from app.domain.entities.fundamental_record import FundamentalRecord


class FundamentalRecordValidator:
    """Validates a batch of FundamentalRecord objects for data-quality issues."""

    @staticmethod
    def validate_batch(
        records: Sequence[FundamentalRecord],
        instrument_id: UUID,
        start_period: date,
        end_period: date,
    ) -> None:
        """
        Validate a batch of fundamental records returned by a provider.

        Raises DataQualityError on the first violation found.
        """
        if not records:
            return

        # Key = (period_end, metric_name, fiscal_year, fiscal_quarter)
        seen_keys: set[tuple[date, str, int | None, int | None]] = set()

        for idx, record in enumerate(records):
            # Wrong instrument
            if record.instrument_id != instrument_id:
                raise DataQualityError(
                    f"Record at index {idx} belongs to instrument "
                    f"{record.instrument_id}, expected {instrument_id}."
                )

            # period_end must fall within the requested period range
            if not (start_period <= record.period_end <= end_period):
                raise DataQualityError(
                    f"Record at index {idx} has period_end {record.period_end} "
                    f"outside requested window [{start_period}, {end_period}]."
                )

            # available_at must not be naive
            if (
                record.available_at is not None
                and record.available_at.tzinfo is None
            ):
                raise DataQualityError(
                    f"Record at index {idx} (metric={record.metric_name}, "
                    f"period_end={record.period_end}) has a timezone-naive "
                    "available_at timestamp."
                )

            # published_at must not be naive
            if (
                record.published_at is not None
                and record.published_at.tzinfo is None
            ):
                raise DataQualityError(
                    f"Record at index {idx} (metric={record.metric_name}, "
                    f"period_end={record.period_end}) has a timezone-naive "
                    "published_at timestamp."
                )

            # Duplicate within batch
            key = (
                record.period_end,
                record.metric_name,
                record.fiscal_year,
                record.fiscal_quarter,
            )
            if key in seen_keys:
                raise DataQualityError(
                    f"Duplicate fundamental record at index {idx}: "
                    f"(period_end={record.period_end}, "
                    f"metric={record.metric_name}, "
                    f"fiscal_year={record.fiscal_year}, "
                    f"fiscal_quarter={record.fiscal_quarter}) "
                    "appears more than once in the same batch."
                )
            seen_keys.add(key)
