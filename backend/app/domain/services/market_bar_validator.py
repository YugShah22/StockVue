from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from app.core.exceptions import DataQualityError
from app.domain.entities.market_bar import MarketBar


class MarketBarValidator:

    @staticmethod
    def validate_batch(
        bars: Sequence[MarketBar],
        instrument_id: UUID,
        start: datetime,
        end: datetime,
    ) -> None:
        if not bars:
            return

        seen_timestamps: set[datetime] = set()

        for idx, bar in enumerate(bars):
            if bar.instrument_id != instrument_id:
                raise DataQualityError(
                    f"Bar at index {idx} belongs to instrument "
                    f"{bar.instrument_id}, expected {instrument_id}."
                )

            if bar.timestamp.tzinfo is None:
                raise DataQualityError(
                    f"Bar at index {idx} (timestamp={bar.timestamp}) "
                    "has a timezone-naive timestamp. "
                    "All market timestamps must be timezone-aware."
                )

            if bar.timestamp < start or bar.timestamp > end:
                raise DataQualityError(
                    f"Bar at index {idx} has timestamp {bar.timestamp} "
                    f"outside requested window [{start}, {end}]."
                )

            if bar.timestamp in seen_timestamps:
                raise DataQualityError(
                    f"Duplicate timestamp {bar.timestamp} at index {idx} "
                    "within the same ingestion batch."
                )
            seen_timestamps.add(bar.timestamp)

        timestamps = [bar.timestamp for bar in bars]
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1]:
                raise DataQualityError(
                    f"Timestamps are not in ascending order: "
                    f"{timestamps[i - 1]} followed by {timestamps[i]} "
                    f"(at indices {i - 1}, {i})."
                )
