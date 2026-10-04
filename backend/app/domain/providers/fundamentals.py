from datetime import date
from typing import Protocol
from uuid import UUID

from app.domain.entities.fundamental_record import FundamentalRecord


class FundamentalsProvider(Protocol):
    def get_fundamentals(
        self,
        instrument_id: UUID,
        start_period: date,
        end_period: date,
    ) -> list[FundamentalRecord]:
        ...
