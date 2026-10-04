from datetime import date
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.entities.fundamental_record import FundamentalRecord
from app.domain.repositories.fundamental_record_repository import (
    FundamentalRecordRepository,
)
from app.infrastructure.database.models.fundamental_record import (
    FundamentalRecordModel,
)


class PostgresFundamentalRecordRepository(FundamentalRecordRepository):
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_id(
        self,
        fundamental_record_id: UUID,
    ) -> FundamentalRecord | None:
        model = self.session.get(
            FundamentalRecordModel,
            fundamental_record_id,
        )

        if model is None:
            return None

        return self._to_domain(model)

    def get_by_instrument_and_period(
        self,
        instrument_id: UUID,
        period_end: date,
    ) -> list[FundamentalRecord]:
        statement = (
            select(FundamentalRecordModel)
            .where(
                FundamentalRecordModel.instrument_id == instrument_id,
                FundamentalRecordModel.period_end == period_end,
            )
            .order_by(
                FundamentalRecordModel.metric_name,
            )
        )

        models = self.session.scalars(statement).all()

        return [self._to_domain(model) for model in models]

    def save(self, record: FundamentalRecord) -> None:
        model = self.session.get(
            FundamentalRecordModel,
            record.fundamental_record_id,
        )

        if model is None:
            model = FundamentalRecordModel(
                fundamental_record_id=record.fundamental_record_id,
                instrument_id=record.instrument_id,
                period_end=record.period_end,
                metric_name=record.metric_name,
                value=record.value,
                fiscal_year=record.fiscal_year,
                fiscal_quarter=record.fiscal_quarter,
                currency=record.currency,
                published_at=record.published_at,
                available_at=record.available_at,
            )
            self.session.add(model)
        else:
            model.instrument_id = record.instrument_id
            model.period_end = record.period_end
            model.metric_name = record.metric_name
            model.value = record.value
            model.fiscal_year = record.fiscal_year
            model.fiscal_quarter = record.fiscal_quarter
            model.currency = record.currency
            model.published_at = record.published_at
            model.available_at = record.available_at

        self.session.flush()

    @staticmethod
    def _to_domain(
        model: FundamentalRecordModel,
    ) -> FundamentalRecord:
        return FundamentalRecord(
            fundamental_record_id=model.fundamental_record_id,
            instrument_id=model.instrument_id,
            period_end=model.period_end,
            metric_name=model.metric_name,
            value=model.value,
            fiscal_year=model.fiscal_year,
            fiscal_quarter=model.fiscal_quarter,
            currency=model.currency,
            published_at=model.published_at,
            available_at=model.available_at,
        )
