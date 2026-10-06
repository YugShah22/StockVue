from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.entities.ingestion_run import IngestionRun
from app.domain.repositories.ingestion_run_repository import (
    IngestionRunRepository,
)
from app.infrastructure.database.models.ingestion_run import (
    IngestionRunModel,
)


class PostgresIngestionRunRepository(IngestionRunRepository):
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_id(
        self,
        run_id: UUID,
    ) -> IngestionRun | None:
        model = self.session.get(
            IngestionRunModel,
            run_id,
        )
        if model is None:
            return None
        return self._to_domain(model)

    def get_by_instrument(
        self,
        instrument_id: UUID,
        limit: int = 50,
    ) -> list[IngestionRun]:
        statement = (
            select(IngestionRunModel)
            .where(IngestionRunModel.instrument_id == instrument_id)
            .order_by(IngestionRunModel.started_at.desc())
            .limit(limit)
        )
        models = self.session.scalars(statement).all()
        return [self._to_domain(model) for model in models]

    def get_latest_run(
        self,
        ingestion_type: str,
        instrument_id: UUID | None = None,
    ) -> IngestionRun | None:
        statement = (
            select(IngestionRunModel)
            .where(
                IngestionRunModel.ingestion_type == ingestion_type.strip().lower()
            )
        )
        if instrument_id is not None:
            statement = statement.where(
                IngestionRunModel.instrument_id == instrument_id
            )

        statement = statement.order_by(
            IngestionRunModel.started_at.desc()
        ).limit(1)

        model = self.session.scalar(statement)
        if model is None:
            return None
        return self._to_domain(model)

    def save(self, run: IngestionRun) -> None:
        model = self.session.get(
            IngestionRunModel,
            run.run_id,
        )
        if model is None:
            model = IngestionRunModel(
                run_id=run.run_id,
                ingestion_type=run.ingestion_type,
                instrument_id=run.instrument_id,
                provider=run.provider,
                started_at=run.started_at,
                completed_at=run.completed_at,
                status=run.status,
                received_count=run.received_count,
                inserted_count=run.inserted_count,
                updated_count=run.updated_count,
                skipped_count=run.skipped_count,
                error_message=run.error_message,
            )
            self.session.add(model)
        else:
            model.ingestion_type = run.ingestion_type
            model.instrument_id = run.instrument_id
            model.provider = run.provider
            model.started_at = run.started_at
            model.completed_at = run.completed_at
            model.status = run.status
            model.received_count = run.received_count
            model.inserted_count = run.inserted_count
            model.updated_count = run.updated_count
            model.skipped_count = run.skipped_count
            model.error_message = run.error_message

        self.session.flush()

    @staticmethod
    def _to_domain(model: IngestionRunModel) -> IngestionRun:
        return IngestionRun(
            run_id=model.run_id,
            ingestion_type=model.ingestion_type,
            instrument_id=model.instrument_id,
            provider=model.provider,
            started_at=model.started_at,
            completed_at=model.completed_at,
            status=model.status,
            received_count=model.received_count,
            inserted_count=model.inserted_count,
            updated_count=model.updated_count,
            skipped_count=model.skipped_count,
            error_message=model.error_message,
        )
