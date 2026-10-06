from abc import ABC, abstractmethod
from uuid import UUID

from app.domain.entities.ingestion_run import IngestionRun


class IngestionRunRepository(ABC):
    @abstractmethod
    def get_by_id(
        self,
        run_id: UUID,
    ) -> IngestionRun | None:
        raise NotImplementedError

    @abstractmethod
    def get_by_instrument(
        self,
        instrument_id: UUID,
        limit: int = 50,
    ) -> list[IngestionRun]:
        raise NotImplementedError

    @abstractmethod
    def get_latest_run(
        self,
        ingestion_type: str,
        instrument_id: UUID | None = None,
    ) -> IngestionRun | None:
        raise NotImplementedError

    @abstractmethod
    def save(
        self,
        run: IngestionRun,
    ) -> None:
        raise NotImplementedError
