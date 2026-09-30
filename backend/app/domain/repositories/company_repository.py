from abc import ABC, abstractmethod
from uuid import UUID

from app.domain.entities.company import Company


class CompanyRepository(ABC):
    @abstractmethod
    def get_by_id(self, company_id: UUID) -> Company | None:
        raise NotImplementedError

    @abstractmethod
    def save(self, company: Company) -> None:
        raise NotImplementedError
