from uuid import UUID

from sqlalchemy.orm import Session

from app.domain.entities.company import Company
from app.domain.repositories.company_repository import CompanyRepository
from app.infrastructure.database.models.company import CompanyModel


class PostgresCompanyRepository(CompanyRepository):
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_id(self, company_id: UUID) -> Company | None:
        model = self.session.get(CompanyModel, company_id)

        if model is None:
            return None

        return Company(
            company_id=model.company_id,
            name=model.name,
            legal_name=model.legal_name,
            country=model.country,
        )

    def save(self, company: Company) -> None:
        model = self.session.get(CompanyModel, company.company_id)

        if model is None:
            model = CompanyModel(
                company_id=company.company_id,
                name=company.name,
                legal_name=company.legal_name,
                country=company.country,
            )
            self.session.add(model)
        else:
            model.name = company.name
            model.legal_name = company.legal_name
            model.country = company.country

        self.session.flush()
