from dataclasses import dataclass
from uuid import UUID


@dataclass
class Company:
    company_id: UUID
    name: str
    legal_name: str
    country: str

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.legal_name = self.legal_name.strip()
        self.country = self.country.strip()

        if not self.name:
            raise ValueError("Company name cannot be empty")

        if not self.legal_name:
            raise ValueError("Company legal name cannot be empty")

        if not self.country:
            raise ValueError("Company country cannot be empty")
