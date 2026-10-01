# Models are registered with SQLAlchemy via app.infrastructure.database.base.
# Do NOT import models here — it creates a circular import:
#   models/__init__.py → models/company.py → base.py → models/company.py (💥)
#
# Import individual models directly from their modules when needed, e.g.:
#   from app.infrastructure.database.models.company import CompanyModel
