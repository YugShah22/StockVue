"""
Model registry — import all ORM models here so that SQLAlchemy's mapper
registers every table before Base.metadata is used (e.g. Alembic autogenerate,
create_all in tests, etc.).

WHY a separate file instead of base.py:
  base.py defines `Base`.
  Every model file does `from app.infrastructure.database.base import Base`.
  If base.py also imported the model files, Python would hit a circular import:
    base.py → models/company.py → base.py (💥 partially initialised)

USAGE — import this module anywhere that needs all tables visible:
  • alembic/env.py  (before target_metadata = Base.metadata)
  • tests/conftest.py  (before create_all / drop_all)

DO NOT remove these imports. Ruff may flag them as F401/E402 — they are
intentional side-effect imports and are suppressed via per-file-ignores.
"""

# noqa: F401, E402 applied via pyproject.toml per-file-ignores for this file.
from app.infrastructure.database.models.company import CompanyModel  # noqa: F401
from app.infrastructure.database.models.exchange import ExchangeModel  # noqa: F401
from app.infrastructure.database.models.instrument import InstrumentModel  # noqa: F401
from app.infrastructure.database.models.instrument_history import (  # noqa: F401
    InstrumentHistoryModel,
)
from app.infrastructure.database.models.market_bar import MarketBarModel  # noqa: F401
from app.infrastructure.database.models.market_holiday import (  # noqa: F401
    MarketHolidayModel,
)

__all__ = [
    "CompanyModel",
    "ExchangeModel",
    "InstrumentModel",
    "InstrumentHistoryModel",
    "MarketBarModel",
    "MarketHolidayModel",
]
