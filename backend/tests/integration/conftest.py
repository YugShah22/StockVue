from typing import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
    )

    connection = engine.connect()
    transaction = connection.begin()

    session = Session(bind=connection)

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
        engine.dispose()