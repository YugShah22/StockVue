from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

import app.infrastructure.database.registry  # noqa: F401 — registers all ORM models
from app.core.config import settings
from app.infrastructure.database.base import Base


@pytest.fixture(scope="session")
def db_engine():
    """Create a test engine and build all tables once per session."""
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
    )
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def db_session(db_engine) -> Generator[Session, None, None]:
    """
    Provide a transactional session that rolls back after each test.
    Uses SQLAlchemy 2.x-compatible Session(bind=...) via join_transaction_mode.
    """
    connection = db_engine.connect()
    transaction = connection.begin()

    session = Session(bind=connection, join_transaction_mode="create_savepoint")

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
