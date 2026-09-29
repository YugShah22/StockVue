"""
AI Multi-Factor Stock Intelligence & Portfolio Decision-Support System
backend/tests/conftest.py

Shared pytest fixtures available to all test categories.

Phase 1: minimal setup.
Later phases will add:
  - Database session fixtures (Phase 2)
  - Provider mock fixtures (Phase 3)
  - Feature store fixtures (Phase 4)
  - FastAPI TestClient fixtures (Phase 13)
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """
    Synchronous FastAPI test client.
    Use for API endpoint tests in tests/integration/ and tests/e2e/.
    """
    with TestClient(app) as c:
        yield c
