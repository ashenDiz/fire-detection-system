"""
Pytest configuration, test database isolation, and dependency overrides.
Guarantees database/fire_system.db is NEVER touched or modified during test runs.
"""

import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.database.session import Base, get_db
from app.main import app

# Dedicated test database path
TEST_DB_DIR = os.path.dirname(__file__)
TEST_DB_PATH = os.path.abspath(os.path.join(TEST_DB_DIR, "test_fire_system.db"))
TEST_DATABASE_URL = f"sqlite:///{TEST_DB_PATH}"

# Test engine and session maker
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Session-wide fixture ensuring strict test database isolation:
    1. Safety assertions confirming production/dev DATABASE_URL is not used.
    2. Tables created exclusively in test_fire_system.db.
    3. FastAPI app get_db dependency overridden to use TestingSessionLocal.
    4. Cleanup on teardown.
    """
    # Strict safety assertions
    prod_path = os.path.abspath(settings.DATABASE_URL.replace("sqlite:///", ""))
    assert TEST_DATABASE_URL != settings.DATABASE_URL, "CRITICAL: Test database URL must not equal production/development URL!"
    assert TEST_DB_PATH != prod_path, "CRITICAL: Test database file path must not equal production database path!"

    # Create tables in dedicated test database
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)

    # Dependency override for FastAPI test requests
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    yield

    # Teardown
    app.dependency_overrides.clear()
    test_engine.dispose()
    import gc
    gc.collect()
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass


@pytest.fixture
def db_session():
    """Fixture providing an isolated database session for unit tests."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
