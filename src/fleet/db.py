"""Database engine, session factory and FastAPI dependency."""

from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from fleet import config

DATABASE_URL = os.getenv("DATABASE_URL", config.DATABASE_URL)
_CONNECT_ARGS = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_CONNECT_ARGS, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """Declarative base shared by every model."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a short-lived session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Create tables. Importing models registers them on Base.metadata."""
    from fleet import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
