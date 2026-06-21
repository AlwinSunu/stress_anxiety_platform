"""Database engine, session factory, and the FastAPI session dependency.

SQLite is used for local development (zero setup). The implementation is
Postgres-ready: set ``DATABASE_URL`` (e.g. ``postgresql+psycopg://...``) and
nothing else changes. The SQLite-only ``check_same_thread`` connect arg is
applied conditionally so it does not break other backends.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Default to a file-based SQLite DB in the backend/ directory.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./stress.db")

# check_same_thread is a SQLite-specific quirk: FastAPI serves requests from a
# threadpool, and SQLite otherwise refuses connections shared across threads.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""


def get_db():
    """Yield a database session and guarantee it is closed (FastAPI dependency)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
