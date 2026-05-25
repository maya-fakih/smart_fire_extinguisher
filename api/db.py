# api/db.py
"""Per-request Supabase/Postgres connection."""

import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import g

logger = logging.getLogger(__name__)


def get_db():
    if "db" not in g:
        g.db = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", 5432),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            cursor_factory=RealDictCursor,
        )
    return g.db


def close_db(error=None):
    db = g.pop("db", None)
    if db is not None:
        try:
            db.close()
        except Exception:
            pass


def init_app(app):
    app.teardown_appcontext(close_db)