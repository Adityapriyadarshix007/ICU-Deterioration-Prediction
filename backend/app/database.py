"""
Database connection and setup for MIMIC-IV
UPDATED: 0-6h Features → 6-18h Outcome (NO DATA LEAKAGE)
"""

import os
import duckdb
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from app.config import (
    DATA_DIR, ICU_TABLES, HOSP_TABLES, DB_NAME, LOG_LEVEL
)


class DatabaseManager:
    """Manages DuckDB connection and table views for MIMIC-IV"""

    def __init__(self, db_path: Optional[str] = None, memory: bool = False):
        self.db_path = db_path or DB_NAME
        self.memory = memory
        self.connection = None
        self._connect()

    def _connect(self):
        if self.memory:
            self.connection = duckdb.connect(database=':memory:')
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self.connection = duckdb.connect(database=self.db_path)

        self.connection.execute("PRAGMA threads=4")
        self.connection.execute("PRAGMA memory_limit='4GB'")
        print(f"✅ Connected to DuckDB: {'memory' if self.memory else self.db_path}")

    def close(self):
        if self.connection:
            self.connection.close()
            print("✅ Database connection closed")

    # ------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------
    def __enter__(self):
        """Support `with DatabaseManager() as db:` syntax."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically close the connection on exit."""
        self.close()
        return False   # don't suppress exceptions

    # ------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------
    def execute(self, query: str) -> duckdb.DuckDBPyRelation:
        return self.connection.execute(query)

    def fetch_df(self, query: str) -> pd.DataFrame:
        return self.connection.execute(query).fetchdf()

    # ------------------------------------------------------------
    # View creation
    # ------------------------------------------------------------
    def create_views(self, data_dir: Optional[str] = None):
        """
        Create DuckDB views for all MIMIC-IV tables.
        Reports missing files explicitly at the end.
        """
        data_dir = data_dir or DATA_DIR
        print("📊 Creating database views...")

        missing = []

        for table in ICU_TABLES:
            file_path = Path(data_dir) / "icu" / f"{table}.csv.gz"
            if file_path.exists():
                self.connection.execute(f"""
                CREATE OR REPLACE VIEW {table} AS
                SELECT * FROM read_csv_auto('{file_path}')
                """)
                print(f"  ✅ Created view: {table}")
            else:
                print(f"  ⚠️ File not found: {file_path}")
                missing.append(table)

        for table in HOSP_TABLES:
            file_path = Path(data_dir) / "hosp" / f"{table}.csv.gz"
            if file_path.exists():
                self.connection.execute(f"""
                CREATE OR REPLACE VIEW {table} AS
                SELECT * FROM read_csv_auto('{file_path}')
                """)
                print(f"  ✅ Created view: {table}")
            else:
                print(f"  ⚠️ File not found: {file_path}")
                missing.append(table)

        # Report accurately based on what actually happened
        if missing:
            print(f"⚠️ {len(missing)} view(s) missing: {missing}")
        else:
            print("✅ All views created successfully")

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------
    def get_table_counts(self) -> Dict[str, int]:
        counts = {}
        tables = ICU_TABLES + HOSP_TABLES

        for table in tables:
            try:
                result = self.connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()
                counts[table] = result[0] if result else 0
            except Exception as e:
                print(f"⚠️ Could not count {table}: {e}")
                counts[table] = 0

        return counts

    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        try:
            return self.connection.execute(
                f"DESCRIBE {table_name}"
            ).fetchdf()
        except Exception as e:
            print(f"⚠️ Could not get schema for {table_name}: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------
    def vacuum(self):
        self.connection.execute("VACUUM")
        print("✅ Database vacuumed")

# ============================================================
# MongoDB client (used by auth routes)
# This is a real MongoDB singleton, separate from the DuckDB DatabaseManager.
# ============================================================
try:
    from pymongo import MongoClient
    _PYMONGO_AVAILABLE = True
except ImportError:
    MongoClient = None
    _PYMONGO_AVAILABLE = False


class MongoDatabase:
    """MongoDB singleton for auth routes (uses settings.MONGODB_URL)."""
    _client = None
    _db = None

    @classmethod
    def connect(cls):
        if not _PYMONGO_AVAILABLE:
            raise RuntimeError("pymongo is not installed")
        if cls._client is None:
            from app.config_api import settings as _s
            cls._client = MongoClient(
                _s.MONGODB_URL,
                serverSelectionTimeoutMS=5000,
                uuidRepresentation="standard",
            )
            cls._db = cls._client[_s.MONGODB_DB_NAME]
        return cls._db

    @classmethod
    def get_db(cls):
        if cls._db is None:
            cls.connect()
        return cls._db

    @classmethod
    def close(cls):
        if cls._client is not None:
            cls._client.close()
            cls._client = None
            cls._db = None


# ============================================================
# Alias for backwards compatibility
# The app's shim will overwrite `Database` with MongoDatabase
# at startup. This default lets imports succeed regardless.
# ============================================================
Database = MongoDatabase
