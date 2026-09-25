"""
FastAPI application entry point.
Launched via: uvicorn app.main_api:app
"""
import sys
import logging
from pathlib import Path

# ------------------------------------------------------------------
# Ensure backend/app/ is on sys.path (flat imports work everywhere)
# ------------------------------------------------------------------
_THIS_DIR = str(Path(__file__).parent.absolute())
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from .config_api import settings
except ImportError:
    from config_api import settings

logger = logging.getLogger(__name__)


# ============================================================
# MongoDB connection (kept here so database.py stays untouched)
# ============================================================
try:
    from pymongo import MongoClient
    PYMONGO_AVAILABLE = True
except ImportError:
    MongoClient = None
    PYMONGO_AVAILABLE = False


class MongoDatabase:
    """MongoDB singleton, exposed as Database.get_db() for the routes."""
    _client = None
    _db = None

    @classmethod
    def connect(cls):
        if not PYMONGO_AVAILABLE:
            raise RuntimeError("pymongo is not installed")
        if cls._client is None:
            cls._client = MongoClient(
                settings.MONGODB_URL,
                serverSelectionTimeoutMS=5000,
                uuidRepresentation="standard",
            )
            cls._db = cls._client[settings.MONGODB_DB_NAME]
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


def _install_mongo_shim():
    """
    Install MongoDatabase as `Database` on the app.database and database
    modules BEFORE any routes import them. This works regardless of
    whether Python resolves `database` flat or as `app.database`.
    """
    try:
        import app.database as db_module
        db_module.Database = MongoDatabase
    except Exception:
        pass
    try:
        import database as db_module
        db_module.Database = MongoDatabase
    except Exception:
        pass


# ============================================================
# FastAPI app factory
# ============================================================
def create_app() -> FastAPI:
    # 1) install MongoDB shim BEFORE importing routes
    _install_mongo_shim()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
    )

    origins = settings.allowed_origins_list
    logger.info(f"CORS allowed origins: {origins}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # 2) import routes (they will now find Database on the database module)
    from routes import auth, google, dashboard, patients, predictions
    from routes.admin import (
        users as admin_users,
        patients as admin_patients,
        logs as admin_logs,
        settings as admin_settings,
        analytics as admin_analytics,
    )

    # 3) mount routes with the exact prefixes the frontend calls
    app.include_router(auth.router,             prefix="/api/auth",      tags=["auth"])
    app.include_router(google.router,           prefix="/api/auth",      tags=["auth-google"])
    app.include_router(predictions.router,      prefix="/api/predict",   tags=["predict"])
    app.include_router(dashboard.router,        prefix="/api/dashboard", tags=["dashboard"])
    app.include_router(patients.router,         prefix="/api",           tags=["patients"])
    app.include_router(admin_patients.router,   prefix="/api/admin",     tags=["admin-patients"])
    app.include_router(admin_users.router,      prefix="/api/admin",     tags=["admin-users"])
    app.include_router(admin_logs.router,       prefix="/api/admin",     tags=["admin-logs"])
    app.include_router(admin_settings.router,   prefix="/api/admin",     tags=["admin-settings"])
    app.include_router(admin_analytics.router,  prefix="/api/admin",     tags=["admin-analytics"])

    @app.get("/health")
    def health():
        return {"status": "ok", "service": settings.APP_NAME}

    @app.get("/api/health")
    def api_health():
        return {"status": "ok"}

    @app.on_event("startup")
    async def startup_event():
        logger.info("Backend starting up...")
        try:
            MongoDatabase.get_db()
            logger.info("MongoDB connection OK")
        except Exception as e:
            logger.error(f"MongoDB connection failed at startup: {e}")
        try:
            from ml_model import ml_model
            _ = ml_model.is_loaded          # triggers lazy load
            if ml_model.model is not None:
                logger.info("Phase 6 model loaded successfully")
            else:
                logger.warning("Phase 6 model NOT loaded — predictions will be stubs")
        except Exception as e:
            logger.error(f"ML model load failed: {e}")
            import traceback
            traceback.print_exc()

    @app.on_event("shutdown")
    async def shutdown_event():
        try:
            MongoDatabase.close()
        except Exception:
            pass

    return app


app = create_app()
