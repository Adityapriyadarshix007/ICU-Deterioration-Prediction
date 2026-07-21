from .auth import router as auth_router
from .predictions import router as predictions_router
from .dashboard import router as dashboard_router
from .google import router as google_router
from .admin import analytics_router, patients_router, settings_router, users_router

# Export all routers
__all__ = [
    'auth_router',
    'predictions_router',
    'dashboard_router',
    'google_router',
    'analytics_router',
    'patients_router',
    'settings_router',
    'users_router'
]