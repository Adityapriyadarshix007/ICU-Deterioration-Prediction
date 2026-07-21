from .users import router as users_router
from .patients import router as patients_router
from .analytics import router as analytics_router
from .settings import router as settings_router
from .logs import router as logs_router

__all__ = [
    'users_router',
    'patients_router',
    'analytics_router',
    'settings_router',
    'logs_router'
]
