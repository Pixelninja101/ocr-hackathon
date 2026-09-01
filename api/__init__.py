"""
FastAPI REST API package for Document Verification & Risk Assessment Engine.
"""

from api.config import api_config
from api.main import app
from api.rate_limiter import rate_limiter

__all__ = ["app", "api_config", "rate_limiter"]
