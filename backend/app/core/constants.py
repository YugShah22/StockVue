"""
AI Multi-Factor Stock Intelligence & Portfolio Decision-Support System
backend/app/core/constants.py

System-wide named constants.
Use these instead of magic strings or numbers anywhere in the codebase.
"""

import zoneinfo

# ---------------------------------------------------------------------------
# Timezones
# ---------------------------------------------------------------------------

UTC = zoneinfo.ZoneInfo("UTC")
IST = zoneinfo.ZoneInfo("Asia/Kolkata")  # Indian Standard Time (NSE / BSE)

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

API_V1_PREFIX = "/api/v1"
HEALTH_ENDPOINT = "/health"

# ---------------------------------------------------------------------------
# Data quality states (string form — canonical enum lives in domain/enums/)
# ---------------------------------------------------------------------------

QUALITY_VALID = "VALID"
QUALITY_WARNING = "WARNING"
QUALITY_STALE = "STALE"
QUALITY_CONFLICTED = "CONFLICTED"
QUALITY_QUARANTINED = "QUARANTINED"
QUALITY_UNAVAILABLE = "UNAVAILABLE"

# ---------------------------------------------------------------------------
# Prediction strength levels (string form — canonical enum in domain/enums/)
# ---------------------------------------------------------------------------

STRENGTH_VERY_STRONG = "VERY_STRONG"
STRENGTH_STRONG = "STRONG"
STRENGTH_MODERATE = "MODERATE"
STRENGTH_DEVELOPING = "DEVELOPING"
STRENGTH_LIMITED = "LIMITED"

# ---------------------------------------------------------------------------
# Numeric defaults
# ---------------------------------------------------------------------------

DEFAULT_LOOKBACK_DAYS: int = 252  # ~1 trading year
DEFAULT_ROLLING_WINDOW: int = 20  # ~1 trading month
MAX_POSITION_WEIGHT: float = 0.20  # 20% max single position (default constraint)
