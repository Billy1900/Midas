"""FXMacroData macro-event features."""

from __future__ import annotations

import json
import os
from typing import Any, Optional
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pandas as pd

FXMACRODATA_BASE_URL = "https://api.fxmacrodata.com/v1"


class FXMacroDataError(RuntimeError):
    """Raised when FXMacroData returns something this loader cannot use."""


class _NoRedirect(HTTPRedirectHandler):
    # urllib copies custom headers onto the redirected request, so following
    # a redirect would send X-API-Key to whatever host it points at.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = build_opener(_NoRedirect)


def _clean_api_key(api_key: Optional[str]) -> Optional[str]:
    if not api_key:
        return None
    api_key = api_key.strip()
    if not api_key or any(ch.isspace() or ord(ch) < 32 for ch in api_key):
        # http.client echoes an invalid header value in its error message.
        raise FXMacroDataError(
            "FXMacroData API key is empty or contains whitespace or control characters"
        )
    return api_key


def _tier(event: dict[str, Any]) -> Optional[int]:
    tier = event.get("market_tier")
    if isinstance(tier, bool) or not isinstance(tier, int):
        return None
    return tier


def load_release_calendar(
    currency: str = "usd",
    *,
    limit: int = 100,
    min_tier: Optional[int] = 2,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Load FXMacroData release-calendar events for alpha feature research.

    ``release_time`` is the UTC time the release is published, which is the
    timestamp to align features on. ``date``, where present, is the reference
    period the release covers, so it is earlier than the release itself.
    """

    limit_count = max(1, min(int(limit), 100))
    params: dict[str, str] = {"limit": str(limit_count)}
    headers = {"User-Agent": "midas-alpha-fxmacrodata/1.0"}
    token = _clean_api_key(api_key or os.getenv("FXMACRODATA_API_KEY"))
    if token:
        headers["X-API-Key"] = token

    url = f"{FXMACRODATA_BASE_URL}/calendar/{currency.lower()}?{urlencode(params)}"
    request = Request(url, headers=headers)
    with _OPENER.open(request, timeout=20) as response:
        try:
            payload = json.load(response)
        except ValueError:
            raise FXMacroDataError("FXMacroData returned a response that is not JSON") from None

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        detail = payload.get("detail") if isinstance(payload, dict) else None
        message = "FXMacroData returned an unexpected response shape"
        if isinstance(detail, str):
            message += f": {detail}"
        raise FXMacroDataError(message)

    events = [event for event in payload["data"] if isinstance(event, dict)]
    if min_tier is not None:
        events = [
            event
            for event in events
            if _tier(event) is not None and _tier(event) <= min_tier
        ]

    frame = pd.DataFrame(events[:limit_count])
    if frame.empty:
        return frame
    if "announcement_datetime_utc" in frame.columns:
        frame["release_time"] = pd.to_datetime(
            frame["announcement_datetime_utc"], utc=True, errors="coerce"
        )
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame
