"""FXMacroData macro-event features."""

from __future__ import annotations

import json
import os
from typing import Any, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

FXMACRODATA_BASE_URL = "https://fxmacrodata.com/api/v1"


def load_release_calendar(
    currency: str = "usd",
    *,
    limit: int = 100,
    min_tier: Optional[int] = 2,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Load FXMacroData release-calendar events for alpha feature research."""

    limit_count = max(1, min(int(limit), 100))
    params: dict[str, str] = {"limit": str(limit_count)}
    token = api_key or os.getenv("FXMACRODATA_API_KEY")
    if token:
        params["api_key"] = token

    url = f"{FXMACRODATA_BASE_URL}/calendar/{currency.lower()}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "midas-alpha-fxmacrodata/1.0"})
    with urlopen(request, timeout=20) as response:
        payload = json.load(response)

    events: list[dict[str, Any]] = payload.get("data", [])
    if min_tier is not None:
        events = [
            event
            for event in events
            if int(event.get("market_tier") or 99) <= min_tier
        ]

    frame = pd.DataFrame(events[:limit_count])
    if not frame.empty and "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame
