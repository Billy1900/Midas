"""Offline tests for the FXMacroData release-calendar loader."""

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from midas import fxmacrodata
from midas.fxmacrodata import FXMacroDataError, load_release_calendar

EVENTS = [
    {
        "release": "inflation",
        "announcement_datetime_utc": "2026-10-14T12:30:00+00:00",
        "date": "2026-09-30",
        "market_tier": 1,
    },
    {
        "release": "trade_balance",
        "announcement_datetime_utc": "2026-10-06T12:30:00+00:00",
        "market_tier": 2,
    },
    {"release": "junk_tier", "market_tier": "1"},
]


class _FakeOpener:
    def __init__(self, body):
        self.body = body
        self.requests = []

    def open(self, request, timeout=None):
        self.requests.append(request)
        if isinstance(self.body, Exception):
            raise self.body
        raw = self.body if isinstance(self.body, bytes) else json.dumps(self.body).encode()
        return io.BytesIO(raw)


def _install(monkeypatch, body):
    opener = _FakeOpener(body)
    monkeypatch.setattr(fxmacrodata, "_OPENER", opener)
    monkeypatch.delenv("FXMACRODATA_API_KEY", raising=False)
    return opener


def test_key_goes_in_header_not_url(monkeypatch):
    opener = _install(monkeypatch, {"data": EVENTS})
    load_release_calendar("USD", api_key="test-key")

    request = opener.requests[0]
    assert request.get_header("X-api-key") == "test-key"
    assert "api_key" not in request.full_url
    assert request.full_url.startswith("https://api.fxmacrodata.com/v1/calendar/usd?")


def test_release_time_and_tier_filter(monkeypatch):
    _install(monkeypatch, {"data": EVENTS})
    frame = load_release_calendar(min_tier=2)

    assert list(frame["release"]) == ["inflation", "trade_balance"]
    assert str(frame["release_time"].iloc[0]) == "2026-10-14 12:30:00+00:00"


@pytest.mark.parametrize("bad", [" secret key", "secret\nkey", "sec ret"])
def test_malformed_key_is_not_echoed(monkeypatch, bad):
    opener = _install(monkeypatch, {"data": EVENTS})
    with pytest.raises(FXMacroDataError) as exc:
        load_release_calendar(api_key=bad)
    assert "secret" not in str(exc.value)
    assert opener.requests == []


def test_error_body_with_200(monkeypatch):
    _install(monkeypatch, {"detail": "Unsupported currency"})
    with pytest.raises(FXMacroDataError, match="Unsupported currency"):
        load_release_calendar("xyz")


@pytest.mark.parametrize("body", [b"<html>", ["a", "list"], {"data": "nope"}])
def test_malformed_payload(monkeypatch, body):
    _install(monkeypatch, body)
    with pytest.raises(FXMacroDataError):
        load_release_calendar()


def test_redirects_are_not_followed():
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append((self.path, self.headers.get("X-API-Key")))
            if self.path == "/start":
                self.send_response(302)
                self.send_header("Location", "/elsewhere")
                self.end_headers()
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/start"
        with pytest.raises(HTTPError) as exc:
            fxmacrodata._OPENER.open(Request(url, headers={"X-API-Key": "secret"}), timeout=5)
    finally:
        server.shutdown()

    assert exc.value.code == 302
    assert "secret" not in str(exc.value)
    assert [path for path, _ in hits] == ["/start"]
