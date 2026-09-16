"""
Tests du diagnostic des sources.

Le diagnostic existe parce qu'un repli silencieux est indiscernable : quota
atteint, ticker inconnu, réseau coupé et clé absente produisent tous « prix :
Nasdaq Data » et appellent des réponses opposées.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from taurus_core import diagnostics
from taurus_core.providers import http

client = TestClient(app, raise_server_exceptions=False)


def make_probe(status=None, error="", attempts=1) -> http.Probe:
    return http.Probe(
        name="test", url="https://example.invalid", status=status,
        latency_ms=12, error=error, excerpt="", attempts=attempts,
    )


# --------------------------------------------------------------------------- #
#  Sondage                                                                     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("status,expected", [
    (200, True), (204, True), (301, False), (404, False), (429, False), (500, False),
])
def test_probe_ok_only_for_success_codes(status, expected):
    assert make_probe(status).ok is expected


def test_a_probe_that_could_not_connect_is_not_ok():
    assert make_probe(None, error="ConnectionError").ok is False


def test_probe_never_raises(monkeypatch):
    """Un diagnostic qui plante n'en est pas un."""
    def explode(*args, **kwargs):
        raise OSError("réseau coupé")

    monkeypatch.setattr(http.session(), "get", explode)
    result = http.probe("test", "https://example.invalid")
    assert result.ok is False
    assert result.error == "OSError"


def test_probe_retries_a_throttled_source(monkeypatch):
    calls = {"n": 0}

    class Response:
        status_code = 429
        text = "Too Many Requests"

    def throttled(*args, **kwargs):
        calls["n"] += 1
        return Response()

    monkeypatch.setattr(http.session(), "get", throttled)
    monkeypatch.setattr(http._time, "sleep", lambda s: None)
    result = http.probe("test", "https://example.invalid", attempts=3)

    assert calls["n"] == 3
    assert result.attempts == 3
    assert result.status == 429


def test_probe_stops_early_on_a_definitive_answer(monkeypatch):
    """Un 404 ne s'améliore pas en réessayant."""
    calls = {"n": 0}

    class Response:
        status_code = 404
        text = "Not Found"

    def missing(*args, **kwargs):
        calls["n"] += 1
        return Response()

    monkeypatch.setattr(http.session(), "get", missing)
    http.probe("test", "https://example.invalid", attempts=3)
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
#  Interprétation                                                              #
# --------------------------------------------------------------------------- #

def test_throttling_is_named_as_such():
    advice = diagnostics._interpretation("Yahoo", make_probe(429))
    assert "quota" in advice.lower()
    assert "Financial Modeling Prep" in advice


def test_unknown_ticker_is_distinguished_from_an_outage():
    assert "ticker" in diagnostics._interpretation("X", make_probe(404)).lower()
    assert "panne" in diagnostics._interpretation("X", make_probe(503)).lower()


def test_a_refused_key_is_distinguished_from_a_missing_one():
    refused = diagnostics._interpretation("X", make_probe(401))
    absent = diagnostics._interpretation("X", make_probe(None), has_key=False)
    assert "refusé" in refused.lower()
    assert "normal" in absent.lower()


def test_no_connection_points_at_the_network():
    advice = diagnostics._interpretation("X", make_probe(None, error="ConnectionError"))
    assert "réseau" in advice.lower()


def test_a_working_source_says_so():
    assert diagnostics._interpretation("X", make_probe(200)) == "Disponible."


# --------------------------------------------------------------------------- #
#  Rapport et route                                                            #
# --------------------------------------------------------------------------- #

@pytest.fixture
def stub_probes(monkeypatch):
    """Toutes les sondes répondent, sans réseau."""
    monkeypatch.setattr(http, "probe",
                        lambda name, url, **kw: make_probe(200))
    monkeypatch.setattr(diagnostics.http, "probe",
                        lambda name, url, **kw: make_probe(200))
    monkeypatch.setattr(diagnostics, "ticker_to_cik", lambda t, cfg=None: "0000320193")


def test_report_covers_every_source(stub_probes):
    report = diagnostics.run("AAPL")
    names = {c["name"] for c in report["checks"]}
    assert {"Yahoo Finance", "Nasdaq Data", "SEC EDGAR",
            "Kenneth R. French Data Library",
            "Banque centrale européenne"} <= names


def test_report_flags_essential_sources(stub_probes):
    report = diagnostics.run("AAPL")
    assert report["summary"]["essential_ok"] is True
    assert report["summary"]["price_sources_total"] >= 2


def test_missing_key_is_reported_without_a_network_call(stub_probes, monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    report = diagnostics.run("AAPL")
    fmp = next(c for c in report["checks"] if c["name"] == "Financial Modeling Prep")
    assert fmp["attempts"] == 0
    assert "facultatif" in fmp["interpretation"].lower()


def test_endpoint_returns_the_report(stub_probes):
    response = client.get("/api/diagnostics?ticker=VZ")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "VZ"
    assert payload["checks"]


def test_endpoint_defaults_to_a_known_ticker(stub_probes):
    assert client.get("/api/diagnostics").json()["ticker"] == "AAPL"
