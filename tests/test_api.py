"""Tests de l'API HTTP et de la sérialisation JSON."""

import math

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.schemas import analysis_to_dict, safe_float
from taurus_core import valuation

from test_valuation import build_factors, build_fundamentals, build_prices, build_quote

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def stub_engine(monkeypatch):
    """Fournisseurs déterministes, pour que l'API ne touche pas le réseau."""
    factors = build_factors()
    prices = build_prices(factors)
    monkeypatch.setattr(
        valuation.prices_provider, "get_monthly_prices",
        lambda ticker, cfg=None: prices,
    )
    monkeypatch.setattr(
        valuation.factors_provider, "get_ff5_factors",
        lambda region="north_america", cfg=None: factors,
    )
    monkeypatch.setattr(
        valuation.fundamentals_provider, "get_fundamentals",
        lambda ticker, cfg=None: build_fundamentals(),
    )
    monkeypatch.setattr(
        valuation.quotes_provider, "get_quote",
        lambda ticker, cfg=None: build_quote(prices.last_price * 1.0e9),
    )


# ── Routes de service ────────────────────────────────────────────────────

def test_health_reports_ok():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_config_exposes_engine_weights():
    payload = client.get("/api/config").json()
    weights = payload["weights"]
    assert weights["alpha"] + weights["capital_structure"] + weights["momentum"] \
        == pytest.approx(1.0)
    assert payload["lookback_months"] == 60


def test_index_serves_the_dashboard():
    response = client.get("/")
    assert response.status_code == 200
    assert "Taurus Dashboard" in response.text


def test_static_assets_are_served():
    for path in ("/styles.css", "/app.js"):
        assert client.get(path).status_code == 200


# ── Analyse ──────────────────────────────────────────────────────────────

def test_analyze_returns_the_full_payload(stub_engine):
    payload = client.get("/api/analyze/TEST").json()

    assert payload["ticker"] == "TEST"
    assert payload["verdict"] in ("SOUS-ÉVALUÉE", "SUR-ÉVALUÉE", "AU JUSTE PRIX")
    assert len(payload["pillars"]) == 3
    for field in ("composite_score", "confidence", "price", "summary",
                  "data_sources", "computed_at", "region", "region_label",
                  "currency"):
        assert field in payload


def test_analyze_lowercase_ticker_is_normalised(stub_engine):
    assert client.get("/api/analyze/test").json()["ticker"] == "TEST"


def test_malformed_ticker_is_a_bad_request():
    """La saisie est en cause, pas la disponibilité des données."""
    response = client.get("/api/analyze/@@@")
    assert response.status_code == 400
    assert "error" in response.json()


def test_unknown_ticker_is_not_found(monkeypatch):
    monkeypatch.setattr(
        valuation.prices_provider, "get_monthly_prices", lambda ticker, cfg=None: None,
    )
    response = client.get("/api/analyze/ZZZZ")
    assert response.status_code == 404
    assert "error" in response.json()


def test_internal_failure_returns_500(monkeypatch):
    def explode(ticker, cfg=None):
        raise RuntimeError("panne du fournisseur")

    monkeypatch.setattr(valuation.prices_provider, "get_monthly_prices", explode)
    response = client.get("/api/analyze/TEST")
    assert response.status_code == 500
    # Le détail technique ne doit pas fuiter vers le client.
    assert "panne du fournisseur" not in response.text


def test_cache_can_be_cleared():
    response = client.post("/api/cache/clear")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── Sérialisation ────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (1.5, 1.5), (0, 0.0), (float("nan"), None),
    (float("inf"), None), (float("-inf"), None), ("abc", None), (None, None),
])
def test_safe_float_neutralises_non_finite_values(value, expected):
    assert safe_float(value) == expected


def test_payload_contains_no_non_finite_floats(stub_engine):
    """NaN et ±∞ ne sont pas représentables en JSON strict.

    Les laisser passer produirait un corps que `JSON.parse` refuse côté
    navigateur, et le dashboard afficherait une erreur réseau trompeuse.
    """
    import json

    response = client.get("/api/analyze/TEST")
    body = response.text
    assert "NaN" not in body
    assert "Infinity" not in body

    # Contrôle structurel : plus aucun flottant non fini dans l'arbre.
    def walk(node):
        if isinstance(node, float):
            assert math.isfinite(node)
        elif isinstance(node, dict):
            for item in node.values():
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(json.loads(body))


def test_analysis_to_dict_handles_missing_values(stub_engine):
    """Une analyse dégradée doit rester sérialisable."""
    from taurus_core.config import DEFAULT_CONFIG

    analysis = valuation.analyze("TEST", DEFAULT_CONFIG)
    analysis.fair_value = float("nan")
    analysis.upside_pct = float("inf")

    payload = analysis_to_dict(analysis)
    assert payload["fair_value"] is None
    assert payload["upside_pct"] is None
