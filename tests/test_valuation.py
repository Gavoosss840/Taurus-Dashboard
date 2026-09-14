"""Tests de l'orchestrateur : agrégation des piliers et verdict.

Les fournisseurs de données sont remplacés par des doublures : ces tests ne
touchent jamais le réseau et restent donc reproductibles.
"""

import numpy as np
import pandas as pd
import pytest

from taurus_core import valuation
from taurus_core.config import ValuationConfig
from taurus_core.providers.fundamentals import Fundamentals
from taurus_core.providers.prices import PriceHistory
from taurus_core.valuation import (
    InvalidTickerError,
    Pillar,
    TickerError,
    VERDICT_FAIR,
    VERDICT_OVERVALUED,
    VERDICT_UNDERVALUED,
    _combine,
    _verdict_from_score,
    analyze,
    normalise_ticker,
)

CFG = ValuationConfig()


# ── Doublures ────────────────────────────────────────────────────────────

def build_factors(n_months: int = 84, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2019-01-31", periods=n_months, freq="ME")
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.006, 0.040, n_months),
            "SMB": rng.normal(0.001, 0.020, n_months),
            "HML": rng.normal(0.001, 0.024, n_months),
            "RMW": rng.normal(0.002, 0.017, n_months),
            "CMA": rng.normal(0.001, 0.015, n_months),
            "RF": np.full(n_months, 0.0035),
        },
        index=index,
    )


def build_prices(factors: pd.DataFrame, alpha: float = 0.002,
                 beta: float = 1.0, seed: int = 9) -> PriceHistory:
    rng = np.random.default_rng(seed)
    returns = (
        alpha + beta * factors["Mkt-RF"].values + factors["RF"].values
        + rng.normal(0.0, 0.02, len(factors))
    )
    prices = 100.0 * np.cumprod(1.0 + returns)
    series = pd.Series(prices, index=factors.index)
    return PriceHistory(series, source="doublure", last_price=float(prices[-1]))


def build_fundamentals(**overrides) -> Fundamentals:
    data = Fundamentals(ticker="TEST", source="doublure")
    data.company_name = "Société de test"
    data.sector = "Industrials"
    data.total_debt = 2.0e10
    data.total_equity = 5.0e10
    data.total_assets = 1.2e11
    data.cash = 1.0e10
    data.ebit = 1.2e10
    data.interest_expense = 8.0e8
    data.revenue = 6.0e10
    data.net_income = 8.0e9
    data.fcf = 9.0e9
    data.tax_rate = 0.21
    data.shares_outstanding = 1.0e9
    for key, value in overrides.items():
        setattr(data, key, value)
    return data


@pytest.fixture
def stub_providers(monkeypatch):
    """Installe des fournisseurs déterministes et renvoie leur état mutable."""
    factors = build_factors()
    state = {
        "prices": build_prices(factors),
        "factors": factors,
        "fundamentals": build_fundamentals(),
    }
    monkeypatch.setattr(
        valuation.prices_provider, "get_monthly_prices",
        lambda ticker, cfg=CFG: state["prices"],
    )
    monkeypatch.setattr(
        valuation.factors_provider, "get_ff5_factors",
        lambda cfg=CFG: state["factors"],
    )
    monkeypatch.setattr(
        valuation.fundamentals_provider, "get_fundamentals",
        lambda ticker, cfg=CFG: state["fundamentals"],
    )
    return state


# ── Validation du ticker ─────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("aapl", "AAPL"), ("  msft ", "MSFT"), ("brk-b", "BRK-B"), ("BRK.B", "BRK.B"),
])
def test_normalise_ticker_accepts_valid_forms(raw, expected):
    assert normalise_ticker(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "@@@", "A B", "TROPLONGTICKER", "-ABC"])
def test_normalise_ticker_rejects_invalid_forms(raw):
    with pytest.raises(InvalidTickerError):
        normalise_ticker(raw)


def test_invalid_ticker_is_a_ticker_error():
    """Le code appelant peut n'attraper que TickerError."""
    assert issubclass(InvalidTickerError, TickerError)


# ── Seuils du verdict ────────────────────────────────────────────────────

@pytest.mark.parametrize("score,expected", [
    (1.50, VERDICT_UNDERVALUED), (0.75, VERDICT_UNDERVALUED),
    (0.49, VERDICT_FAIR), (0.00, VERDICT_FAIR), (-0.49, VERDICT_FAIR),
    (-0.75, VERDICT_OVERVALUED), (-1.50, VERDICT_OVERVALUED),
])
def test_verdict_thresholds(score, expected):
    verdict, _ = _verdict_from_score(score, CFG)
    assert verdict == expected


def test_strong_qualifier_beyond_one():
    _, moderate = _verdict_from_score(0.70, CFG)
    _, strong = _verdict_from_score(1.40, CFG)
    assert "fortement" not in moderate.lower()
    assert "fortement" in strong.lower()


# ── Agrégation des piliers ───────────────────────────────────────────────

def make_pillar(key, score, weight, available=True):
    return Pillar(key=key, name=key, score=score, weight=weight,
                  available=available, headline="", verdict="", explanation="")


def test_combine_is_the_weighted_average():
    pillars = [
        make_pillar("alpha", 1.0, 0.40),
        make_pillar("capital_structure", -1.0, 0.30),
        make_pillar("momentum", 0.5, 0.30),
    ]
    assert _combine(pillars, False, CFG) == pytest.approx(0.4 - 0.3 + 0.15)


def test_missing_pillar_weight_is_redistributed():
    """Un signal absent n'est pas un signal neutre.

    Le compter comme un zéro diluerait mécaniquement les piliers disponibles
    vers le verdict « au juste prix ».
    """
    pillars = [
        make_pillar("alpha", 1.0, 0.40),
        make_pillar("capital_structure", 0.0, 0.30, available=False),
        make_pillar("momentum", 1.0, 0.30),
    ]
    # Sans redistribution : 0,40 + 0,30 = 0,70. Avec : 1,00.
    assert _combine(pillars, False, CFG) == pytest.approx(1.0)


def test_crash_regime_shifts_weight_from_momentum_to_alpha():
    """En régime de krach, la moitié du poids du momentum passe à l'alpha."""
    pillars = [
        make_pillar("alpha", 1.0, 0.40),
        make_pillar("capital_structure", 0.0, 0.30),
        make_pillar("momentum", -1.0, 0.30),
    ]
    calm = _combine(pillars, False, CFG)
    crash = _combine(pillars, True, CFG)
    # Le momentum étant négatif et l'alpha positif, l'amortissement remonte
    # le score : 0,40−0,30 = 0,10 devient 0,55−0,15 = 0,40.
    assert crash > calm
    assert crash == pytest.approx(0.55 - 0.15)


def test_combine_returns_zero_without_any_pillar():
    pillars = [make_pillar("alpha", 1.0, 0.40, available=False)]
    assert _combine(pillars, False, CFG) == 0.0


# ── Analyse complète ─────────────────────────────────────────────────────

def test_analyze_returns_three_pillars(stub_providers):
    result = analyze("TEST", CFG)
    assert [p.key for p in result.pillars] == [
        "alpha", "capital_structure", "momentum",
    ]
    assert result.ticker == "TEST"
    assert result.company_name == "Société de test"


def test_analyze_is_undervalued_when_cheap(stub_providers):
    """Société très rentable dont le cours progresse : décote sur les trois piliers.

    Le verdict d'ensemble exige un accord entre piliers : le score de chacun
    étant borné à ±2 et le pilier Modigliani-Miller ne pesant que 30 %, il ne
    peut pas à lui seul franchir le seuil si les deux autres le contredisent.
    """
    stub_providers["fundamentals"] = build_fundamentals(ebit=4.0e10)
    stub_providers["prices"] = build_prices(
        stub_providers["factors"], alpha=0.010, beta=1.0,
    )
    result = analyze("TEST", CFG)

    assert result.verdict == VERDICT_UNDERVALUED
    assert result.upside_pct > 0


def test_one_saturated_pillar_cannot_carry_the_verdict(stub_providers):
    """Un seul pilier au maximum ne suffit pas à emporter le verdict.

    C'est la garantie recherchée : le score composite exige la concordance
    d'au moins deux piliers, faute de quoi une donnée comptable aberrante
    déclencherait à elle seule un signal d'achat.
    """
    stub_providers["fundamentals"] = build_fundamentals(ebit=4.0e10)
    result = analyze("TEST", CFG)

    mm = next(p for p in result.pillars if p.key == "capital_structure")
    assert mm.score == pytest.approx(CFG.score_clip)   # saturé
    others = [p for p in result.pillars if p.key != "capital_structure"]
    assert all(p.score < 0 for p in others)            # en sens inverse
    assert result.verdict == VERDICT_FAIR


def test_analyze_is_overvalued_when_expensive(stub_providers):
    """Résultat d'exploitation dérisoire au regard de la capitalisation."""
    stub_providers["fundamentals"] = build_fundamentals(ebit=6.0e8)
    result = analyze("TEST", CFG)
    assert result.verdict == VERDICT_OVERVALUED
    assert result.upside_pct < 0


def test_analyze_without_fundamentals_still_produces_a_verdict(stub_providers):
    """Le pilier MM manquant ne doit pas bloquer l'analyse."""
    stub_providers["fundamentals"] = None
    result = analyze("TEST", CFG)

    mm = next(p for p in result.pillars if p.key == "capital_structure")
    assert mm.available is False
    assert any("fondamentaux" in w.lower() for w in result.warnings)
    # Les deux autres piliers portent alors tout le poids.
    assert result.confidence < 1.0
    assert result.verdict in (VERDICT_UNDERVALUED, VERDICT_FAIR, VERDICT_OVERVALUED)


def test_analyze_without_prices_raises(stub_providers):
    stub_providers["prices"] = None
    with pytest.raises(TickerError):
        analyze("TEST", CFG)


def test_analyze_warns_when_dividends_are_excluded(stub_providers):
    history = stub_providers["prices"]
    stub_providers["prices"] = PriceHistory(
        history.monthly, source="Nasdaq Data",
        last_price=history.last_price, total_return=False,
    )
    result = analyze("TEST", CFG)
    assert any("dividende" in w.lower() for w in result.warnings)


def test_analyze_without_factors_degrades_gracefully(stub_providers):
    stub_providers["factors"] = None
    result = analyze("TEST", CFG)

    alpha = next(p for p in result.pillars if p.key == "alpha")
    momentum = next(p for p in result.pillars if p.key == "momentum")
    assert alpha.available is False
    assert momentum.available is False
    # Seul le pilier Modigliani-Miller subsiste.
    assert next(p for p in result.pillars if p.key == "capital_structure").available


def test_confidence_drops_with_warnings(stub_providers):
    clean = analyze("TEST", CFG).confidence
    stub_providers["fundamentals"] = build_fundamentals(
        warnings=["données incomplètes", "bilan périmé"],
    )
    noisy = analyze("TEST", CFG).confidence
    assert noisy < clean


def test_negative_ebit_neutralises_the_mm_pillar(stub_providers):
    stub_providers["fundamentals"] = build_fundamentals(ebit=-2.0e9)
    result = analyze("TEST", CFG)

    mm = next(p for p in result.pillars if p.key == "capital_structure")
    assert mm.available is False
    assert any("exploitation" in w.lower() for w in result.warnings)


def test_summary_reports_divergent_pillars(stub_providers):
    """Le désaccord entre piliers doit apparaître dans la synthèse."""
    result = analyze("TEST", CFG)
    under = [p for p in result.pillars if p.available and p.verdict == VERDICT_UNDERVALUED]
    over = [p for p in result.pillars if p.available and p.verdict == VERDICT_OVERVALUED]
    if under and over:
        assert "divergent" in result.summary.lower()


def test_market_cap_is_rebuilt_from_price_and_shares(stub_providers):
    result = analyze("TEST", CFG)
    expected = (
        stub_providers["fundamentals"].shares_outstanding
        * stub_providers["prices"].last_price
    )
    assert result.market_cap == pytest.approx(expected)
