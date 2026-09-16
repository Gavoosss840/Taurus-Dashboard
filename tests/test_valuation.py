"""Tests de l'orchestrateur : agrégation des piliers et verdict.

Les fournisseurs de données sont remplacés par des doublures : ces tests ne
touchent jamais le réseau et restent donc reproductibles.
"""

import math

import numpy as np
import pandas as pd
import pytest

from taurus_core import valuation
from taurus_core.config import ValuationConfig
from taurus_core.providers.fundamentals import Fundamentals
from taurus_core.providers.prices import PriceHistory
from taurus_core.providers.quotes import Quote
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

# Les facteurs synthétiques présentent un mois très volatil qui déclenche le
# régime de krach : la moitié du poids du momentum bascule alors sur l'alpha,
# ce qui change les seuils et rend certains scénarios inatteignables. Les tests
# qui portent sur la zone d'achat le neutralisent pour isoler ce qu'ils mesurent.
CFG_CALM = ValuationConfig(momentum_crash_dampen=False)


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


def build_quote(market_cap: float = 1.0e11, **overrides) -> Quote:
    data = {
        "market_cap": market_cap,
        "currency": "USD",
        "sector": "Industrials",
        "source": "doublure",
    }
    data.update(overrides)
    return Quote(**data)


@pytest.fixture
def stub_providers(monkeypatch):
    """Installe des fournisseurs déterministes et renvoie leur état mutable.

    Couvre toute la chaîne — cours, facteurs, fondamentaux, cotation, change —
    pour qu'aucun test ne touche le réseau.
    """
    factors = build_factors()
    prices = build_prices(factors)
    state = {
        "prices": prices,
        "factors": factors,
        "fundamentals": build_fundamentals(),
        # Capitalisation cohérente avec la doublure de cours et d'actions.
        "quote": build_quote(prices.last_price * 1.0e9),
    }
    monkeypatch.setattr(
        valuation.prices_provider, "get_monthly_prices",
        lambda ticker, cfg=CFG, failures=None: state["prices"],
    )
    monkeypatch.setattr(
        valuation.factors_provider, "get_ff5_factors",
        lambda region="north_america", cfg=CFG: state["factors"],
    )
    monkeypatch.setattr(
        valuation.fundamentals_provider, "get_fundamentals",
        lambda ticker, cfg=CFG: state["fundamentals"],
    )
    monkeypatch.setattr(
        valuation.quotes_provider, "get_quote",
        lambda ticker, cfg=CFG: state["quote"],
    )
    return state


# ── Validation du ticker ─────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("aapl", "AAPL"), ("  msft ", "MSFT"), ("brk-b", "BRK-B"), ("BRK.B", "BRK.B"),
    # Places locales : le suffixe fait partie du ticker.
    ("mc.pa", "MC.PA"), ("7203.t", "7203.T"), ("0700.hk", "0700.HK"),
    ("reliance.ns", "RELIANCE.NS"),
])
def test_normalise_ticker_accepts_valid_forms(raw, expected):
    assert normalise_ticker(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "@@@", "A B", "A" * 17, "-ABC"])
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


def test_market_cap_comes_from_the_quote_provider(stub_providers):
    """La capitalisation vient du titre coté, pas de « actions × cours ».

    Un ADR Toyota vaut dix actions ordinaires : reconstituer la capitalisation
    à partir du nombre d'actions publié à la SEC la multiplierait par dix.
    """
    result = analyze("TEST", CFG)
    assert result.market_cap == pytest.approx(stub_providers["quote"].market_cap)


def test_market_cap_falls_back_to_shares_times_price(stub_providers):
    """À défaut de fournisseur de cotation, la reconstitution reprend la main."""
    stub_providers["quote"] = None
    result = analyze("TEST", CFG)
    expected = (
        stub_providers["fundamentals"].shares_outstanding
        * stub_providers["prices"].last_price
    )
    assert result.market_cap == pytest.approx(expected)


def test_filed_sic_sector_wins_over_the_quote_provider(stub_providers):
    """Nasdaq classe Altria, cigarettier, en « Health Care ».

    Le secteur pilote le taux de destruction en faillite du modèle
    Modigliani-Miller — 35 % pour la santé contre 20 % pour la consommation de
    base — l'erreur n'est donc pas cosmétique. Le code SIC déposé à la SEC est
    stable et auditable : il prime.
    """
    stub_providers["fundamentals"] = build_fundamentals(sector="Consumer Staples")
    stub_providers["quote"] = build_quote(1.0e11, sector="Health Care")
    assert analyze("TEST", CFG).sector == "Consumer Staples"


def test_quote_sector_fills_in_when_the_sic_is_unknown(stub_providers):
    stub_providers["fundamentals"] = build_fundamentals(sector="Unknown")
    stub_providers["quote"] = build_quote(1.0e11, sector="Information Technology")
    assert analyze("TEST", CFG).sector == "Information Technology"


def test_region_defaults_to_north_america(stub_providers):
    result = analyze("TEST", CFG)
    assert result.region == "north_america"


def test_region_follows_the_ticker_suffix(stub_providers):
    """Un suffixe de place détermine la région sans ambiguïté."""
    assert analyze("MC.PA", CFG).region == "europe"
    assert analyze("7203.T", CFG).region == "japan"
    assert analyze("0700.HK", CFG).region == "asia_pacific"


def test_region_follows_the_filing_country(stub_providers):
    stub_providers["fundamentals"] = build_fundamentals(country="Japan")
    assert analyze("TEST", CFG).region == "japan"


# ── Devises ──────────────────────────────────────────────────────────────

def test_market_cap_is_displayed_in_the_quote_currency(stub_providers):
    """La capitalisation s'affiche à côté du cours : même devise que lui.

    L'écran Modigliani-Miller raisonne dans la devise des COMPTES — ASML
    publie en euros — mais montrer 544 milliards d'euros sous un symbole
    dollar induirait en erreur.
    """
    from taurus_core.providers import fx as fx_module

    stub_providers["fundamentals"] = build_fundamentals(currency="EUR")
    stub_providers["quote"] = build_quote(1.0e11, currency="USD")
    # 1 USD = 0,90 EUR, donc 1 EUR = 1,111 USD.
    rates = {("USD", "EUR"): 0.90, ("EUR", "USD"): 1.0 / 0.90}
    original = fx_module.latest_rate
    fx_module.latest_rate = lambda a, b, cfg=None: rates.get((a.upper(), b.upper()))
    try:
        result = analyze("TEST", CFG)
    finally:
        fx_module.latest_rate = original

    assert result.currency == "USD"
    # Affichée telle quelle : la cotation est déjà en dollars.
    assert result.market_cap == pytest.approx(1.0e11)


def test_valuation_converts_accounts_to_the_market_cap_currency(stub_providers):
    """Comptes et capitalisation doivent être rapprochés dans une seule devise.

    Sans conversion, la divergence mesurerait la parité de change, pas une
    décote.
    """
    from taurus_core.providers import fx as fx_module

    baseline = analyze("TEST", CFG)

    # Mêmes chiffres, mais déclarés en euros avec une capitalisation en
    # dollars : le rapprochement doit appliquer le taux, donc changer la
    # divergence.
    stub_providers["fundamentals"] = build_fundamentals(currency="EUR")
    rates = {("USD", "EUR"): 0.50, ("EUR", "USD"): 2.0}
    original = fx_module.latest_rate
    fx_module.latest_rate = lambda a, b, cfg=None: rates.get((a.upper(), b.upper()))
    try:
        converted = analyze("TEST", CFG)
    finally:
        fx_module.latest_rate = original

    base_mm = next(p for p in baseline.pillars if p.key == "capital_structure")
    conv_mm = next(p for p in converted.pillars if p.key == "capital_structure")
    assert base_mm.available and conv_mm.available
    assert conv_mm.details["divergence_pct"] != pytest.approx(
        base_mm.details["divergence_pct"]
    )


def test_missing_exchange_rate_neutralises_the_mm_pillar(stub_providers):
    """Le dollar de Taïwan n'est pas publié par la BCE.

    Supposer la parité comparerait une capitalisation en dollars à des comptes
    en TWD — un facteur trente, silencieusement.
    """
    from taurus_core.providers import fx as fx_module

    stub_providers["fundamentals"] = build_fundamentals(currency="TWD")
    original = fx_module.latest_rate
    fx_module.latest_rate = lambda a, b, cfg=None: None
    try:
        result = analyze("TEST", CFG)
    finally:
        fx_module.latest_rate = original

    mm = next(p for p in result.pillars if p.key == "capital_structure")
    assert not mm.available
    assert any("change" in w.lower() for w in result.warnings)


# ── Formulation du pilier alpha ──────────────────────────────────────────

def alpha_pillar_for(tstat: float, n_obs: int = 60):
    """Construit le pilier alpha pour un t-stat donné."""
    from taurus_core.alpha import AlphaResult
    from taurus_core.valuation import _build_alpha_pillar

    result = AlphaResult(
        alpha_monthly=0.01, alpha_annual=0.122, alpha_tstat=tstat,
        alpha_stderr=0.01 / max(abs(tstat), 1e-9), t_critical=2.0,
        p_value=0.25, r_squared=0.49, n_obs=n_obs,
        betas={"Mkt-RF": 1.0}, window_start="2021-01-31", window_end="2025-12-31",
    )
    return _build_alpha_pillar(result, CFG)


def test_a_non_significant_alpha_is_not_called_proven():
    """Un t de 1,17 penche sans démontrer.

    Le verdict du pilier suit le score (t / seuil = 0,58, donc au-dessus de
    0,5), mais l'annoncer « sous-évaluée » tout en écrivant « pas
    significatif » était contradictoire à l'écran.
    """
    pillar = alpha_pillar_for(1.17)
    assert pillar.verdict == VERDICT_UNDERVALUED     # le score le justifie
    assert "n'atteint pas le seuil" in pillar.explanation
    assert "penche" in pillar.explanation
    # Et surtout : plus d'affirmation de significativité.
    assert "est statistiquement significatif" not in pillar.explanation


def test_a_significant_alpha_says_so():
    pillar = alpha_pillar_for(2.60)
    assert "statistiquement significatif" in pillar.explanation


def test_a_weak_alpha_is_plainly_noise():
    pillar = alpha_pillar_for(0.38)
    assert pillar.verdict == "NEUTRE"
    assert "distinguer du bruit" in pillar.explanation


def test_months_needed_follows_the_square_root_law():
    """Le t-stat croît comme la racine du nombre d'observations.

    Passer de 1,17 à 2,00 demande de multiplier T par (2,00/1,17)² ≈ 2,9.
    """
    pillar = alpha_pillar_for(1.17, n_obs=60)
    needed = pillar.details["months_for_significance"]
    assert needed == round(60 * (2.0 / 1.17) ** 2)
    assert 165 <= needed <= 180
    assert "ans" in pillar.explanation


def test_no_months_estimate_when_already_significant():
    assert alpha_pillar_for(2.60).details["months_for_significance"] is None


def test_r_squared_is_always_reported():
    for tstat in (0.38, 1.17, 2.60):
        assert "variance des rendements" in alpha_pillar_for(tstat).explanation


# ── Rendement total reconstitué ──────────────────────────────────────────

def dividend_history(prices, quarterly_yield: float = 0.015):
    quarters = prices.index[2::3]
    return pd.Series(prices.reindex(quarters) * quarterly_yield, index=quarters)


def test_price_only_source_triggers_reconstruction(stub_providers):
    """Sans dividendes, l'alpha d'une valeur de rendement est faux de ~0,9 point."""
    history = stub_providers["prices"]
    stub_providers["prices"] = PriceHistory(
        history.monthly, source="Nasdaq Data",
        last_price=history.last_price, total_return=False,
    )
    stub_providers["fundamentals"] = build_fundamentals(
        dividends_per_share=dividend_history(history.monthly),
    )
    result = analyze("TEST", CFG)

    assert result.data_sources.get("dividendes") == "SEC EDGAR (reconstitués)"
    assert any("reconstitués" in w for w in result.warnings)


def test_reconstruction_raises_the_measured_alpha(stub_providers):
    history = stub_providers["prices"]
    price_only = PriceHistory(
        history.monthly, source="Nasdaq Data",
        last_price=history.last_price, total_return=False,
    )

    stub_providers["prices"] = price_only
    stub_providers["fundamentals"] = build_fundamentals()          # sans dividendes
    without = analyze("TEST", CFG)

    stub_providers["fundamentals"] = build_fundamentals(
        dividends_per_share=dividend_history(history.monthly),
    )
    with_dividends = analyze("TEST", CFG)

    alpha_before = next(p for p in without.pillars if p.key == "alpha")
    alpha_after = next(p for p in with_dividends.pillars if p.key == "alpha")
    assert alpha_after.details["alpha_annual"] > alpha_before.details["alpha_annual"]


def test_reconstruction_leaves_price_and_market_cap_alone(stub_providers):
    """Le cours affiché et la capitalisation restent sur la base des cours.

    Seules la régression et le momentum travaillent sur l'indice de rendement
    total : un indice n'est pas un prix de marché.
    """
    history = stub_providers["prices"]
    stub_providers["prices"] = PriceHistory(
        history.monthly, source="Nasdaq Data",
        last_price=history.last_price, total_return=False,
    )
    stub_providers["fundamentals"] = build_fundamentals(
        dividends_per_share=dividend_history(history.monthly),
    )
    result = analyze("TEST", CFG)

    assert result.price == pytest.approx(history.last_price)
    assert result.market_cap == pytest.approx(stub_providers["quote"].market_cap)


def test_warning_survives_when_reconstruction_is_impossible(stub_providers):
    """Exxon ne publie que deux trimestres : l'avertissement doit rester."""
    history = stub_providers["prices"]
    stub_providers["prices"] = PriceHistory(
        history.monthly, source="Nasdaq Data",
        last_price=history.last_price, total_return=False,
    )
    stub_providers["fundamentals"] = build_fundamentals(
        dividends_per_share=dividend_history(history.monthly).iloc[:3],
    )
    result = analyze("TEST", CFG)

    assert "dividendes" not in result.data_sources
    assert any("sous-estimés" in w for w in result.warnings)


def test_total_return_source_is_left_untouched(stub_providers):
    """Yahoo avec `adjclose` fournit déjà un rendement total."""
    result = analyze("TEST", CFG)
    assert "dividendes" not in result.data_sources
    assert not any("dividende" in w.lower() for w in result.warnings)


# ── Message d'échec : la cause, pas une accusation ───────────────────────

def test_a_local_venue_failure_does_not_blame_the_ticker(stub_providers, monkeypatch):
    """« MC.PA » est correct : c'est la source qui manque, pas la saisie.

    Renvoyer « vérifiez le ticker » envoie corriger une saisie qui n'a rien à
    se reprocher, et masque la seule action utile — cotation américaine ou clé
    d'API.
    """
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    stub_providers["prices"] = None
    monkeypatch.setattr(
        valuation.prices_provider, "get_monthly_prices",
        lambda ticker, cfg=CFG, failures=None: None,
    )

    with pytest.raises(TickerError) as excinfo:
        analyze("MC.PA", CFG)

    message = str(excinfo.value)
    assert "place locale" in message
    assert "Vérifiez l'orthographe" not in message
    assert "Financial Modeling Prep" in message


def test_the_message_notes_a_configured_key(stub_providers, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "clé-de-test")
    monkeypatch.setattr(
        valuation.prices_provider, "get_monthly_prices",
        lambda ticker, cfg=CFG, failures=None: None,
    )

    with pytest.raises(TickerError) as excinfo:
        analyze("MC.PA", CFG)

    # Inutile de signaler l'absence d'une clé qui est là.
    assert "Aucune clé Financial Modeling Prep n'est configurée" not in str(excinfo.value)


def test_an_unknown_ticker_is_named_as_such(stub_providers, monkeypatch):
    """Quand chaque source répond mais qu'aucune ne connaît le titre."""
    def nothing_found(ticker, cfg=CFG, failures=None):
        if failures is not None:
            failures.update({name: "aucune donnée exploitable"
                             for name in ("yahoo", "nasdaq")})
        return None

    monkeypatch.setattr(valuation.prices_provider, "get_monthly_prices", nothing_found)

    with pytest.raises(TickerError) as excinfo:
        analyze("ZZZQQQ", CFG)

    assert "orthographe" in str(excinfo.value)


def test_a_transient_failure_points_at_the_diagnostic(stub_providers, monkeypatch):
    """Sources injoignables : ni le ticker ni la place ne sont en cause."""
    def unreachable(ticker, cfg=CFG, failures=None):
        if failures is not None:
            failures.update({"yahoo": "ConnectionError", "nasdaq": "Timeout"})
        return None

    monkeypatch.setattr(valuation.prices_provider, "get_monthly_prices", unreachable)

    with pytest.raises(TickerError) as excinfo:
        analyze("AAPL", CFG)

    assert "Diagnostic des sources" in str(excinfo.value)


# ── Zone d'achat : le seuil du verdict traduit en cours ──────────────────

def test_the_buy_price_makes_the_composite_cross_the_threshold(stub_providers):
    """À ce cours, le score composite doit valoir exactement le seuil.

    C'est la propriété qui définit la zone : annoncer un prix d'achat que le
    modèle ne confirmerait pas à ce prix-là serait pire que ne rien annoncer.
    """
    # Le pilier Modigliani-Miller pèse 0,30 et sature à ±2 : il ne peut
    # apporter que 0,60 au composite. Il faut donc que les deux autres piliers
    # ne s'y opposent pas, sans quoi aucun cours ne franchit le seuil.
    stub_providers["prices"] = build_prices(
        stub_providers["factors"], alpha=0.012, beta=1.0,
    )
    result = analyze("TEST", CFG_CALM)
    assert math.isfinite(result.buy_below)

    # On rejoue l'analyse en plaçant le cours à la borne annoncée.
    history = stub_providers["prices"]
    factor = result.buy_below / history.last_price
    stub_providers["prices"] = PriceHistory(
        history.monthly * factor, source=history.source,
        last_price=result.buy_below, total_return=history.total_return,
    )
    stub_providers["quote"] = build_quote(
        stub_providers["quote"].market_cap * factor,
    )
    at_threshold = analyze("TEST", CFG_CALM)

    assert at_threshold.composite_score == pytest.approx(
        CFG_CALM.verdict_threshold, abs=0.05
    )


def test_a_cheaper_price_moves_the_verdict_towards_undervalued(stub_providers):
    baseline = analyze("TEST", CFG)
    stub_providers["quote"] = build_quote(stub_providers["quote"].market_cap * 0.5)
    cheaper = analyze("TEST", CFG)
    assert cheaper.composite_score > baseline.composite_score


def test_the_band_uses_every_pillar_not_just_valuation(stub_providers):
    """Le seuil doit bouger quand l'alpha ou le momentum changent.

    Une zone calculée sur le seul pilier Modigliani-Miller serait insensible
    aux deux autres, et ne répondrait donc pas à la question posée : « à quel
    prix ce titre devient-il une opportunité selon l'ENSEMBLE du modèle ? »
    """
    stub_providers["prices"] = build_prices(
        stub_providers["factors"], alpha=0.006, beta=1.0,
    )
    weak = analyze("TEST", CFG_CALM).buy_below

    # Un alpha plus fort relève le seuil : il reste moins de chemin à
    # parcourir depuis le cours pour faire basculer le composite.
    stub_providers["prices"] = build_prices(
        stub_providers["factors"], alpha=0.012, beta=1.0,
    )
    strong = analyze("TEST", CFG_CALM).buy_below

    assert math.isfinite(weak) and math.isfinite(strong)
    assert strong > weak


def test_an_unreachable_threshold_is_reported_as_such(stub_providers):
    """Quand les autres piliers s'y opposent, aucun cours ne suffit.

    Le pilier Modigliani-Miller est borné à ±2 : passé cette saturation, une
    décote supplémentaire n'ajoute plus rien au score. Inventer un prix serait
    trompeur.
    """
    from taurus_core.valuation import _price_for_score, _pillar_weights

    pillars = [
        Pillar(key="alpha", name="alpha", score=-2.0, weight=0.40,
               available=True, headline="", verdict="", explanation=""),
        Pillar(key="capital_structure", name="mm", score=0.0, weight=0.30,
               available=True, headline="", verdict="", explanation=""),
        Pillar(key="momentum", name="mom", score=-2.0, weight=0.30,
               available=True, headline="", verdict="", explanation=""),
    ]
    weights = _pillar_weights(pillars, False, CFG)
    price = _price_for_score(
        CFG.verdict_threshold, pillars, weights, lambda p: 100.0, 50.0, CFG,
    )
    assert math.isnan(price)


def test_no_band_without_the_valuation_pillar(stub_providers):
    """Sans fondamentaux, aucun pilier ne dépend du cours.

    C'est le cas d'une cotation locale hors périmètre SEC, sans clé d'API :
    le verdict tient sur l'alpha et le momentum, qu'un prix hypothétique
    aujourd'hui ne change pas.
    """
    stub_providers["fundamentals"] = None
    result = analyze("TEST", CFG)

    assert not next(p for p in result.pillars if p.key == "capital_structure").available
    assert math.isnan(result.buy_below)
    assert math.isnan(result.sell_above)
