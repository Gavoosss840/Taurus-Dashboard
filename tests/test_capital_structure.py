"""Tests du pilier Modigliani-Miller."""

import math

import pytest

from taurus_core.capital_structure import (
    SECTOR_DISTRESS_RATE,
    credit_spread,
    distress_rate,
    mm_valuation,
    unlever_beta,
    unlevered_cost_of_capital,
)
from taurus_core.config import ValuationConfig

CFG = ValuationConfig()


def sound_company(**overrides):
    """Société rentable et peu endettée, servant de base aux variantes."""
    base = {
        "total_debt": 2.0e10,
        "total_equity": 5.0e10,
        "cash": 1.0e10,
        "ebit": 1.2e10,
        "interest_expense": 8.0e8,
        "tax_rate": 0.21,
        "fcf": 9.0e9,
        "sector": "Industrials",
    }
    base.update(overrides)
    return base


# ── Dé-leviérisation et coût du capital ──────────────────────────────────

def test_unlever_beta_reduces_beta_when_debt_present():
    levered = unlever_beta(1.2, total_debt=5.0e10, equity_value=1.0e11, tax_rate=0.21)
    assert levered < 1.2
    # β_U = 1,2 / (1 + 0,79 × 0,5) = 0,8622
    assert levered == pytest.approx(1.2 / (1 + 0.79 * 0.5), rel=1e-9)


def test_unlever_beta_is_identity_without_debt():
    assert unlever_beta(1.1, 0.0, 1.0e11, 0.21) == pytest.approx(1.1)


def test_unlever_beta_undefined_without_equity():
    assert math.isnan(unlever_beta(1.1, 1.0e10, 0.0, 0.21))


def test_cost_of_capital_floors_negative_beta():
    """Un bêta négatif donnerait un taux inférieur au taux sans risque."""
    rate = unlevered_cost_of_capital(-0.5, CFG)
    assert rate > CFG.risk_free_rate_annual


def test_cost_of_capital_uses_default_beta_when_missing():
    assert unlevered_cost_of_capital(float("nan"), CFG) == pytest.approx(
        CFG.risk_free_rate_annual + CFG.default_unlevered_beta * CFG.equity_risk_premium
    )


# ── Spread de crédit et coûts de détresse ────────────────────────────────

def test_credit_spread_increases_with_leverage():
    assert credit_spread(0.2, CFG) < credit_spread(1.0, CFG) < credit_spread(3.0, CFG)


def test_credit_spread_is_capped():
    assert credit_spread(50.0, CFG) <= 0.10


def test_distress_rate_is_sector_specific():
    """Le logiciel détruit plus de valeur en faillite qu'un réseau électrique."""
    assert distress_rate("Information Technology", CFG) > distress_rate("Utilities", CFG)
    assert distress_rate("secteur inexistant", CFG) == SECTOR_DISTRESS_RATE["Unknown"]


def test_distress_rate_flat_when_disabled():
    flat = ValuationConfig(industry_distress_costs=False)
    assert distress_rate("Information Technology", flat) == 0.20


# ── Valorisation ─────────────────────────────────────────────────────────

def test_valuation_is_independent_of_market_price():
    """Le correctif central : la juste valeur ne doit pas suivre le cours.

    Le modèle d'origine posait V_U = capitalisation + dette nette − bouclier,
    ce qui faisait mécaniquement coller la juste valeur au cours et rendait
    la divergence toujours négative ou nulle.
    """
    cheap = mm_valuation(sound_company(), 1.0e11, 0.25, 1.0e9, 1.0, CFG)
    rich = mm_valuation(sound_company(), 3.0e11, 0.25, 1.0e9, 1.0, CFG)

    assert cheap is not None and rich is not None

    # Il subsiste un couplage résiduel : la dé-leviérisation de Hamada prend
    # D/E en valeur de marché. Il doit rester d'un ordre de grandeur inférieur
    # à la variation du cours — ici, moins de 10 % pour un cours triplé.
    drift = abs(rich.unlevered_value / cheap.unlevered_value - 1.0)
    assert drift < 0.10, f"couplage résiduel trop fort : {drift:.1%}"

    # La juste valeur des capitaux propres, elle, bouge à peine…
    equity_drift = abs(rich.fair_equity_value / cheap.fair_equity_value - 1.0)
    assert equity_drift < 0.10, f"la juste valeur suit le cours : {equity_drift:.1%}"

    # …tandis que la divergence, qui compare cette valeur au cours, bascule.
    assert cheap.divergence_pct > 0 > rich.divergence_pct


def test_undervaluation_is_reachable():
    """Le modèle d'origine ne pouvait jamais signaler une sous-évaluation."""
    result = mm_valuation(sound_company(), 5.0e10, 0.25, 1.0e9, 0.9, CFG)
    assert result is not None
    assert result.divergence_pct > CFG.leverage_gap_threshold * 100
    assert result.undervalued is True


def test_overvaluation_is_reachable():
    result = mm_valuation(sound_company(), 1.0e12, 0.25, 1.0e9, 1.1, CFG)
    assert result is not None
    assert result.overvalued is True


def test_negative_ebit_yields_no_valuation():
    """Une perpétuité de flux négatifs n'a pas de sens économique."""
    assert mm_valuation(sound_company(ebit=-1.0e9), 1.0e11, 0.3, 1.0e9, 1.0, CFG) is None


def test_zero_market_cap_yields_no_valuation():
    assert mm_valuation(sound_company(), 0.0, 0.3, 1.0e9, 1.0, CFG) is None


def test_negative_book_equity_does_not_explode_leverage():
    """Les rachats d'actions massifs donnent des capitaux propres négatifs.

    Diviser la dette par ces capitaux propres produirait un levier de l'ordre
    de la dette en valeur absolue, donc des coûts d'agence délirants.
    """
    result = mm_valuation(
        sound_company(total_equity=-5.0e9), 1.0e11, 0.25, 1.0e9, 1.0, CFG,
    )
    assert result is not None
    assert result.leverage_ratio <= 10.0
    assert any("capitaux propres" in note.lower() for note in result.notes)


def test_interest_is_imputed_when_unreported():
    result = mm_valuation(
        sound_company(interest_expense=0.0), 1.0e11, 0.25, 1.0e9, 1.0, CFG,
    )
    assert result is not None
    assert result.interest_imputed is True
    assert result.pv_tax_shield > 0


def test_distress_costs_grow_with_volatility():
    calm = mm_valuation(sound_company(), 1.0e11, 0.15, 1.0e9, 1.0, CFG)
    wild = mm_valuation(sound_company(), 1.0e11, 0.90, 1.0e9, 1.0, CFG)
    assert calm is not None and wild is not None
    assert wild.prob_default > calm.prob_default
    assert wild.pv_distress > calm.pv_distress


def test_fair_value_per_share_matches_equity_value():
    shares = 2.5e9
    result = mm_valuation(sound_company(), 1.0e11, 0.25, shares, 1.0, CFG)
    assert result is not None
    assert result.fair_value_per_share == pytest.approx(
        result.fair_equity_value / shares
    )


def test_sensitivity_grid_is_monotonic():
    """Une croissance plus forte, ou un taux plus bas, valorise davantage."""
    result = mm_valuation(sound_company(), 1.0e11, 0.25, 1.0e9, 1.0, CFG)
    assert result is not None
    grid = result.sensitivity_grid
    assert len(grid) >= 6

    by_key = {(c["discount_rate"], c["growth_rate"]): c["equity_value"] for c in grid}
    rates = sorted({r for r, _ in by_key})
    growths = sorted({g for _, g in by_key})

    for rate in rates:
        values = [by_key[(rate, g)] for g in growths if (rate, g) in by_key]
        assert values == sorted(values), "la valeur doit croître avec g"

    for growth in growths:
        values = [by_key[(r, growth)] for r in rates if (r, growth) in by_key]
        assert values == sorted(values, reverse=True), "la valeur doit décroître avec r"
