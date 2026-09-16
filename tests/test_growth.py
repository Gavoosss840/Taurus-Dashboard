"""
Tests du modèle de valorisation en deux étages.

Une perpétuité à taux unique sous-valorise mécaniquement toute société
croissant plus vite que ce taux. Alphabet devait croître à 7 % perpétuels pour
justifier son cours, et ressortait donc décoté de 65 % sous un taux uniforme
de 2,5 % — une zone d'achat 69 % sous le marché, inexploitable.
"""

import math

import pytest

from taurus_core.capital_structure import (
    _solve_initial_growth,
    _two_stage_value,
    initial_growth,
    mm_valuation,
)
from taurus_core.config import ValuationConfig
from taurus_core.providers.fundamentals import revenue_growth

CFG = ValuationConfig()


# --------------------------------------------------------------------------- #
#  Croissance de départ                                                        #
# --------------------------------------------------------------------------- #

def test_the_company_growth_is_used():
    assert initial_growth({"revenue_cagr": 0.08}, CFG) == pytest.approx(0.08)


def test_growth_is_capped_at_an_economic_maximum():
    """Aucune société ne croît à 40 % pendant dix ans."""
    assert initial_growth({"revenue_cagr": 0.40}, CFG) == CFG.max_initial_growth


def test_a_declining_business_has_a_floor():
    assert initial_growth({"revenue_cagr": -0.50}, CFG) == CFG.min_initial_growth


def test_a_missing_history_falls_back_to_the_default():
    assert initial_growth({}, CFG) == CFG.default_initial_growth
    assert initial_growth({"revenue_cagr": float("nan")}, CFG) == CFG.default_initial_growth
    assert initial_growth({"revenue_cagr": "n/d"}, CFG) == CFG.default_initial_growth


# --------------------------------------------------------------------------- #
#  Valorisation en deux étages                                                 #
# --------------------------------------------------------------------------- #

def test_it_matches_a_perpetuity_when_growth_is_flat():
    """Sans convergence à opérer, les deux modèles doivent coïncider."""
    nopat, rate, growth = 100.0, 0.09, 0.025
    two_stage = _two_stage_value(nopat, rate, growth, growth, 10)
    perpetuity = nopat * (1 + growth) / (rate - growth)
    assert two_stage == pytest.approx(perpetuity, rel=1e-9)


def test_growth_may_exceed_the_discount_rate_in_the_explicit_stage():
    """La contrainte g < r ne porte que sur la perpétuité terminale.

    Sur un étage fini, une croissance supérieure au taux d'actualisation est
    le cas normal d'une société en expansion. L'y plafonner ramenait Alphabet
    de 16,7 % à 9,3 % et annulait l'essentiel de la correction.
    """
    value = _two_stage_value(100.0, 0.09, 0.15, 0.025, 10)
    assert math.isfinite(value) and value > 0


def test_more_growth_is_worth_more():
    low = _two_stage_value(100.0, 0.09, 0.02, 0.025, 10)
    high = _two_stage_value(100.0, 0.09, 0.12, 0.025, 10)
    assert high > low


def test_a_higher_discount_rate_is_worth_less():
    cheap = _two_stage_value(100.0, 0.07, 0.08, 0.025, 10)
    dear = _two_stage_value(100.0, 0.12, 0.08, 0.025, 10)
    assert cheap > dear


def test_a_diverging_terminal_rate_is_refused():
    assert math.isnan(_two_stage_value(100.0, 0.05, 0.08, 0.06, 10))


def test_a_growth_company_is_worth_more_than_under_a_flat_perpetuity():
    """C'est la correction recherchée, mesurée."""
    nopat, rate, terminal = 100.0, 0.0975, 0.025
    flat = nopat * (1 + terminal) / (rate - terminal)
    staged = _two_stage_value(nopat, rate, 0.15, terminal, 10)
    assert staged > flat * 1.3


# --------------------------------------------------------------------------- #
#  Croissance implicite du cours                                               #
# --------------------------------------------------------------------------- #

def test_the_implied_growth_reproduces_the_target():
    """Par construction : à ce taux, le modèle rend exactement la cible."""
    nopat, rate, terminal, years = 100.0, 0.09, 0.025, 10
    target = _two_stage_value(nopat, rate, 0.11, terminal, years)

    implied = _solve_initial_growth(nopat, rate, terminal, years, target)
    assert implied == pytest.approx(0.11, abs=1e-4)
    assert _two_stage_value(nopat, rate, implied, terminal, years) == pytest.approx(
        target, rel=1e-4
    )


def test_an_unjustifiable_price_returns_nothing():
    """Un cours qu'aucune croissance plausible n'explique."""
    assert math.isnan(_solve_initial_growth(100.0, 0.09, 0.025, 10, 1e15))
    assert math.isnan(_solve_initial_growth(100.0, 0.09, 0.025, 10, 1.0))


def test_no_implied_growth_without_earnings():
    assert math.isnan(_solve_initial_growth(-50.0, 0.09, 0.025, 10, 1000.0))


# --------------------------------------------------------------------------- #
#  Effet sur la valorisation complète                                          #
# --------------------------------------------------------------------------- #

def company(growth: float) -> dict:
    return {
        "total_debt": 1.0e10, "total_equity": 8.0e10, "cash": 2.0e10,
        "ebit": 1.2e10, "interest_expense": 4.0e8, "tax_rate": 0.21,
        "fcf": 9.0e9, "sector": "Information Technology",
        "revenue_cagr": growth, "revenue_years": 8,
    }


def test_a_faster_grower_is_valued_higher():
    slow = mm_valuation(company(0.02), 2.0e11, 0.25, 1.0e9, 1.0, CFG)
    fast = mm_valuation(company(0.14), 2.0e11, 0.25, 1.0e9, 1.0, CFG)
    assert slow is not None and fast is not None
    assert fast.fair_equity_value > slow.fair_equity_value
    assert fast.divergence_pct > slow.divergence_pct


def test_the_result_carries_both_growth_rates():
    result = mm_valuation(company(0.09), 2.0e11, 0.25, 1.0e9, 1.0, CFG)
    assert result is not None
    assert result.growth_start == pytest.approx(0.09)
    assert result.growth_rate == pytest.approx(CFG.terminal_growth)


def test_the_sensitivity_grid_varies_the_initial_growth():
    """C'est elle qui distingue les sociétés ; le taux terminal leur est commun."""
    result = mm_valuation(company(0.09), 2.0e11, 0.25, 1.0e9, 1.0, CFG)
    assert result is not None
    growths = {cell["growth_rate"] for cell in result.sensitivity_grid}
    assert len(growths) >= 3
    assert max(growths) > CFG.terminal_growth


# --------------------------------------------------------------------------- #
#  Extraction de l'historique                                                  #
# --------------------------------------------------------------------------- #

def annual_book(values: dict, concept: str = "Revenues") -> dict:
    return {
        concept: {"units": {"USD": [
            {"start": f"{year}-01-01", "end": f"{year}-12-31",
             "val": value, "filed": f"{year + 1}-02-01"}
            for year, value in values.items()
        ]}}
    }


def test_revenue_growth_is_measured_over_the_filings():
    book = annual_book({y: 100.0 * (1.10 ** (y - 2018)) for y in range(2018, 2026)})
    cagr, years = revenue_growth(book)
    assert cagr == pytest.approx(0.10, abs=0.005)
    assert years == 8


def test_endpoints_are_smoothed():
    """Un exercice d'arrivée exceptionnel décalerait le taux de plusieurs points."""
    values = {y: 100.0 for y in range(2018, 2026)}
    values[2025] = 400.0
    cagr, _ = revenue_growth(annual_book(values))

    # Le taux naïf, pris sur les seules extrémités, dépasse 21 % ; le lissage
    # sur deux exercices le ramène nettement en dessous.
    naive = (400.0 / 100.0) ** (1 / 7) - 1
    assert cagr < naive * 0.80


def test_a_short_history_yields_nothing():
    cagr, years = revenue_growth(annual_book({2024: 100.0, 2025: 110.0}))
    assert math.isnan(cagr)
    assert years == 0


def test_negative_revenue_is_refused():
    cagr, _ = revenue_growth(annual_book({y: -100.0 for y in range(2018, 2026)}))
    assert math.isnan(cagr)
